"""Official FIBO sampling inside Forge's shared processing/save lifecycle.

This is a version-bound compatibility engine, not a replacement process_images
loop. Its scheduler and conditioning belong to the official FIBO pipeline.
"""
from pathlib import Path
from types import MethodType, SimpleNamespace
import weakref
import torch

from .contract import validate_request

_current = None

def release_current():
    global _current
    current = _current() if _current else None
    if current:
        current.release()
    _current = None

class Fibo2Engine:
    is_fibo2 = True
    is_sd1 = is_sdxl = is_wan = is_inpaint = False
    tiling_enabled = use_distilled_cfg_scale = use_shift = False
    current_lora_hash = '[]'
    ini_latent = None

    def __init__(self, pipeline, manifest):
        global _current
        from backend.diffusion_engine.base import ForgeObjects
        from backend.patcher.clip import CLIP
        from backend.patcher.base import ModelPatcher
        from backend.patcher.unet import UnetPatcher
        from backend.modules.k_prediction import PredictionDiscreteFlow
        from backend import memory_management
        self.pipe = pipeline
        self.manifest = manifest
        self.ref_latents = []
        self.extra_generation_params = {}
        self.comments = []
        self.model_config = SimpleNamespace(model_type=SimpleNamespace(name='FLOW'), memory_usage_factor=16,
                                           sampling_settings={'shift':3.0, 'multiplier':1.0})
        clip = CLIP({'fibo2_qwen3vl_4b': pipeline.text_encoder}, {'fibo2_processor': pipeline.processor})
        # These patchers expose the actual owned modules to host callbacks. The
        # official pipeline owns placement during its compatibility sampling.
        model = TransformerBoundary(pipeline.transformer, memory_management.get_torch_device())
        unet = UnetPatcher.from_model(model=model, diffusers_scheduler=None, config=self.model_config,
                                     k_predictor=PredictionDiscreteFlow(self.model_config))
        vae = SimpleNamespace(first_stage_model=pipeline.vae, latent_channels=32, upscale_ratio=8,
                              patcher=ModelPatcher(pipeline.vae, load_device=memory_management.get_torch_device(), offload_device=torch.device('cpu')))
        self.forge_objects = ForgeObjects(unet, clip, vae, None)
        self.forge_objects_original = self.forge_objects.shallow_copy()
        self.forge_objects_after_applying_lora = self.forge_objects.shallow_copy()
        _current = weakref.ref(self)

    def release(self):
        if self.pipe is not None:
            from .runtime import release
            release(self.pipe)
            self.pipe = None

    def __del__(self):
        try:
            self.release()
        except Exception:
            pass

    def is_webui_legacy_model(self):
        return False

    def set_clip_skip(self, *_):
        pass

    def clear_references(self):
        self.ref_latents.clear()

    def get_prompt_lengths_on_ui(self, prompt):
        count = len(self.pipe.processor.tokenizer.encode(prompt))
        return count, 2048

    def get_learned_conditioning(self, prompt):
        raise RuntimeError('FIBO 2.0 encodes multimodal conditioning inside its official sampling stage.')

    def get_first_stage_encoding(self, x):
        return x

    @torch.inference_mode()
    def encode_first_stage(self, pixels):
        pipe = self.pipe
        x = pipe._patchify_latents(pipe.vae.encode(pixels.to(dtype=pipe.vae.dtype)).latent_dist.mode().float())
        mean = pipe.vae.bn.running_mean.view(1, -1, 1, 1).to(x)
        scale = (pipe.vae.bn.running_var + pipe.vae.config.batch_norm_eps).rsqrt().view(1, -1, 1, 1).to(x)
        return pipe._unpatchify_latents((x-mean)*scale)

    @torch.inference_mode()
    def decode_first_stage(self, latents):
        from modules import shared
        shared.state.textinfo = 'FIBO 2.0: decoding image'
        pipe = self.pipe
        x = pipe._patchify_latents(latents)
        mean = pipe.vae.bn.running_mean.view(1, -1, 1, 1).to(x)
        scale = (pipe.vae.bn.running_var + pipe.vae.config.batch_norm_eps).rsqrt().view(1, -1, 1, 1).to(x)
        x = pipe._unpatchify_latents(x / scale + mean)
        return pipe.vae.decode(x.to(dtype=pipe.vae.dtype), return_dict=False)[0]

    def bind(self, p):
        p._fibo2_variant = getattr(self.pipe, '_fibo2_variant', 'edit-turbo')
        p.init = MethodType(_init, p)
        p.sample = MethodType(_sample, p)
        p.setup_conds = MethodType(_setup_conds, p)
        p.extra_generation_params.update({'FIBO engine': 'official-flow compatibility', 'FIBO model': '2.0 Dev' if p._fibo2_variant == 'dev' else '2.0 Edit Turbo',
            'FIBO actual schedule': 'FlowMatchEulerDiscreteScheduler; official dynamic shift',
            'FIBO denoising strength': 'Not applicable; reference edit', 'FIBO negative prompt': 'Not applicable; CFG 1',
            'FIBO RNG': 'Forge ImageRNG', 'FIBO precision': getattr(self.pipe, '_fibo2_precision', 'BF16')})
        if p._fibo2_variant == 'dev':
            p.extra_generation_params['FIBO negative prompt'] = 'Native Dev classifier-free guidance'
            p.extra_generation_params['FIBO denoising strength'] = 'Not applicable; generation from noise'
        p.extra_generation_params['FIBO memory'] = getattr(self.pipe,'fibo2_memory_policy','Not loaded')
        if hasattr(self.pipe,'fibo2_attention_policy'):
            p.extra_generation_params['FIBO attention']=self.pipe.fibo2_attention_policy
        for role,name in getattr(self.pipe,'fibo2_component_names',{}).items():
            p.extra_generation_params['FIBO '+role] = name

class TransformerBoundary(torch.nn.Module):
    """Expose the actual FIBO network to Forge's model/extension ownership API."""
    def __init__(self, transformer, load_device):
        super().__init__()
        self.transformer = transformer
        self.storage_dtype = self.computation_dtype = torch.bfloat16
        self.load_device, self.offload_device = load_device, torch.device('cpu')
        self.initial_device = torch.device('cpu')
    def forward(self, x, t, context=None, **kwargs):
        if context is None or context.ndim != 4 or context.shape[1] != 6:
            raise RuntimeError('FIBO requires six bundles of Qwen3-VL hidden states; ordinary SD conditioning is unsupported.')
        return self.transformer(hidden_states=x, timestep=t, encoder_hidden_states=context,
                                return_dict=False)[0]

def _setup_conds(p):
    p.c = p.uc = None

def _init(p, *_):
    from PIL import Image, ImageOps, ImageFilter
    validate_request(p, getattr(p, '_fibo2_variant', 'edit-turbo'))
    from modules import shared
    if getattr(shared.opts, 'sd_vae_decode_method', 'Full') != 'Full':
        raise ValueError('Select Full VAE decoding for FIBO 2.0; SD approximate decoders are incompatible.')
    sources = getattr(p, 'init_images', None) or []
    source = ImageOps.exif_transpose(sources[0]).convert('RGB') if sources else None
    p._fibo2_source = source.resize((int(p.width), int(p.height)), Image.Resampling.LANCZOS) if source is not None else None
    p._fibo2_mask = None
    p._fibo2_edit_context = None
    p.overlay_images = None
    p.mask_for_overlay = None
    p.color_corrections = None
    p.paste_to = None
    p.is_using_inpainting_conditioning = False
    mask = getattr(p, 'image_mask', None)
    if mask is not None:
        if source is None:
            raise ValueError('A mask requires a source image.')
        mask=mask.convert('L')
        if getattr(p, 'inpainting_mask_invert', 0):
            mask = ImageOps.invert(mask)
        from .editing import prepare_edit
        prepared=prepare_edit(sources[0],mask,(int(p.width),int(p.height)),
            only_masked=bool(getattr(p,'inpaint_full_res',False)),
            padding=getattr(p,'inpaint_full_res_padding',32),
            blur_x=getattr(p,'mask_blur_x',0),blur_y=getattr(p,'mask_blur_y',0))
        p._fibo2_edit_context=prepared
        if prepared.has_alpha and getattr(shared.opts,'samples_format','png').lower() not in ('png','webp'):
            raise ValueError('Select PNG or WebP output to preserve source transparency in a masked edit.')
        p._fibo2_source,p._fibo2_mask=prepared.reference,prepared.mask
        p.paste_to=prepared.paste_to
        p.mask_for_overlay=prepared.support
        p.overlay_images=[prepared.overlay]*int(p.batch_size)
        p.extra_generation_params=getattr(p,'extra_generation_params',{})
        p.extra_generation_params['FIBO edit area']=str(prepared.paste_to or 'Whole picture')
    p.sampler = SimpleNamespace(config=SimpleNamespace(options={}, name='Euler'), last_latent=None)

@torch.inference_mode()
def _sample(p, conditioning, unconditional_conditioning, seeds, subseeds, subseed_strength, prompts):
    from modules import shared
    from modules.sd_samplers_common import InterruptedException
    state, engine = shared.state, p.sd_model
    outputs = []
    state.sampling_steps = int(p.steps) * len(prompts)
    noise = p.rng.next().float()
    if noise.shape[0] != len(prompts):
        raise RuntimeError('Forge noise batch does not match the FIBO prompt batch.')
    if p.scripts is not None:
        p.scripts.process_before_every_sampling(p, x=noise, noise=noise,
            c=conditioning, uc=unconditional_conditioning)
    if getattr(p, 'modified_noise', None) is not None:
        noise = p.modified_noise.float()
    for index, (prompt, seed) in enumerate(zip(prompts, seeds)):
        if state.interrupted or state.stopping_generation:
            if outputs:
                break
            raise RuntimeError('FIBO editing cancelled before sampling.')
        last = None
        def callback(pipe, step, timestep, values):
            nonlocal last
            last = values['latents']
            p.sampler.last_latent = last
            if state.interrupted or state.stopping_generation or state.skipped:
                raise InterruptedException()
            state.sampling_step = index * int(p.steps) + step + 1
            state.textinfo = f'FIBO 2.0: {state.sampling_step}/{state.sampling_steps}'
            # FIBO's 32-channel latents cannot use SD's approximate preview.
            return values
        try:
            state.textinfo = 'FIBO 2.0: encoding source and prompt'
            variant = getattr(p, '_fibo2_variant', 'edit-turbo')
            negative = getattr(p, 'negative_prompt', '') if variant == 'dev' else None
            if isinstance(negative, list):
                negative = negative[index]
            result = engine.pipe(prompt=prompt, image=p._fibo2_source, mask=p._fibo2_mask,
                negative_prompt=negative,
                width=int(p.width), height=int(p.height), num_inference_steps=int(p.steps), guidance_scale=float(getattr(p, 'cfg_scale', 1.0)),
                generator=torch.Generator('cpu').manual_seed(int(seed)),
                latents=noise[index:index+1],
                fonts=getattr(engine.pipe, 'fibo2_fonts', []),
                output_type='latent', callback_on_step_end=callback)
            outputs.append(result.images)
        except InterruptedException:
            engine.pipe.maybe_free_model_hooks()
            if last is not None:
                outputs.append(last)
            if outputs:
                break
            raise RuntimeError('FIBO editing cancelled; no completed sampling result.')
    return torch.cat(outputs)
