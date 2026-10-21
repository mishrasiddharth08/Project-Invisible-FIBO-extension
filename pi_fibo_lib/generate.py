"""Resolved Forge request -> official FIBO -> native saving and Processed result."""
import inspect
import time

from .catalog import VARIANTS, resolve_folder
from .prompts import prepare, check_budget
from . import runtime
from .settings import resolve
from .saving import save_completed, finish_gallery
from .progress import LiveProgress
from .preview import DevelopingPreview, publish
from .degrid import process_image

class SkipImage(Exception):
    pass

def _stop(state):
    if state.interrupted or getattr(state, "stopping_generation", False):
        raise InterruptedError("FIBO generation stopped")
    if state.skipped:
        raise SkipImage()

def step_callback(state, progress=None, preview=None):
    def callback(pipe, index, timestep, values):
        _stop(state)
        state.sampling_step = index + 1
        if preview:
            preview.update(pipe, index, values)
        if progress:
            progress.phase("Decoding image" if index + 1 >= state.sampling_steps else "Generating image")
        else:
            state.textinfo = f"FIBO: step {index + 1}/{state.sampling_steps}"
        return values
    return callback

def _info(p, prompt, negative, seed, variant, route):
    # Standard PNG infotext syntax; no access to an unrelated shared.sd_model.
    from modules.infotext_utils import quote
    params = {"Steps": p.steps, "Sampler": "Euler", "Schedule type": "Simple",
              "CFG scale": p.cfg_scale, "Seed": seed, "Size": f"{p.width}x{p.height}",
              "Model": VARIANTS[variant].label, "FIBO repository": VARIANTS[variant].repo,
              "FIBO memory": route, "FIBO precision": "BF16", "FIBO approximation": "Off"}
    params.update({key: value for key, value in p.extra_generation_params.items() if key.startswith("FIBO DeGrid")})
    return prompt + (f"\nNegative prompt: {negative}" if negative else "") + "\n" + ", ".join(f"{k}: {quote(v)}" for k, v in params.items())

def generate(p, variant, checkpoint, forge_root):
    import torch
    from modules import shared, processing, images
    from backend import memory_management
    spec = VARIANTS[variant]
    settings = resolve(p, shared.opts)
    references = getattr(p, "init_images", None)
    is_img2img = getattr(p, "is_img2img", hasattr(p, "init_images"))
    if spec.edit and not is_img2img:
        raise ValueError("Editing belongs in img2img: open that tab, choose FIBO Edit, and add an image.")
    if not spec.edit and is_img2img:
        raise ValueError("For img2img, select FIBO Edit Base or Turbo. Use this generation checkpoint in txt2img.")
    if spec.edit and not references:
        raise ValueError("Add your source image in img2img before editing.")
    if spec.edit and getattr(p, "inpaint_full_res", False):
        raise ValueError("FIBO masked editing currently supports Whole picture. Select that in img2img; Only masked needs a separate crop/merge implementation.")
    folder = resolve_folder(variant, forge_root)
    if spec.guidance == 1 and float(p.cfg_scale) != 1:
        raise ValueError(f"{spec.label} is distilled for CFG 1. Set the normal CFG slider to 1; your value was not silently replaced.")
    p.setup_prompts()
    prompts = [prepare(x, edit=spec.edit) for x in p.all_prompts]
    negatives = [prepare(x) for x in p.all_negative_prompts]
    if spec.guidance == 1 and any(negatives):
        print("[Invisible-FIBO] CFG 1: negative prompts have no effect in the official distilled pipeline")
    count = len(prompts)
    if not count:
        raise ValueError("No prompts to generate")
    if isinstance(p.seed, list):
        if len(p.seed) != count:
            raise ValueError("Seed list must match the requested image count")
        seeds = [processing.get_fixed_seed(x) for x in p.seed]
    else:
        base = processing.get_fixed_seed(p.seed)
        seeds = [(int(base) + i) % 4294967295 for i in range(count)]
    p.all_seeds = seeds
    p.all_subseeds = [processing.get_fixed_seed(p.subseed)] * count
    p.sd_model_name = spec.label
    # A checkpoint manifest hash is not a hash of its multi-file model weights.
    p.sd_model_hash = None
    p.sd_vae_name = "FIBO official Wan VAE"
    p.extra_generation_params.update({"FIBO": spec.repo, "FIBO approximation": "Off"})
    state = shared.state
    completed, used_prompts, used_negatives, used_seeds, infos = [], [], [], [], []
    state.job_count = count
    state.sampling_steps = int(p.steps)
    state.sampling_step = 0
    state.current_latent = None  # Never send FIBO latents through another model's preview decoder.
    print(f"[Invisible-FIBO] {count} requested image(s); batch {p.batch_size} x count {p.n_iter}; serial decoding preserves VRAM")
    with runtime.LOCK, torch.inference_mode(), LiveProgress(state) as progress:
        _stop(state)
        memory_management.unload_all_models()
        try:
            pipe = runtime.acquire(folder, variant, settings["memory"], pixels=int(p.width) * int(p.height))
        except Exception:
            runtime.release()
            raise
        for prompt, negative in zip(prompts, negatives):
            check_budget(pipe.tokenizer, prompt)
            if p.cfg_scale > 1:
                check_budget(pipe.tokenizer, negative)
        progress.phase("Preparing prompt")
        for index, (prompt, negative, seed) in enumerate(zip(prompts, negatives, seeds)):
            try:
                _stop(state)
                state.job = f"FIBO image {index + 1}/{count}"
                p.iteration = index // int(p.batch_size)
                p.batch_index = index % int(p.batch_size)
                state.sampling_step = 0
                progress.phase("Encoding prompt")
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
                started = time.perf_counter()
                preview = DevelopingPreview(state, int(p.width), int(p.height), float(p.cfg_scale),
                    enabled=shared.opts.live_previews_enable and getattr(shared.opts, "show_progress_every_n_steps", 1) != -1)
                kwargs = dict(prompt=prompt, negative_prompt=negative if p.cfg_scale > 1 else None,
                              width=int(p.width), height=int(p.height), num_inference_steps=int(p.steps),
                              guidance_scale=float(p.cfg_scale), generator=torch.Generator(device="cpu").manual_seed(seed),
                              callback_on_step_end=step_callback(state, progress, preview), max_sequence_length=3000)
                if spec.edit:
                    kwargs["image"] = references[min(index, len(references) - 1)].convert("RGB")
                    mask = getattr(p, "image_mask", None)
                    if mask is not None:
                        from PIL import ImageOps, ImageFilter
                        mask = mask.convert("L")
                        if getattr(p, "inpainting_mask_invert", 0):
                            mask = ImageOps.invert(mask)
                        blur = max(getattr(p, "mask_blur_x", 0), getattr(p, "mask_blur_y", 0))
                        if blur:
                            mask = mask.filter(ImageFilter.GaussianBlur(blur))
                        kwargs["mask"] = mask
                    if "_auto_resize" in inspect.signature(pipe.__call__).parameters:
                        kwargs["_auto_resize"] = False
                    print("[Invisible-FIBO] Edit uses the instruction as edit strength; the SD denoising-strength slider does not apply")
                if settings["degrid"] != "Off":
                    kwargs["output_type"] = "np"  # Clean float RGB before 8-bit rounding.
                transformer = getattr(pipe, "transformer", None)
                phase_hook = transformer.register_forward_pre_hook(lambda *_: progress.phase("Generating image")) if transformer is not None else None
                preview_hook = transformer.register_forward_hook(preview.capture) if transformer is not None and preview.enabled else None
                try:
                    result = pipe(**kwargs)
                finally:
                    if phase_hook is not None:
                        phase_hook.remove()
                    if preview_hook is not None:
                        preview_hook.remove()
                    preview.velocity = None
                image = result.images[0]
                if settings["degrid"] != "Off":
                    progress.phase("Checking 2px grid")
                image, cleanup_info = process_image(image, settings["degrid"])
                p.extra_generation_params.update(cleanup_info)
                if image.size != (int(p.width), int(p.height)):
                    raise RuntimeError(f"Official pipeline returned {image.size}; expected {(p.width, p.height)}")
                publish(state, image)  # Show the real final image before disk saving.
                torch.cuda.synchronize()
                elapsed = time.perf_counter() - started
                peak = torch.cuda.max_memory_allocated() / 1024**3
                info = _info(p, prompt, negative, seed, variant, pipe._pi_memory_route)
                image.info["parameters"] = info
                # Save each fully decoded image immediately, even if Stop arrives after decode.
                if shared.opts.samples_save and not p.do_not_save_samples:
                    progress.phase("Saving image")
                    save_completed(images, shared, image, p, checkpoint, seed, p.all_prompts[index], info)
                completed.append(image)
                used_prompts.append(p.all_prompts[index])
                used_negatives.append(p.all_negative_prompts[index])
                used_seeds.append(seed)
                infos.append(info)
                print(f"[Invisible-FIBO] Saved/returned image {index + 1}: {elapsed:.2f}s, peak allocated {peak:.2f} GiB, seed {seed}")
            except SkipImage:
                state.skipped = False
                pipe.maybe_free_model_hooks()
            except InterruptedError:
                pipe.maybe_free_model_hooks()
                break
            except Exception:
                runtime.release()
                raise
            finally:
                state.current_latent = None
                state.nextjob()
    gallery, infos, first_image = finish_gallery(images, shared, completed, infos, p,
                                                checkpoint, used_seeds, used_prompts)
    return processing.Processed(p, gallery, seed=used_seeds[0] if used_seeds else seeds[0],
                                info=infos[0] if infos else "FIBO stopped before an image completed",
                                all_prompts=used_prompts or p.all_prompts,
                                all_negative_prompts=used_negatives or p.all_negative_prompts,
                                all_seeds=used_seeds or seeds, infotexts=infos,
                                index_of_first_image=first_image)
