"""Display-only clean predictions; preserve sampling tensors and native compositing."""
from time import monotonic
import torch
import torch.nn.functional as F
from pi_fibo_lib.preview import publish

class DevelopingPreview:
    def __init__(self, engine, state, opts, guidance, request=None):
        self.request = request
        self.engine, self.state, self.guidance = engine, state, guidance
        self.enabled = bool(getattr(opts, 'live_previews_enable', False)) and getattr(opts, 'show_progress_every_n_steps', 1) != -1
        self.predictions = []
        self.frames, self.last = 0, float('-inf')

    def capture(self, module, args, output):
        if self.enabled:
            value = output[0] if isinstance(output, tuple) else output.sample
            self.predictions.append(value.detach())
            del self.predictions[:-2]

    def update(self, pipe, step, latents, steps):
        predictions, self.predictions = self.predictions, []
        if not self.enabled or not predictions or step + 1 >= steps or monotonic() - self.last < 1:
            return
        offloaded = getattr(pipe, 'fibo2_memory_policy', '') == 'model-cpu-offload'
        try:
            if self.guidance > 1:
                if len(predictions) != 2:
                    raise ValueError('Missing positive/negative prediction')
                positive, negative = predictions
                velocity = negative.float() + self.guidance * (positive.float() - negative.float())
            else:
                velocity = predictions[-1].float()
            sigma = pipe.scheduler.sigmas[pipe.scheduler.step_index].to(latents)
            clean = latents[:1].float() - sigma * velocity[:1].float()
            # FIBO2 is BCHW, with a 2x2 VAE normalization patch.
            h, w = clean.shape[-2:]
            ratio = min(1.0, 32 / max(h, w))
            size = (max(2, int(h * ratio) // 2 * 2), max(2, int(w * ratio) // 2 * 2))
            if size != (h, w):
                clean = F.interpolate(clean, size=size, mode='area')
            if offloaded:
                # Structural thumbnail only; never transfer the VAE during sampling.
                from PIL import Image
                rgb = clean[0, :3].sigmoid().mul(255).byte().permute(1, 2, 0).cpu().numpy()
                image = Image.fromarray(rgb).resize((size[1] * 8, size[0] * 8), Image.Resampling.BILINEAR)
            else:
                decoded = self.engine._decode_latents(clean)
                image = pipe.image_processor.postprocess(decoded, output_type='pil')[0]
            prepared = getattr(self.request, '_fibo2_edit_context', None)
            if prepared is not None:
                from modules import processing
                from .editing import restore_masked_alpha
                from PIL import Image
                image = image.resize((int(self.request.width), int(self.request.height)), Image.Resampling.LANCZOS)
                image, _ = processing.apply_overlay(image, prepared.paste_to, prepared.overlay)
                image = restore_masked_alpha(image, prepared)
                image.thumbnail((256, 256))
            publish(self.state, image)
            self.frames += 1
            self.last = monotonic()
        except Exception as error:
            self.enabled = False
            print(f'[FIBO 2.0] Developing preview disabled ({type(error).__name__}); final output continues')
