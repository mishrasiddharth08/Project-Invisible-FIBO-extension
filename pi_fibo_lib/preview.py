"""Small previews of actual denoising predictions; never modify sampler tensors."""
from time import monotonic

def publish(state, image):
    state.current_latent = None
    state.current_image = image
    state.current_image_sampling_step = state.sampling_step
    state.id_live_preview += 1

class DevelopingPreview:
    def __init__(self, state, width, height, guidance, enabled=True):
        self.state, self.width, self.height = state, width, height
        self.guidance, self.enabled = guidance, enabled
        self.velocity = None
        self.last = float("-inf")
        self.frames = 0

    def capture(self, module, args, output):
        if self.enabled:
            self.velocity = (output[0] if isinstance(output, tuple) else output.sample).detach()

    def update(self, pipe, index, values):
        velocity, self.velocity = self.velocity, None
        if not self.enabled or velocity is None or index + 1 >= self.state.sampling_steps:
            return  # The final full-size decode is already next; don't decode it twice.
        if monotonic() - self.last < 1.0:
            return
        # Offloaded GPUs would otherwise transfer the full transformer for every
        # preview. Limit those to two small previews plus the actual final image.
        if getattr(pipe, "_pi_memory_route", "resident") != "resident":
            stride = max(1, self.state.sampling_steps // 3)
            if self.frames >= 2 or (index + 1) % stride:
                return
        try:
            import torch
            import torch.nn.functional as F
            latents = values["latents"]
            if self.guidance > 1:
                uncond, cond = velocity.chunk(2)
                velocity = uncond + self.guidance * (cond - uncond)
            sigma = pipe.scheduler.sigmas[pipe.scheduler.step_index].to(latents.device)
            clean = latents[:1].float() - sigma * velocity[:1, :latents.shape[1]].float()
            spatial = pipe._unpack_latents_no_patch(clean, self.height, self.width, pipe.vae_scale_factor)
            h, w = spatial.shape[-2:]
            ratio = min(1.0, 256 / max(self.height, self.width))
            spatial = F.interpolate(spatial, size=(max(1, int(h * ratio)), max(1, int(w * ratio))), mode="area")
            spatial = spatial.unsqueeze(2).to(dtype=pipe.vae.dtype)
            mean = torch.tensor(pipe.vae.config.latents_mean, device=spatial.device, dtype=spatial.dtype).view(1, -1, 1, 1, 1)
            std = torch.tensor(pipe.vae.config.latents_std, device=spatial.device, dtype=spatial.dtype).view(1, -1, 1, 1, 1)
            decoded = pipe.vae.decode(spatial * std + mean, return_dict=False)[0]
            image = pipe.image_processor.postprocess(decoded.squeeze(2), output_type="pil")[0]
            publish(self.state, image)
            self.frames += 1
            self.last = monotonic()
        except Exception as error:
            # A display-only failure must not discard the generated picture.
            self.enabled = False
            print(f"[Invisible-FIBO] Developing preview disabled ({type(error).__name__}); final image will still display")
