# Project Invisible FIBO for Forge Neo

FIBO 2.0 Dev text-to-image and Edit Turbo reference editing inside Forge Neo's
existing controls. Compatibility preview: `forge_fibo2` version **0.1.0**;
the retained legacy module identifies itself as `2.0.0-dev`.

| Model route | Existing Forge workflow | Validation |
|---|---|---|
| FIBO 2.0 Dev | **FIBO**, Dev checkpoint, txt2img, Euler/Simple,30 steps,CFG5 | Real official INT8 diffusion + BF16 encoder generation and portrait inspected |
| FIBO 2.0 Edit Turbo | **FIBO**, Turbo checkpoint, img2img source/inpaint input, Euler/Simple,4 steps,CFG1 | Real official INT8 diffusion with BF16 and official INT8 encoders; masked editing/recovery verified |
| FIBO 1.5 generation | **FIBO** and FIBO1.5 manifest checkpoint; txt2img | Separate implementation retained; dispatch/loader/generation regression tests, not full-weight validation in this run |
| FIBO Edit1.5 Base/Turbo | Legacy edit manifest checkpoints; img2img | Separate `BriaFiboEditPipeline` route retained; mocked-model regression coverage, not full-weight validation in this run |

**No new generation tab, accordion, source widget or Generate button.** Use the
normal main checkpoint/encoder/VAE selector and native Generate/Interrupt.
Forge core, backend and shared packages are not patched by installation.

The streamlined UI presents one **FIBO** item in the existing **UI Preset**
dropdown. The selected checkpoint chooses the Dev, Turbo or legacy route.
Internal `fibo`, `fibo2` and `fibo2_dev` option names remain for saved-setting
compatibility; they are not three separate displayed choices. Editing stays in
native img2img. This UI cleanup does not expand the supported control combinations.

## Install and link existing FIBO2 assets

Place this extension at `extensions/project-invisible-pi-fibo-15`. Update the
existing extension with a backup rather than installing a duplicate. The
installed dependency check is read-only; no dependency upgrade or startup model
download occurs. Tested environment: Windows, RTX5090, Torch2.13.0+cu130,
Diffusers0.37.1, Transformers4.57.6. Other environments are not certified.

Obtain the appropriate official files from
[Comfy-Org/Fibo-2.0](https://huggingface.co/Comfy-Org/Fibo-2.0) under their own terms.
Keep diffusion checkpoints in `models/Stable-diffusion/FIBO-2.0`, encoders under
`models/text_encoder`, and the shared FLUX2 VAE under `models/VAE`. Compatible
AutoLink names are supported; identity is verified by pinned SHA256, not filename.
The VAE is FP32 on disk; do not invent an INT8 VAE variant.

The runtime also needs the processor/tokenizer, scheduler, component configs,
and fonts from the pinned BRIA Diffusers snapshot recorded in
`forge_fibo2/assets.py`. These resources and model weights are **not bundled**.
Use an existing complete local configuration folder: it must contain
`model_index.json`, `processor/`, `scheduler/`, `fonts/`, and either
`configs/{transformer,text_encoder,vae}/config.json` or the corresponding
Diffusers component folders. Do not place credentials in this repository.

Run the following explicitly using Forge's Python, replacing each example path:

```text
python tools/link_existing_official.py --forge-root /path/to/forge --runtime-dir /path/to/local/configuration --checkpoint /path/to/official-checkpoint.safetensors --te-bf16 /path/to/official-bf16-encoder.safetensors --te-int8 /path/to/official-int8-encoder.safetensors --vae /path/to/flux2-vae.safetensors
```

This only validates local hashes and writes the ignored `model-manifest.json`.
It refuses to overwrite an existing manifest. It does not download, convert,
accept agreements, change packages or run GPU work. Both encoder paths are
recorded for the current AutoLink policy; select the intended precision in the
main bar. Restart Forge after installing the extension.

For legacy1.5, retain a complete matching official Diffusers folder and run
`python -m pi_fibo_lib.setup --variant fibo15 --folder /path/to/Fibo-1.5`.
Legacy1.5 uses a distinct SmolLM3 encoder and48-channel Wan VAE; it does not use
the FIBO2 Qwen3-VL/FLUX2 components. See `pi_fibo_lib/catalog.py` for legacy variants.

## Existing img2img controls

Editing remains in **img2img**. Prompt, size, native seed/RNG, batch lifecycle,
source/inpaint input, mask inversion/blur/padding, Whole picture/Only masked,
Generate/Interrupt, gallery/save lifecycle and PNG metadata are reused.
FIBO reference editing conditions the encoder and clean reference latents;
it starts the target from noise. The existing denoising-strength slider explains
why ordinary img2img denoising strength does not apply.

The current validated recipes require Euler/Simple and the steps/CFG above.
Unsupported resize modes other than Just resize, non-neutral inpaint Fill modes,
variation strength, tiling, Hires/refiner and unvalidated LoRAs fail explicitly.
Full control parity, other attention kernels and arbitrary schedules are not claimed.

## Validation limits

The single-entry UI regression suite passed **103 tests;2 optional checks skipped**
with CUDA hidden. Real Gradio callbacks verified Dev/Turbo/legacy routing, saved
option compatibility and unchanged non-FIBO outputs. Live checks additionally
covered native API generation, both encoder precisions, exact outside-mask RGBA
preservation, cancellation/recovery, PNG infotext roundtrip and restart persistence.

Post-restart checks used the installed extension in an isolated Forge profile:

| Native request | Settings | Total request | Sampling loop |
|---|---|---:|---:|
| Dev txt2img | 1024x1024,30 steps,CFG5,seed64 | 84.9s | 46s |
| Turbo source edit | 1024x1024,4 steps,CFG1,seed64 | 21.2s | 12s |
| Dev portrait | 896x1152,30 steps,CFG5,seed20261021 | 66.1s | 46s |
| Turbo hat edit of that portrait | 896x1152,4 steps,CFG1,seed20261021 | 34.0s | 12s |

All four used official INT8_CONVROT diffusion, BF16 Qwen3-VL text encoder and
Euler/Simple on one RTX5090. Total request includes selection/loading/preparation,
encoding, sampling, decoding, saving and API response; server startup is excluded.
Full component loading and peak VRAM were not separately instrumented. These are
one-machine observations, not general benchmarks. The hat edit was visually
inspected: identity/composition remained recognizable, with minor skin/shading
changes. It was unmasked, so exact unchanged pixels are not claimed.

**Browser mouse-click Generate/Interrupt and rendered gallery/save acceptance are
still pending:** no browser/app surface was available to the verifier. Served UI
configuration and native processing/API checks are separate evidence. BF16
diffusion checkpoint validation is deferred; BF16 text-encoder inference passed.

Local model paths, original backups, download/cache state, model weights, private
logs and generated private manifests are excluded from publication. Upstream
code notices remain in `NOTICE` and `forge_fibo2/vendor/PROVENANCE.json`.

Run CPU regressions with `python -m pytest tests -q`. No model weights or GPU job
are required; optional real-asset/tokenizer checks skip when resources are absent.
The separate random-weight CUDA smoke is import-safe and requires an
explicit `python tests/test_tiny_pipeline.py --cuda-smoke` invocation.
