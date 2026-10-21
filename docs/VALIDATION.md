# Verification scope

The installed Windows/RTX5090 snapshot passed90 tests with1 optional local-asset
test skipped. It also completed real native Forge API inference and visual checks:

| Check | Result |
|---|---|
| FIBO2 Turbo INT8 + BF16 encoder | Passed; native img2img,1024square,4steps CFG1 |
| FIBO2 Turbo INT8 + official INT8 encoder | Passed; same native editing path |
| FIBO2 Dev INT8 + BF16 encoder | Passed; genuine noise-only txt2img,30steps CFG5 |
| Photorealistic Dev portrait | Passed visual review;896x1152,80.08s including processing |
| Masked reference edit | Passed; outside-mask RGBA exactly retained |
| Cancel/recovery | Interrupted Dev after1/30 steps; following edit succeeded |
| Save metadata | Returned PNG infotext matched native API infotext |
| Restart persistence | Selected test checkpoint/components retained |
| Installed-code smoke after dedup | Passed;42.05s native Turbo request |
| Existing controls |34 served native controls and Generate wiring recorded; no FIBO accordion |
| Legacy1.5 routes | Detection, complete-folder checks, loader selection and mocked generation/edit regressions; no full-weight GPU claim |
| Browser click/rendered gallery | Pending: no available browser surface; API/config evidence does not replace it |
| BF16 diffusion | Deferred; BF16 encoder inference is verified separately |

The publication snapshot includes no model resources or private machine paths.
Its CPU suite separately checks packaging fixtures and the explicit local-manifest
helper. The real-tokenizer check skips when a tokenizer is not installed. GitHub
CI status is reported from the actual workflow run after publication, not assumed
from local test passes. No inference is run by CI.
