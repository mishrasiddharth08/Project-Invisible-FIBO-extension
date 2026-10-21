# Changes

## 2026-10-22 (IST): single FIBO preset

- Show exactly **FIBO** in the existing Forge UI Preset dropdown. The selected
  checkpoint routes to Dev, Edit Turbo or retained legacy FIBO support.
- Preserve existing saved option namespaces and unrelated preset callbacks.
- Keep generation in txt2img and editing in native img2img; no tab or accordion
  is added. Supported controls and explicit unsupported-combination errors remain.
- Add regression coverage for labels, routing, native arguments, saved settings
  and reversible adapter cleanup. No inference implementation change.
- Recheck installed-code Dev generation, Turbo editing and a portrait hat edit.
  See README for timings and precise validation limits. Browser mouse-click and
  rendered-gallery checks remain pending; legacy 1.5 full-weight inference and
  BF16 diffusion validation are not claimed.
