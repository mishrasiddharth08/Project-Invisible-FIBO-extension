"""Opt-in 2px grid cleanup after decoding, before preview/save. No ComfyUI dependency."""
MODES = ("Off", "Auto")

def process_image(image, mode):
    if mode not in MODES:
        raise ValueError("FIBO DeGrid must be Off or Auto")
    if mode == "Off":
        return image, {"FIBO DeGrid": "Off"}
    import numpy as np
    import torch
    from PIL import Image
    from .vendor.degrid_core import degrid
    array = image if isinstance(image, np.ndarray) else np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    if array.ndim != 3 or array.shape[-1] != 3:
        raise ValueError("DeGrid expects a decoded RGB image")
    tensor = torch.from_numpy(np.array(array, dtype=np.float32, copy=True)).unsqueeze(0)
    try:
        with torch.inference_mode():
            cleaned, _, stats = degrid(tensor, mode="auto", skip_when_clean=True)
    except Exception as error:
        # Display/save the successfully generated picture if optional cleanup fails.
        print(f"[Invisible-FIBO] DeGrid skipped ({type(error).__name__}); preserving the generated image")
        output = image if isinstance(image, Image.Image) else Image.fromarray((array.clip(0, 1) * 255).round().astype(np.uint8))
        return output, {"FIBO DeGrid": "Auto", "FIBO DeGrid result": f"Skipped: {type(error).__name__}"}
    stat = stats[0]
    info = {"FIBO DeGrid": "Auto", "FIBO DeGrid result": "Skipped: no grid detected" if stat["skipped"] else "Applied",
            "FIBO DeGrid amplitude": round(stat["amp_255"], 4),
            "FIBO DeGrid limit": round(stat["limit"], 6), "FIBO DeGrid revision": "5699bc33"}
    if stat["skipped"] and isinstance(image, Image.Image):
        return image, info
    pixels = (cleaned[0].numpy().clip(0, 1) * 255).round().astype(np.uint8)
    return Image.fromarray(pixels), info
