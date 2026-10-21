"""Resolve extension settings once; normal Forge controls remain authoritative."""
def resolve(p, opts):
    result = {"memory": getattr(opts, "pi_fibo_memory", "Auto"),
              "degrid": getattr(opts, "pi_fibo_degrid", "Off")}
    if result["degrid"] not in ("Off", "Auto"):
        raise ValueError("FIBO DeGrid must be Off or Auto")
    if result["memory"] not in ("Auto", "Low VRAM"):
        raise ValueError("FIBO memory setting must be Auto or Low VRAM")
    if getattr(opts, "forge_unet_storage_dtype", "Automatic") != "Automatic":
        raise ValueError("The verified FIBO baseline uses the official BF16 weights. Set Diffusion in Low Bits to Automatic; other formats need separate validation.")
    if getattr(opts, "forge_additional_modules", []):
        raise ValueError("FIBO uses the matching text encoder and VAE in its selected model folder. Clear the separate TE/VAE selection or choose the fibo UI Preset.")
    unsupported = {
        "Hires fix": getattr(p, "enable_hr", False),
        "Variation seed strength": getattr(p, "subseed_strength", 0),
        "Seed resize": getattr(p, "seed_resize_from_w", 0) > 0 or getattr(p, "seed_resize_from_h", 0) > 0,
        "Seamless tiling": getattr(p, "tiling", False),
        "Face restoration": getattr(p, "restore_faces", False),
        "Refiner": getattr(p, "refiner_checkpoint", None),
    }
    selected = [key for key, enabled in unsupported.items() if enabled]
    if selected:
        raise ValueError("These controls have no verified FIBO implementation yet: " + ", ".join(selected) + ". Disable them for the official pipeline.")
    if getattr(p, "sampler_name", "Euler") != "Euler" or getattr(p, "scheduler", "Simple") not in ("Simple", "Automatic"):
        raise ValueError("FIBO requires its official Euler flow schedule. Select the fibo UI Preset (Euler / Simple).")
    if int(p.width) % 16 or int(p.height) % 16 or min(int(p.width), int(p.height)) < 16:
        raise ValueError("FIBO width and height must be positive multiples of 16; resolution was not changed")
    if int(p.steps) < 1 or int(p.batch_size) < 1 or int(p.n_iter) < 1:
        raise ValueError("Steps, batch size and batch count must be positive")
    if getattr(p, "override_settings", {}).keys() - {"sd_model_checkpoint"}:
        raise ValueError("FIBO currently accepts only the checkpoint override; other override settings need explicit implementation")
    return result
