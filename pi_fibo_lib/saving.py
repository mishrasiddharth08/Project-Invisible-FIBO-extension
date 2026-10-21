"""Native filename patterns and callbacks with the correct active model metadata."""
from types import SimpleNamespace

def save_completed(images, shared, image, p, checkpoint, seed, prompt, info, **kwargs):
    # Forge filename generators consult shared.sd_model even when p has metadata.
    # A short-lived metadata object prevents names from the previously loaded SD
    # model leaking into FIBO filenames. Restore the exact object even on failure.
    original = shared.sd_model
    metadata = SimpleNamespace(sd_model_hash=p.sd_model_hash, sd_checkpoint_info=checkpoint,
                               is_sd1=False, is_sdxl=False, is_fibo=True)
    shared.sd_model = metadata
    try:
        path = kwargs.pop("path", p.outpath_samples)
        basename = kwargs.pop("basename", "")
        extension = kwargs.pop("extension", shared.opts.samples_format)
        return images.save_image(image, path, basename, seed=seed, prompt=prompt,
                                 extension=extension, info=info, p=p, **kwargs)
    finally:
        shared.sd_model = original

def finish_gallery(images, shared, completed, infos, p, checkpoint, seeds, prompts):
    """Match Forge's grid preferences and gallery indexing without losing per-image seeds."""
    gallery, texts = list(completed), list(infos)
    opts = shared.opts
    first_image = 0
    if not completed or getattr(p, "do_not_save_grid", False):
        return gallery, texts, first_image
    if len(completed) < 2 and getattr(opts, "grid_only_if_multiple", True):
        return gallery, texts, first_image
    show_grid = getattr(opts, "return_grid", False)
    save_grid = getattr(opts, "grid_save", False)
    if not (show_grid or save_grid):
        return gallery, texts, first_image
    grid = images.image_grid(completed, p.batch_size)
    text = infos[0]
    if getattr(opts, "enable_pnginfo", True):
        grid.info["parameters"] = text
    if save_grid:
        save_completed(images, shared, grid, p, checkpoint, seeds[0], prompts[0], text,
                       path=p.outpath_grids, basename="grid", extension=opts.grid_format,
                       short_filename=not opts.grid_extended_filename, grid=True)
    if show_grid:
        gallery.insert(0, grid)
        texts.insert(0, text)
        first_image = 1
    return gallery, texts, first_image
