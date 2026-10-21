"""Transparent native Generate/API and checkpoint hooks, with atomic installation."""
from functools import wraps
import inspect
from pathlib import Path
import sys

from .catalog import ROOT, VARIANTS, selected_variant, resolve_folder

TAG = "[Invisible-FIBO]"
ALIASES = ("modules.api.api", "modules.txt2img", "modules.img2img")

def selection(p, sd_models, shared):
    override = getattr(p, "override_settings", None) or {}
    label = override.get("sd_model_checkpoint", getattr(shared.opts, "sd_model_checkpoint", None))
    ci = sd_models.checkpoint_aliases.get(label)
    if ci is None:
        ci = next((c for c in sd_models.checkpoints_list.values()
                   if str(getattr(c, "filename", "")) == str(label)), None)
    if ci is None:
        return None
    variant = selected_variant(ci.filename)
    return (variant, ci) if variant else None

def register_checkpoints(sd_models, forge_root):
    for variant, spec in VARIANTS.items():
        path = ROOT / "checkpoints" / f"{variant}.json"
        if selected_variant(path) is None:
            raise RuntimeError(f"Invalid FIBO checkpoint manifest: {path}")
        ci = sd_models.CheckpointInfo(str(path))
        try:
            resolve_folder(variant, forge_root)
            suffix = ""
        except (OSError, ValueError):
            suffix = " (setup required)"
        ci.name = ci.title = ci.short_title = spec.label + suffix
        ci.model_name = variant
        ci.name_for_extra = spec.label
        # Keep the absolute path stable across labels, refreshes and configuration saves.
        ci.ids = [str(path), spec.label, ci.title]
        ci.register()

def install(scripts, processing, sd_models, shared, forge_root, generate=None, release=None):
    """Dependency injection allows genuine wrapper/coexistence tests without booting Forge."""
    if getattr(scripts.ScriptRunner.run, "_pi_fibo_v2", False):
        return True
    if generate is None:
        from .generate import generate
    if release is None:
        from .runtime import release
    original_run = scripts.ScriptRunner.run
    original_process = processing.process_images
    original_list = sd_models.list_models
    # Verify every required surface BEFORE committing any wrappers.
    try:
        inspect.signature(original_run).bind(object(), object())
        inspect.signature(original_process).bind(object())
        inspect.signature(original_list).bind()
        inspect.signature(processing.Processed).bind(object(), [], seed=0, info="")
        for name in ("checkpoints_list", "checkpoint_aliases", "CheckpointInfo"):
            getattr(sd_models, name)
        if not callable(getattr(shared.state, "nextjob", None)):
            raise TypeError("Forge State.nextjob unavailable")
    except (AttributeError, TypeError, ValueError) as error:
        print(f"{TAG} disabled: host interface changed ({error}). Stock functions retained.")
        return False

    def dispatch(p, fallback, args, kwargs):
        choice = selection(p, sd_models, shared)
        if choice is None:
            try:
                release()
            except Exception as error:
                print(f"{TAG} FIBO cleanup failed: {error}; forwarding the non-FIBO call unchanged")
            return fallback(*args, **kwargs)
        return generate(p, choice[0], choice[1], forge_root)

    @wraps(original_run)
    def run(self, p, *args, **kwargs):
        # Let Forge run the selected script (e.g. X/Y/Z orchestration). Its
        # process_images calls still reach our model-specific API bridge.
        # Taking over before this point silently ignored the script dropdown.
        if selection(p, sd_models, shared) is not None and args and args[0] not in (None, 0):
            return original_run(self, p, *args, **kwargs)
        return dispatch(p, original_run, (self, p, *args), kwargs)

    @wraps(original_process)
    def process(p, *args, **kwargs):
        return dispatch(p, original_process, (p, *args), kwargs)

    @wraps(original_list)
    def listing(*args, **kwargs):
        result = original_list(*args, **kwargs)
        try:
            register_checkpoints(sd_models, forge_root)
        except Exception as error:
            print(f"{TAG} checkpoint registration failed: {error}; stock list retained")
        return result

    run._pi_fibo_v2 = process._pi_fibo_v2 = listing._pi_fibo_v2 = True
    scripts.ScriptRunner.run = run
    processing.process_images = process
    sd_models.list_models = listing
    for name in ALIASES:
        module = sys.modules.get(name)
        if module is not None and getattr(module, "process_images", None) is original_process:
            module.process_images = process
    try:
        register_checkpoints(sd_models, forge_root)
    except Exception as error:
        print(f"{TAG} initial checkpoint registration unavailable: {error}")
    return True

def start():
    try:
        from modules import scripts, processing, sd_models, shared, script_callbacks
        from modules.paths_internal import script_path
        from .runtime import release
        if not install(scripts, processing, sd_models, shared, script_path):
            return False
        from .preset import install as install_preset
        install_preset()
        script_callbacks.on_script_unloaded(release)
        print(f"{TAG} native checkpoint selection, Generate and API installed; Forge core untouched")
        return True
    except Exception as error:
        print(f"{TAG} integration unavailable: {error}; no core files or dependencies changed")
        return False
