"""Add FIBO 2.0 to Forge's native UI Preset without modifying enum members.

Forge's preset handler uses string option keys, so an additive public choices
wrapper and engine-owned options are sufficient. Core files, stock defaults,
enum membership and sibling extensions' recipe tables remain untouched.
"""
from __future__ import annotations

from copy import copy, deepcopy
from functools import wraps

PRESET_NAME = "fibo2"
DEV_PRESET_NAME = 'fibo2_dev'
TAG = "[Invisible-FIBO 2.0]"


def _wrap_choices(arch):
    original = arch.choices
    if getattr(original, "_pi_fibo2_injected", False):
        return

    @wraps(original)
    def choices(*args, **kwargs):
        result = list(original(*args, **kwargs))
        for name in (PRESET_NAME,DEV_PRESET_NAME):
            if name not in result:
                result.append(name)
        return result

    choices._pi_fibo2_injected = True
    arch.choices = staticmethod(choices)


def _register_options(presets, opts, preset_name=PRESET_NAME):
    # Copy the plain-image SD template using Forge's real component classes.
    # Do NOT add an enum member: later sibling register() calls must stay safe.
    template = {}
    presets.register(template)
    ours = {}
    for key, info in template.items():
        if key.startswith("sd_"):
            target = preset_name + key[2:]
        elif key in ("forge_checkpoint_sd", "forge_additional_modules_sd",
                     "forge_unet_storage_dtype_sd"):
            target = key[:-2] + preset_name
        else:
            continue
        cloned = copy(info)
        cloned.default = deepcopy(info.default)
        if getattr(info, "section", (None,))[0] == "ui_sd":
            cloned.section = ('ui_'+preset_name, 'FIBO 2.0 Dev' if preset_name==DEV_PRESET_NAME else 'FIBO 2.0 Edit Turbo')
        if target.endswith("_sampler"):
            cloned.default = "Euler"
        elif target.endswith("_scheduler"):
            cloned.default = "Simple"
        elif target.endswith("_step"):
            cloned.default = 30 if preset_name==DEV_PRESET_NAME else 4
        elif target.endswith("_cfg"):
            cloned.default = 5.0 if preset_name==DEV_PRESET_NAME else 1.0
        elif target.endswith(("_width", "_height")):
            cloned.default = 1024
        if cloned.component is not None and not callable(cloned.component):
            raise TypeError(f"Unsupported Forge setting component: {target}")
        ours[target] = cloned
    required = {"forge_checkpoint_fibo2", "forge_additional_modules_fibo2",
                "forge_unet_storage_dtype_fibo2", "fibo2_t2i_step",
                "fibo2_t2i_cfg", "fibo2_t2i_width", "fibo2_t2i_height"}
    required={key.replace('fibo2',preset_name) for key in required}
    if not required.issubset(ours):
        raise RuntimeError("Forge's preset option layout changed; registration skipped")
    for key, info in ours.items():
        if key not in opts.data_labels:
            opts.add_option(key, info)


def _report(*args, **kwargs):
    try:
        from modules_forge import main_entry
        raw = main_entry.ui_forge_preset.choices
        names = [c[1] if isinstance(c, (tuple, list)) else c for c in raw]
        labels = [c[0] if isinstance(c,(tuple,list)) else c for c in raw]
        status = "OK" if PRESET_NAME in names or 'FIBO' in labels else "MISSING"
        print(f"{TAG} UI Preset 'fibo2' served to browser: {status}")
    except Exception as error:
        print(f"{TAG} UI Preset verification unavailable: {error}")


def install() -> bool:
    """Register at script import, before Forge builds its UI. Safe to repeat."""
    try:
        from modules import shared
        from modules_forge import presets
        arch = presets.PresetArch
        arch.choices()  # Probe compatibility before touching anything.
        _register_options(presets, shared.opts)
        _register_options(presets, shared.opts, DEV_PRESET_NAME)
        _wrap_choices(arch)
        key = "forge_checkpoint_fibo2"
        current = getattr(shared.opts, "sd_model_checkpoint", None)
        if not shared.opts.data.get(key) and current and "fibo2" in str(current).lower():
            shared.opts.data[key] = current
            # Preserve this checkpoint's active modules on the first switch.
            modules_key = "forge_additional_modules_fibo2"
            if not shared.opts.data.get(modules_key):
                shared.opts.data[modules_key] = list(
                    getattr(shared.opts, "forge_additional_modules", None) or [])
        # Store the registration marker on the host class, not this module:
        # script/module reloads must not accumulate callbacks or choice entries.
        if not getattr(arch, "_pi_fibo2_report_installed", False):
            try:
                from modules import script_callbacks
                script_callbacks.on_app_started(_report)
                arch._pi_fibo2_report_installed = True
            except (ImportError, AttributeError):
                pass
        print(f"{TAG} UI Preset 'fibo2' registered (no Forge core-file edits)")
        return True
    except Exception as error:
        print(f"{TAG} WARNING: FIBO 2.0 UI Preset registration failed: {error}")
        return False

