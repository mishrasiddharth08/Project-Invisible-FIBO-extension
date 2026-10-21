"""Reversible runtime policy for Project Invisible Model Autolink."""
from __future__ import annotations

import importlib
import json
import threading
from dataclasses import replace
from pathlib import Path

from . import assets

FAMILY = "fibo2"
PRESET = "fibo2"
DEV_FAMILY = 'fibo2_dev'
POLICY = {
    "slots": [["qwen3vl_4b"], ["llada_vae"]],
    "description": "FIBO 2.0 Edit Turbo",
}
_LOCK = threading.RLock()
_ACTIVE = None
_MISSING = object()


class AutolinkError(RuntimeError):
    pass


def _module(name):
    try:
        return importlib.import_module(f"model_autolink.{name}")
    except ImportError as exc:
        raise AutolinkError("Project Invisible Model Autolink is unavailable") from exc


def _manifest(path: Path) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, ValueError) as exc:
        raise AutolinkError(f"Invalid FIBO 2 asset manifest: {path}") from exc
    from .provenance import PINS
    local=value.get('revision')==assets.REVISION and value.get('variant')==assets.VARIANT
    official=(value.get('repository')==PINS['repository'] and value.get('revision')==PINS['revision']
              and value.get('variant') in ('dev','edit-turbo'))
    if value.get('family')!=FAMILY or not (local or official):
        raise AutolinkError("Asset manifest family, variant, or pinned revision is wrong")
    return value


def _component_paths(manifest: dict) -> dict:
    components = manifest.get("components", {})
    try:
        return {
            "checkpoint": Path(components["checkpoint"]["path"]),
            "text_encoder_bf16": Path(components["text_encoder_bf16"]["path"]),
            "text_encoder_int8_convrot": Path(components["text_encoder_int8_convrot"]["path"]),
            "vae": Path(components["vae"]["path"]),
        }
    except (KeyError, TypeError) as exc:
        raise AutolinkError("Asset manifest is missing exact component paths") from exc


def _restore(mapping: dict, key, previous) -> None:
    if previous is _MISSING:
        mapping.pop(key, None)
    else:
        mapping[key] = previous


class PolicyHandle:
    def __init__(self, manifest_path: Path, undo):
        self.manifest_path = manifest_path
        self._undo = undo
        self.active = True

    def uninstall(self) -> None:
        global _ACTIVE
        with _LOCK:
            if not self.active:
                return
            self._undo()
            self.active = False
            if _ACTIVE is self:
                _ACTIVE = None


def register_runtime_policy(manifest_path) -> PolicyHandle:
    """Register FIBO 2 with the installed Autolink process; no files are edited."""
    global _ACTIVE
    manifest_path = Path(manifest_path).resolve()
    with _LOCK:
        if _ACTIVE and _ACTIVE.active:
            if _ACTIVE.manifest_path == manifest_path:
                return _ACTIVE
            raise AutolinkError("A different FIBO 2 Autolink manifest is already active")

        manifest = _manifest(manifest_path)
        paths = _component_paths(manifest)
        try:
            from .provenance import PINS
            if manifest.get('repository')==PINS['repository']:
                from .runtime import _validate_metadata
                for key,role in (('checkpoint','transformer'),('text_encoder_bf16','text_encoder'),('vae','vae')):
                    _validate_metadata(paths[key],role)
                if paths['text_encoder_int8_convrot'].is_file():
                    _validate_metadata(paths['text_encoder_int8_convrot'],'text_encoder')
            else:
                assets.validate_model(paths, verify_hashes=False)
        except assets.AssetError as exc:
            raise AutolinkError(str(exc)) from exc

        compatibility = _module("compatibility")
        smartname = _module("smartname")
        integration = _module("integration")
        registry_module = _module("registry")
        resolver_module = _module("resolver")
        fingerprint_module = _module("fingerprint")
        if hasattr(importlib.import_module("model_autolink"), "register_runtime_policy"):
            raise AutolinkError("Installed Autolink now has a public runtime-policy API; adapt to its exact contract")

        checkpoint_fp = fingerprint_module.fingerprint(paths["checkpoint"])
        previous_policy = compatibility.MATRIX.get(FAMILY, _MISSING)
        previous_display = smartname.FAMILY_DISPLAY.get(FAMILY, _MISSING)
        previous_preset = integration.FAMILY_PRESETS.get(FAMILY, _MISSING)
        original_load = registry_module.Registry.load_registry
        original_rank = resolver_module.Resolver._variant_rank
        original_families = smartname._families_for_component
        original_classify_path = registry_module.Registry.classify_path
        previous_dev = (compatibility.MATRIX.get(DEV_FAMILY,_MISSING),
                        smartname.FAMILY_DISPLAY.get(DEV_FAMILY,_MISSING),
                        integration.FAMILY_PRESETS.get(DEV_FAMILY,_MISSING))
        touched = {}

        def classify_path(registry,path,kind):
            fp,match=original_classify_path(registry,path,kind)
            if kind=='checkpoints' and match is not None and match.label==FAMILY:
                from .provenance import identify
                try:
                    identity=identify(path,'transformer')
                except ValueError:
                    identity=None  # legacy Turbo files retain their existing policy
                if identity is not None and identity['variant']=='dev':
                    match=replace(match,label=DEV_FAMILY)
            return fp,match

        def add_prototype(registry):
            key = id(registry)
            if key not in touched:
                touched[key] = (registry, list(registry.prototypes["checkpoints"]))
            entries = registry.prototypes["checkpoints"]
            if not any(label == FAMILY and fp.identity == checkpoint_fp.identity for label, fp in entries):
                entries.append((FAMILY, checkpoint_fp))

        def load_registry(registry, *args, **kwargs):
            result = original_load(registry, *args, **kwargs)
            add_prototype(registry)
            return result

        def variant_rank(family, slot, item):
            if family in (FAMILY,DEV_FAMILY):
                if manifest.get('repository')==PINS['repository']:
                    selected=Path(item.path).resolve()
                    if selected==paths['text_encoder_bf16'].resolve() or selected==paths['vae'].resolve():
                        return 0
                    if selected==paths['text_encoder_int8_convrot'].resolve():
                        return 1
                    return 4  # old compatible conversions never outrank reviewed assets
                path = str(item.path).lower().replace("\\", "/")
                return 0 if "/fibo-2.0/" in path or "--for-fibo-2.0" in path else 3
            return original_rank(family, slot, item)

        def families_for_component(label):
            # Resolver compatibility is runtime-only. Do not let Smart Rename
            # append FIBO to unrelated compatible component filenames.
            return [family for family in original_families(label) if family not in (FAMILY,DEV_FAMILY)]

        def undo():
            if registry_module.Registry.load_registry is load_registry:
                registry_module.Registry.load_registry = original_load
            if resolver_module.Resolver._variant_rank is variant_rank:
                resolver_module.Resolver._variant_rank = staticmethod(original_rank)
            if smartname._families_for_component is families_for_component:
                smartname._families_for_component = original_families
            if registry_module.Registry.classify_path is classify_path:
                registry_module.Registry.classify_path = original_classify_path
            for registry, previous in touched.values():
                registry.prototypes["checkpoints"] = previous
            _restore(compatibility.MATRIX, FAMILY, previous_policy)
            _restore(smartname.FAMILY_DISPLAY, FAMILY, previous_display)
            _restore(integration.FAMILY_PRESETS, FAMILY, previous_preset)
            for mapping,previous in zip((compatibility.MATRIX,smartname.FAMILY_DISPLAY,integration.FAMILY_PRESETS),previous_dev):
                _restore(mapping,DEV_FAMILY,previous)
            current = getattr(integration, "_resolver", None)
            if current is not None:
                current.registry.scan_components(current.component_folders)

        try:
            policy = dict(POLICY)
            if manifest.get('repository') == PINS['repository']:
                # Official Comfy repacks use the shared FLUX2 VAE, rather
                # than the older relocated Diffusers VAE prototype.
                policy['slots'] = [['qwen3vl_4b'], ['flux2_vae']]
            compatibility.MATRIX[FAMILY] = policy
            compatibility.MATRIX[DEV_FAMILY] = dict(policy,description='FIBO 2.0 Dev')
            smartname.FAMILY_DISPLAY[FAMILY] = assets.FAMILY_DISPLAY
            smartname.FAMILY_DISPLAY[DEV_FAMILY] = 'FIBO-2.0-Dev'
            integration.FAMILY_PRESETS[FAMILY] = PRESET
            integration.FAMILY_PRESETS[DEV_FAMILY] = DEV_FAMILY
            registry_module.Registry.load_registry = load_registry
            registry_module.Registry.classify_path = classify_path
            resolver_module.Resolver._variant_rank = staticmethod(variant_rank)
            smartname._families_for_component = families_for_component
            live_resolver = getattr(integration, "_resolver", None)
            if live_resolver is not None:
                add_prototype(live_resolver.registry)
                live_resolver.registry.scan_components(live_resolver.component_folders)
        except Exception as exc:
            try:
                undo()
            except Exception:
                pass
            raise AutolinkError(f"FIBO 2 Autolink registration rolled back: {exc}") from exc

        _ACTIVE = PolicyHandle(manifest_path, undo)
        return _ACTIVE


def unregister_runtime_policy(handle: PolicyHandle | None = None) -> None:
    target = handle or _ACTIVE
    if target is not None:
        target.uninstall()


install = register_runtime_policy
