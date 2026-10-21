import json
import os
import sys
import threading
import types
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from forge_fibo2 import assets, autolink


def fake_modules(monkeypatch):
    package = types.ModuleType("model_autolink")
    package.__path__ = []
    compatibility = types.ModuleType("model_autolink.compatibility")
    compatibility.MATRIX = {}
    smartname = types.ModuleType("model_autolink.smartname")
    smartname.FAMILY_DISPLAY = {}
    smartname._families_for_component = lambda label: ["existing"]
    integration = types.ModuleType("model_autolink.integration")
    integration.FAMILY_PRESETS = {}

    class Registry:
        def __init__(self):
            self.prototypes = {"checkpoints": [], "components": []}
            self.scans = 0
        def load_registry(self):
            self.prototypes = {"checkpoints": [], "components": []}
        def scan_components(self, folders):
            self.scans += 1
        def classify_path(self,path,kind):
            @dataclass
            class Match:
                label:str='fibo2'
                confidence:float=1
            return NS(identity='checkpoint-identity'),Match()

    class Resolver:
        @staticmethod
        def _variant_rank(family, slot, item):
            return 9

    registry = types.ModuleType("model_autolink.registry")
    registry.Registry = Registry
    resolver = types.ModuleType("model_autolink.resolver")
    resolver.Resolver = Resolver
    fingerprint = types.ModuleType("model_autolink.fingerprint")
    fingerprint.fingerprint = lambda path: NS(identity="checkpoint-identity")
    live_registry = Registry()
    integration._resolver = NS(registry=live_registry, component_folders=[Path("components")])
    modules = {"model_autolink": package, "model_autolink.compatibility": compatibility,
               "model_autolink.smartname": smartname, "model_autolink.integration": integration,
               "model_autolink.registry": registry, "model_autolink.resolver": resolver,
               "model_autolink.fingerprint": fingerprint}
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)
    return compatibility, smartname, integration, registry, resolver


def manifest(tmp_path):
    components = {}
    for key in ("checkpoint", "text_encoder_bf16", "text_encoder_int8_convrot", "vae"):
        components[key] = {"path": str(tmp_path / f"{key}.safetensors")}
    path = tmp_path / "assets.json"
    path.write_text(json.dumps({"family": "fibo2", "revision": assets.REVISION,
                                "variant": assets.VARIANT, "components": components}), encoding="utf-8")
    return path


def test_runtime_policy_is_scoped_reversible_and_prefers_fibo_paths(tmp_path, monkeypatch):
    compatibility, smartname, integration, registry, resolver = fake_modules(monkeypatch)
    monkeypatch.setattr(assets, "validate_model", lambda paths, **kwargs: {"ok": True})
    handle = autolink.register_runtime_policy(manifest(tmp_path))
    try:
        assert compatibility.MATRIX["fibo2"]["slots"] == [["qwen3vl_4b"], ["llada_vae"]]
        assert smartname.FAMILY_DISPLAY["fibo2"] == "FIBO-2.0"
        assert integration.FAMILY_PRESETS["fibo2"] == "fibo2"
        assert smartname._families_for_component("qwen3vl_4b") == ["existing"]
        assert integration._resolver.registry.prototypes["checkpoints"][0][0] == "fibo2"
        fresh = registry.Registry()
        fresh.load_registry()
        assert fresh.prototypes["checkpoints"][0][0] == "fibo2"
        assert resolver.Resolver._variant_rank("fibo2", ["qwen3vl_4b"], NS(path="X/FIBO-2.0/te.safetensors")) == 0
        assert resolver.Resolver._variant_rank("other", [], NS(path="x")) == 9
    finally:
        handle.uninstall()
    assert "fibo2" not in compatibility.MATRIX
    assert "fibo2" not in smartname.FAMILY_DISPLAY
    assert "fibo2" not in integration.FAMILY_PRESETS
    assert integration._resolver.registry.prototypes["checkpoints"] == []
    assert resolver.Resolver._variant_rank("other", [], NS(path="x")) == 9


def test_wrong_revision_is_rejected_before_autolink_import(tmp_path):
    path = manifest(tmp_path)
    value = json.loads(path.read_text())
    value["revision"] = "wrong"
    path.write_text(json.dumps(value))
    with pytest.raises(autolink.AutolinkError, match="pinned revision"):
        autolink.register_runtime_policy(path)


def test_official_policy_uses_shared_flux2_vae_and_restores_policy(tmp_path,monkeypatch):
    compatibility,_,_,_,resolver=fake_modules(monkeypatch)
    path=manifest(tmp_path)
    value=json.loads(path.read_text())
    from forge_fibo2.provenance import PINS
    value.update(repository=PINS['repository'],revision=PINS['revision'])
    path.write_text(json.dumps(value))
    from forge_fibo2 import runtime
    monkeypatch.setattr(runtime,'_validate_metadata',lambda *args: ({},[]))
    handle=autolink.register_runtime_policy(path)
    try:
        assert compatibility.MATRIX['fibo2']['slots']==[['qwen3vl_4b'],['flux2_vae']]
        rank=resolver.Resolver._variant_rank
        assert rank('fibo2',[],NS(path=value['components']['text_encoder_bf16']['path']))==0
        assert rank('fibo2',[],NS(path=value['components']['text_encoder_int8_convrot']['path']))==1
        assert rank('fibo2',[],NS(path=tmp_path/'legacy.safetensors'))==4
    finally:
        handle.uninstall()
    assert 'fibo2' not in compatibility.MATRIX


def test_dev_checkpoint_selects_dev_native_recipe_by_verified_identity(tmp_path,monkeypatch):
    compatibility,_,integration,registry,_=fake_modules(monkeypatch)
    monkeypatch.setattr(assets,'validate_model',lambda *args,**kwargs: {})
    from forge_fibo2 import provenance
    monkeypatch.setattr(provenance,'identify',lambda path,role: {'variant':'dev' if str(path)=='dev' else 'edit-turbo'})
    original=registry.Registry.classify_path
    handle=autolink.register_runtime_policy(manifest(tmp_path))
    try:
        assert registry.Registry().classify_path('dev','checkpoints')[1].label=='fibo2_dev'
        assert registry.Registry().classify_path('turbo','checkpoints')[1].label=='fibo2'
        assert integration.FAMILY_PRESETS['fibo2_dev']=='fibo2_dev'
        assert compatibility.MATRIX['fibo2_dev']['slots']==compatibility.MATRIX['fibo2']['slots']
    finally:
        handle.uninstall()
    assert registry.Registry.classify_path is original


def test_scan_failure_rolls_back_every_runtime_mutation(tmp_path, monkeypatch):
    compatibility, smartname, integration, registry, resolver = fake_modules(monkeypatch)
    monkeypatch.setattr(assets, "validate_model", lambda paths, **kwargs: {"ok": True})
    original_load = registry.Registry.load_registry
    original_rank = resolver.Resolver._variant_rank
    original_families = smartname._families_for_component
    integration._resolver.registry.scan_components = lambda folders: (_ for _ in ()).throw(RuntimeError("scan failed"))
    with pytest.raises(autolink.AutolinkError, match="rolled back"):
        autolink.register_runtime_policy(manifest(tmp_path))
    assert "fibo2" not in compatibility.MATRIX
    assert "fibo2" not in smartname.FAMILY_DISPLAY
    assert "fibo2" not in integration.FAMILY_PRESETS
    assert integration._resolver.registry.prototypes["checkpoints"] == []
    assert registry.Registry.load_registry is original_load
    assert resolver.Resolver._variant_rank is original_rank
    assert smartname._families_for_component is original_families


@pytest.mark.skipif(os.environ.get("PI_FIBO2_REAL_ASSET_TEST") != "1", reason="explicit local asset audit")
def test_real_autolink_fingerprints_and_int8_preference():
    project = Path(__file__).resolve().parents[1]
    forge = next(p for p in project.parents if (p/'backend').is_dir())
    autolink_root = forge / "extensions" / "project-invisible-model-autolink"
    sys.path.insert(0, str(autolink_root))
    try:
        from model_autolink import compatibility as real_compatibility
        from model_autolink.fingerprint import fingerprint, quantization
        from model_autolink.registry import IndexedFile, Registry
        from model_autolink.resolver import Resolver

        value = json.loads((project / "model-manifest.json").read_text(encoding="utf-8"))
        paths = autolink._component_paths(value)
        validated = assets.validate_model(paths)
        assert validated["text_encoder_preferred"] == str(paths["text_encoder_int8_convrot"])

        registry = Registry(autolink_root)
        checkpoint_bf16 = fingerprint(paths["checkpoint"])
        checkpoint_int8_path = Path(value["components"]["checkpoint_int8_convrot"]["path"])
        checkpoint_int8 = fingerprint(checkpoint_int8_path)
        registry.prototypes["checkpoints"].append(("fibo2", checkpoint_bf16))
        assert registry.classify(checkpoint_bf16, "checkpoints").label == "fibo2"
        assert registry.classify(checkpoint_int8, "checkpoints").label == "fibo2"

        candidates = []
        for path in (paths["text_encoder_bf16"], paths["text_encoder_int8_convrot"], paths["vae"]):
            fp = fingerprint(path)
            matches = registry.classify_all(fp, "components")
            assert matches
            candidates.append(IndexedFile(str(path), fp, tuple(item.label for item in matches),
                                          matches[0].confidence, quantization(path, fp)))
        assert "qwen3vl_4b" in candidates[1].components
        assert candidates[1].quant == "int8_convrot"
        assert "llada_vae" in candidates[2].components

        registry.index = candidates
        resolver = Resolver.__new__(Resolver)
        resolver.registry = registry
        resolver.lock = threading.RLock()
        resolver.store = NS(get=lambda identity: None, put=lambda *args, **kwargs: None)
        previous = real_compatibility.MATRIX.get("fibo2")
        real_compatibility.MATRIX["fibo2"] = dict(autolink.POLICY)
        try:
            selected, family, _ = resolver.resolve(str(paths["checkpoint"]), persist=False)
        finally:
            if previous is None:
                real_compatibility.MATRIX.pop("fibo2", None)
            else:
                real_compatibility.MATRIX["fibo2"] = previous
        assert family == "fibo2"
        assert str(paths["text_encoder_int8_convrot"]) in selected
        assert str(paths["vae"]) in selected
    finally:
        sys.path.remove(str(autolink_root))
