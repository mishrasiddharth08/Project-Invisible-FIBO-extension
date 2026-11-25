"""No-download setup and converter CLI regression checks."""
import ast
import hashlib
import importlib.util
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import Mock, patch

STAGE = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("setup_fibo2", STAGE / "setup_fibo2.py")
setup = importlib.util.module_from_spec(spec)
spec.loader.exec_module(setup)
READY_REAL = setup._ready


class SetupTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.host = Path(self.tmp.name)
        for name in ("backend", "modules_forge"):
            (self.host / name).mkdir()
        self.root = self.host / "extensions" / "fibo"
        self.root.mkdir(parents=True)
        self.payload = b"pinned-test-weight"
        self.item = types.SimpleNamespace(rfilename="weight.bin", size=len(self.payload),
            lfs=types.SimpleNamespace(sha256=hashlib.sha256(self.payload).hexdigest()))
        self.api = Mock()
        self.api.model_info.return_value = types.SimpleNamespace(siblings=[self.item])
        self.download = Mock(side_effect=self.write_snapshot)
        self.relocate = Mock()
        self.modules = {
            "forge_fibo2": types.ModuleType("forge_fibo2"),
            "forge_fibo2.assets": types.SimpleNamespace(REPOSITORY="pinned/repo", REVISION="fixed-revision", relocate=self.relocate),
            "huggingface_hub": types.SimpleNamespace(HfApi=lambda: self.api, get_token=lambda: None, snapshot_download=self.download)}
        self.ready = patch.object(setup, "_ready", return_value=False).start()
        self.addCleanup(patch.stopall)
        patch.object(setup, "__file__", str(self.root / "setup_fibo2.py")).start()
        patch.dict(sys.modules, self.modules).start()
        patch.object(setup.shutil, "disk_usage", return_value=types.SimpleNamespace(free=10**14)).start()

    def write_snapshot(self, repo, **kwargs):
        snapshot = kwargs["local_dir"]
        snapshot.mkdir(parents=True)
        (snapshot / "weight.bin").write_bytes(self.payload)

    def test_check_never_downloads(self):
        self.assertEqual(setup.main(["--check"]), 0)
        self.download.assert_not_called()
        self.relocate.assert_not_called()
        self.api.model_info.assert_called_once_with("pinned/repo", revision="fixed-revision", token=None, files_metadata=True)

    def test_ready_manifest_skips_network(self):
        path = self.root / "model-manifest.json"
        path.write_text("original")
        self.ready.return_value = True
        self.assertEqual(setup.main([]), 0)
        self.api.model_info.assert_not_called()
        self.download.assert_not_called()
        self.assertEqual(path.read_text(), "original")

    def test_stale_shipped_manifest_replaced_after_validation(self):
        source = STAGE / "tests" / "fixtures" / "shipped-model-manifest.json"
        original = source.read_text()
        path = self.root / "model-manifest.json"
        path.write_text(original)
        self.ready.side_effect = [False, True]
        def relocated(snapshot, host, manifest):
            manifest.write_text("replacement")
        self.relocate.side_effect = relocated
        self.assertEqual(setup.main([]), 0)
        self.assertEqual(path.read_text(), "replacement")
        backups = list(self.root.glob("model-manifest.json.backup-*"))
        self.assertEqual(len(backups), 1)
        self.assertEqual(backups[0].read_text(), original)

    def test_failed_relocation_preserves_existing_manifest(self):
        path = self.root / "model-manifest.json"
        path.write_text("original")
        self.relocate.side_effect = RuntimeError("conflicting model")
        with self.assertRaisesRegex(RuntimeError, "conflicting model"):
            setup.main([])
        self.assertEqual(path.read_text(), "original")

    def test_verified_snapshot_relocated(self):
        self.ready.return_value = True
        self.relocate.side_effect = lambda snapshot, host, manifest: manifest.write_text("new")
        self.assertEqual(setup.main([]), 0)
        self.assertEqual(self.download.call_args.kwargs["revision"], "fixed-revision")
        self.relocate.assert_called_once()

    def test_hash_mismatch_blocks_relocation(self):
        self.item.lfs.sha256 = "0" * 64
        with self.assertRaisesRegex(RuntimeError, "SHA256 mismatch"):
            setup.main([])
        self.relocate.assert_not_called()

    def test_no_space_never_downloads(self):
        with patch.object(setup.shutil, "disk_usage", return_value=types.SimpleNamespace(free=0)):
            with self.assertRaisesRegex(RuntimeError, "space"):
                setup.main([])
        self.download.assert_not_called()

    def test_real_readiness_checks_configs_and_stale_paths(self):
        import json
        def validate(path, role):
            if not path.is_file():
                raise RuntimeError("missing weight")
        def directory(data, role):
            return self.root / role
        self.modules["forge_fibo2.runtime"] = types.SimpleNamespace(
            _validate_metadata=validate, _runtime_dir=directory)
        self.modules["forge_fibo2.provenance"] = types.SimpleNamespace(
            PINS={"repository": "official/repo", "revision": "official-revision"})
        sys.modules.update(self.modules)
        data = {"schema": 1, "family": "fibo2", "repository": "pinned/repo",
                "revision": "fixed-revision", "components": {}}
        for key in ("checkpoint", "text_encoder_bf16", "vae"):
            weight = self.root / (key + ".weights")
            weight.write_bytes(b"mock validated weights")
            data["components"][key] = {"path": str(weight)}
        for role, filename in (("transformer", "config.json"), ("text_encoder", "config.json"),
                               ("vae", "config.json"), ("processor", "processor_config.json"),
                               ("scheduler", "scheduler_config.json")):
            directory(data, role).mkdir(exist_ok=True)
            (directory(data, role) / filename).write_text("{}")
        (self.root / "fonts").mkdir()
        for name in ("Aileron-Regular.otf", "DejaVuSans.ttf"):
            (self.root / "fonts" / name).write_bytes(b"font")
        manifest = self.root / "model-manifest.json"
        manifest.write_text(json.dumps(data))
        self.assertTrue(READY_REAL(manifest))
        (self.root / "transformer" / "config.json").unlink()
        self.assertFalse(READY_REAL(manifest))
        (self.root / "transformer" / "config.json").write_text("{}")
        data["components"]["checkpoint"]["path"] = str(self.root / "missing.weights")
        manifest.write_text(json.dumps(data))
        self.assertFalse(READY_REAL(manifest))

    def test_conversion_variant_is_recovered_and_unknown_source_rejected(self):
        tree = ast.parse((STAGE / "forge_fibo2/runtime.py").read_text())
        fn = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "_validate_metadata")
        source = {"pi_fibo2_component": "dit", "pi_fibo2_format": "int8_convrot",
                  "pi_fibo2_source_sha256": "known", "pi_fibo2_source_variant": "dev"}
        namespace = {"Path": Path, "RuntimeAssetError": RuntimeError,
                     "_metadata": lambda path: (source, ["layer.weight"])}
        sys.modules["forge_fibo2.provenance"] = types.SimpleNamespace(PINS={"assets": [
            {"role": "transformer", "sha256": "known", "variant": "dev"}]})
        namespace["__package__"] = "forge_fibo2"
        exec(compile(ast.Module(body=[fn], type_ignores=[]), "<variant>", "exec"), namespace)
        meta, keys = namespace["_validate_metadata"](Path("converted"), "transformer")
        self.assertEqual(meta["verified_official_variant"], "dev")
        source["pi_fibo2_source_sha256"] = "unknown"
        with self.assertRaisesRegex(RuntimeError, "source identity"):
            namespace["_validate_metadata"](Path("converted"), "transformer")

    def test_converter_offers_only_implemented_format(self):
        tree = ast.parse((STAGE / "forge_fibo2" / "converter.py").read_text())
        targets = next(node.value for node in tree.body if isinstance(node, ast.Assign)
            and any(isinstance(name, ast.Name) and name.id == "TARGETS" for name in node.targets))
        self.assertEqual(ast.literal_eval(targets), ("int8_convrot",))


if __name__ == "__main__":
    unittest.main()
