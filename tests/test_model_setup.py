from pathlib import Path
from types import SimpleNamespace as NS
import hashlib, json, sys, tempfile, unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pi_fibo_lib import model_setup

class Tests(unittest.TestCase):
    def test_catalog_is_single_source(self):
        from pi_fibo_lib.catalog import VARIANTS
        self.assertIs(model_setup.VARIANTS, VARIANTS)
        self.assertEqual(set(model_setup.LABEL_TO_KEY.values()), set(VARIANTS))

    def test_manual_default(self):
        text = model_setup.manual_instructions(model_setup.VARIANTS["fibo15"].label, "X:/Fibo")
        self.assertIn("Manual setup (recommended)", text)
        self.assertIn("normal Hugging Face login", text)

    def test_success_maps_folder(self):
        with tempfile.TemporaryDirectory() as temp:
            root, target = Path(temp)/"extension", Path(temp)/"models"/"Fibo"
            root.mkdir()
            payloads = {"model_index.json": b"{}", "model.safetensors": b"safe"}
            files = [NS(rfilename=n, size=len(d), lfs=NS(sha256=hashlib.sha256(d).hexdigest()))
                     for n, d in payloads.items()]
            hub = NS(get_token=lambda: "hidden",
                     HfApi=lambda: NS(model_info=lambda *a, **k: NS(siblings=files, sha="commit")),
                     snapshot_download=lambda *a, **k: [(Path(k["local_dir"])/n).write_bytes(d) for n,d in payloads.items()])
            old_hub, old_root, old_validate = sys.modules.get("huggingface_hub"), model_setup.ROOT, model_setup.validate_folder
            checked = []
            sys.modules["huggingface_hub"] = hub
            model_setup.ROOT, model_setup.validate_folder = root, lambda folder,key: checked.append((folder,key))
            try:
                self.assertEqual(model_setup.download_snapshot("fibo15", target), target.resolve())
                self.assertEqual(json.loads((root/"model_locations.json").read_text())["fibo15"], str(target.resolve()))
                self.assertEqual(checked, [(target.resolve(), "fibo15")])
                self.assertFalse(list(root.glob("*.tmp")))
            finally:
                model_setup.ROOT, model_setup.validate_folder = old_root, old_validate
                if old_hub is None: sys.modules.pop("huggingface_hub", None)
                else: sys.modules["huggingface_hub"] = old_hub

    def test_gate_error_hides_secret(self):
        class GatedRepoError(Exception): pass
        original = model_setup.download_snapshot
        model_setup.download_snapshot = lambda *a: (_ for _ in ()).throw(GatedRepoError("secret-token"))
        try:
            result = model_setup._download_from_ui(model_setup.VARIANTS["fibo15"].label, "X:/Fibo")
            self.assertIn("Access required", result)
            self.assertNotIn("secret-token", result)
        finally: model_setup.download_snapshot = original

    def test_existing_folder_is_registered(self):
        with tempfile.TemporaryDirectory() as temp:
            root, target = Path(temp)/"extension", Path(temp)/"existing"
            root.mkdir(); target.mkdir()
            old_root, old_validate = model_setup.ROOT, model_setup.validate_folder
            model_setup.ROOT, model_setup.validate_folder = root, lambda *a: None
            try:
                result = model_setup._use_existing_from_ui(model_setup.VARIANTS["fibo15"].label, str(target))
                self.assertIn("validated and registered", result)
                self.assertEqual(json.loads((root/"model_locations.json").read_text())["fibo15"], str(target.resolve()))
            finally: model_setup.ROOT, model_setup.validate_folder = old_root, old_validate

    def test_refuses_extension_destination(self):
        with tempfile.TemporaryDirectory() as temp:
            old = model_setup.ROOT
            model_setup.ROOT = Path(temp)/"extension"; model_setup.ROOT.mkdir()
            try:
                with self.assertRaises(RuntimeError): model_setup._safe_destination(model_setup.ROOT)
            finally: model_setup.ROOT = old

if __name__ == "__main__": unittest.main()
