import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from pi_fibo_lib.catalog import validate_folder


class OfficialFibo15ConfigTests(unittest.TestCase):
    def test_official_internal_alias_requires_pipeline_and_architecture(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)

            def write(name, data):
                p = root / name
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_text(json.dumps(data), encoding="utf-8")

            manifest = {"_class_name": "BriaFiboPipeline",
                        "transformer": ["diffusers", "BriaFiboTransformer2DModel"]}
            config = {"_class_name": "Bria4Transformer2DModel", "in_channels": 48,
                      "joint_attention_dim": 4096, "text_encoder_dim": 2048,
                      "num_layers": 8, "num_single_layers": 38,
                      "num_attention_heads": 24, "attention_head_dim": 128,
                      "patch_size": 1, "guidance_embeds": False}
            write("model_index.json", manifest)
            write("transformer/config.json", config)
            write("text_encoder/config.json", {"model_type": "smollm3"})
            write("vae/config.json", {"_class_name": "AutoencoderKLWan", "z_dim": 48,
                                      "latents_mean": [0] * 48, "latents_std": [1] * 48})
            write("tokenizer/tokenizer.json", {})
            write("scheduler/scheduler_config.json", {"_class_name": "FlowMatchEulerDiscreteScheduler"})
            for component in ("transformer", "text_encoder", "vae"):
                (root / component / "test.safetensors").write_bytes(b"0" * 16)

            self.assertEqual(validate_folder(root, "fibo15"), root)
            with self.assertRaisesRegex(ValueError, "transformer is not FIBO"):
                validate_folder(root, "base")
            write("transformer/config.json", {**config, "in_channels": 16})
            with self.assertRaisesRegex(ValueError, "transformer is not FIBO"):
                validate_folder(root, "fibo15")
            write("transformer/config.json", config)
            write("model_index.json", {**manifest, "transformer": ["diffusers", "OtherModel"]})
            with self.assertRaisesRegex(ValueError, "transformer is not FIBO"):
                validate_folder(root, "fibo15")
            write("model_index.json", manifest)
            (root / "transformer" / "test.safetensors").unlink()
            with self.assertRaisesRegex(FileNotFoundError, "Missing or incomplete transformer"):
                validate_folder(root, "fibo15")

