import sys
import unittest
from pathlib import Path

from .official import BriaFibo2Pipeline, BriaFibo2Transformer2DModel, SUPPORTED_MODELS, official_classes


class VendorImportTests(unittest.TestCase):
    def test_imports_without_replacing_diffusers(self):
        import diffusers

        self.assertIs(BriaFibo2Pipeline.__module__.startswith("forge_fibo2.vendor."), True)
        self.assertIs(BriaFibo2Transformer2DModel.__module__.startswith("forge_fibo2.vendor."), True)
        self.assertNotIn("forge_fibo2/vendor", Path(diffusers.__file__).as_posix())
        self.assertIs(sys.modules["diffusers"], diffusers)
        self.assertEqual(
            set(official_classes()),
            {"pipeline", "transformer", "vae", "scheduler", "text_encoder", "processor"},
        )

    def test_tiny_transformer_constructs(self):
        model = BriaFibo2Transformer2DModel(
            in_channels=1,
            patch_size=2,
            dim=32,
            n_layers=2,
            n_refiner_layers=1,
            n_heads=2,
            cap_feat_dim=48,
            injection_layer_ids=(1,),
            perceiver_num_layers=1,
            min_num_gist_tokens=4,
            max_num_gist_tokens=8,
            gist_step=4,
            gist_min_text_len=4,
            gist_max_text_len=16,
            axes_dims=(4, 6, 6),
            axes_lens=(16, 16, 16),
        )
        self.assertEqual(model.config.n_layers, 2)
        self.assertEqual(len(SUPPORTED_MODELS), 2)


if __name__ == "__main__":
    unittest.main()
