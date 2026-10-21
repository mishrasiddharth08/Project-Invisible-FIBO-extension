from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch
import numpy as np
import torch
from PIL import Image
from pi_fibo_lib.degrid import process_image
from pi_fibo_lib.preview import DevelopingPreview, publish

class DeGridPreviewTests(unittest.TestCase):
    def test_off_returns_identical_object(self):
        image = Image.new("RGB", (32, 32), (71, 93, 150))
        self.assertIs(process_image(image, "Off")[0], image)

    def test_auto_preserves_clean_image(self):
        image = Image.new("RGB", (32, 32), (71, 93, 150))
        output, info = process_image(image, "Auto")
        self.assertIs(output, image)
        self.assertIn("Skipped", info["FIBO DeGrid result"])

    def test_optional_cleanup_failure_keeps_original(self):
        image = Image.new("RGB", (32, 32), "red")
        with patch("pi_fibo_lib.vendor.degrid_core.degrid", side_effect=RuntimeError("failed")):
            output, info = process_image(image, "Auto")
        self.assertIs(output, image)
        self.assertIn("Skipped", info["FIBO DeGrid result"])

    def test_reduces_known_grid_without_changing_input(self):
        y, x = np.indices((64, 64))
        original = np.repeat((0.5 + 0.01 * (-1.) ** (x + y))[..., None], 3, axis=2).astype(np.float32)
        before = original.copy()
        output, info = process_image(original, "Auto")
        after = np.asarray(output) / 255.
        np.testing.assert_array_equal(original, before)
        self.assertLess(after[8:-8, 8:-8].std(), original.std() / 2)
        self.assertEqual(info["FIBO DeGrid result"], "Applied")

    def test_preview_cannot_mutate_sampler_and_skips_duplicate_final_decode(self):
        state = NS(sampling_steps=3, sampling_step=1, id_live_preview=0)
        latents = torch.full((1, 4, 3), 0.4)
        velocity = torch.full((1, 4, 3), 0.1)
        original = latents.clone()
        decode_calls = []
        def decode(x, **kwargs):
            decode_calls.append(x)
            return (x,)
        pipe = NS(scheduler=NS(sigmas=torch.tensor([1., 0.5, 0.]), step_index=1),
                  vae_scale_factor=16, vae=NS(dtype=torch.float32, config=NS(latents_mean=[0]*3, latents_std=[1]*3), decode=decode),
                  _unpack_latents_no_patch=lambda x, *a: x.view(1, 2, 2, 3).permute(0, 3, 1, 2),
                  image_processor=NS(postprocess=lambda *a, **kw: [Image.new("RGB", (32, 32))]))
        preview = DevelopingPreview(state, 32, 32, 1)
        preview.capture(None, (), (velocity,))
        preview.update(pipe, 0, {"latents": latents})
        torch.testing.assert_close(latents, original, rtol=0, atol=0)
        self.assertEqual(state.id_live_preview, 1)
        preview.capture(None, (), (velocity,))
        preview.update(pipe, 2, {"latents": latents})
        self.assertEqual(len(decode_calls), 1)
        final = Image.new("RGB", (32, 32), "red")
        publish(state, final)
        self.assertIs(state.current_image, final)
        self.assertEqual(state.id_live_preview, 2)
