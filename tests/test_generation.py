"""Exercise the actual native generation owner with a fake model only.

Regression: a completed image must survive Stop during the next image. Saving
must use FIBO metadata and restore the original shared model after failure.
"""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from types import SimpleNamespace as NS
from unittest.mock import patch
import unittest
from PIL import Image
from pi_fibo_lib.generate import generate
from pi_fibo_lib.saving import save_completed, finish_gallery

class Generation(unittest.TestCase):
    def test_native_grid_output_and_separate_save_preferences(self):
        original = object()
        saved = []
        shared = NS(sd_model=original, opts=NS(samples_format="png", return_grid=True,
                    grid_save=True, grid_only_if_multiple=True, enable_pnginfo=True,
                    grid_format="jpg", grid_extended_filename=False))
        p = NS(outpath_samples="samples", outpath_grids="grids", sd_model_hash=None,
               batch_size=2, do_not_save_grid=False, do_not_save_samples=True)
        completed = [Image.new("RGB", (32, 32)), Image.new("RGB", (32, 32))]
        images = NS(image_grid=lambda imgs, size: Image.new("RGB", (64, 32)),
                    save_image=lambda *a, **kw: saved.append((a, kw)))
        gallery, infos, first = finish_gallery(images, shared, completed, ["one", "two"],
                    p, NS(name_for_extra="FIBO"), [41, 42], ["a", "b"])
        self.assertEqual(len(gallery), 3)
        self.assertEqual(first, 1)
        self.assertIs(gallery[1], completed[0])
        self.assertEqual(infos, ["one", "one", "two"])
        self.assertEqual(saved[0][0][1:3], ("grids", "grid"))
        self.assertEqual(saved[0][1]["extension"], "jpg")
        self.assertTrue(saved[0][1]["grid"])
        self.assertIs(shared.sd_model, original)
        p.do_not_save_grid = True
        gallery, infos, first = finish_gallery(images, shared, completed, ["one", "two"],
                    p, NS(), [41, 42], ["a", "b"])
        self.assertEqual(len(gallery), 2)
        self.assertEqual(first, 0)
        self.assertEqual(len(saved), 1)

    def test_grid_without_completed_images_is_never_attempted(self):
        gallery, infos, first = finish_gallery(NS(), NS(opts=NS()), [], [], NS(), NS(), [], [])
        self.assertEqual((gallery, infos, first), ([], [], 0))

    def test_save_restores_original_model_on_error(self):
        original = object()
        shared = NS(sd_model=original, opts=NS(samples_format="png"))
        p = NS(sd_model_hash="fibo-hash", outpath_samples="unused")
        ci = NS(name_for_extra="FIBO")
        def save(*args, **kwargs):
            self.assertIs(shared.sd_model.sd_checkpoint_info, ci)
            raise OSError("disk full")
        with self.assertRaises(OSError):
            save_completed(NS(save_image=save), shared, None, p, ci, 1, "prompt", "info")
        self.assertIs(shared.sd_model, original)

    def test_completed_images_survive_stop_and_preserve_seeds(self):
        self._run_case()

    def test_img2img_edit_with_mask_preserves_image_and_instruction(self):
        self._run_case(edit=True)

    def test_degrid_result_is_published_and_recorded_before_save(self):
        self._run_case(degrid=True)

    def _run_case(self, edit=False, degrid=False):
        original = object()
        saved = []
        calls = []
        state = NS(interrupted=False, skipped=False, stopping_generation=False,
                   job_no=0, id_live_preview=0)
        def nextjob():
            state.job_no += 1
        state.nextjob = nextjob
        opts = NS(samples_save=True, samples_format="png", live_previews_enable=True,
                  pi_fibo_memory="Auto", pi_fibo_degrid="Auto" if degrid else "Off")
        shared = NS(state=state, opts=opts, sd_model=original)
        p = NS(width=32, height=32, steps=6, cfg_scale=1, batch_size=1, n_iter=3,
               seed=91, subseed=0, prompt="a natural portrait", negative_prompt="",
               override_settings={}, sampler_name="Euler", scheduler="Simple",
               extra_generation_params={}, outpath_samples="unused", do_not_save_samples=False)
        def setup():
            p.all_prompts = [p.prompt] * 3
            p.all_negative_prompts = [""] * 3
        p.setup_prompts = setup
        if edit:
            p.init_images = [Image.new("RGB", (32, 32), "red")]
            p.image_mask = Image.new("L", (32, 32), 255)
        def save(image, path, basename, **kwargs):
            self.assertIsNone(shared.sd_model.sd_model_hash)
            self.assertIs(state.current_image, image)
            if degrid:
                self.assertIn("FIBO DeGrid: Auto", kwargs["info"])
                self.assertIn("Skipped: no grid detected", kwargs["info"])
            saved.append((image, kwargs))
        class Pipe:
            _pi_memory_route = "model"
            tokenizer = staticmethod(lambda *args, **kw: {"input_ids": [1]})
            def __call__(self, **kwargs):
                calls.append(kwargs)
                if len(calls) == 2:
                    state.interrupted = True
                kwargs["callback_on_step_end"](self, 0, 1000, {})
                return NS(images=[Image.new("RGB", (32, 32))])
            def maybe_free_model_hooks(self):
                pass
        def processed(p, completed, **kwargs):
            return NS(images=completed, **kwargs)
        modules = NS(shared=shared, processing=NS(get_fixed_seed=lambda x: x, Processed=processed), images=NS(save_image=save))
        backend = NS(memory_management=NS(unload_all_models=lambda: None))
        with patch.dict(sys.modules, {"modules": modules, "backend": backend,
                                      "modules.infotext_utils": NS(quote=str)}), \
             patch("pi_fibo_lib.generate.resolve_folder", return_value=Path("unused")), \
             patch("pi_fibo_lib.runtime.acquire", return_value=Pipe()), \
             patch("torch.cuda.synchronize"), patch("torch.cuda.reset_peak_memory_stats"), \
             patch("torch.cuda.max_memory_allocated", return_value=0):
            result = generate(p, "edit_turbo" if edit else "fibo15", NS(shorthash="pointer"), ".")
        self.assertEqual(len(result.images), 1)
        self.assertEqual(result.all_seeds, [91])
        self.assertEqual(len(saved), 1)
        self.assertEqual(len(calls), 2)
        self.assertEqual(calls[0]["generator"].initial_seed(), 91)
        self.assertEqual(calls[1]["generator"].initial_seed(), 92)
        self.assertIs(shared.sd_model, original)
        self.assertTrue(state.interrupted)
        self.assertEqual(p.steps, 6)
        self.assertEqual(p.n_iter, 3)
        if edit:
            import json
            self.assertEqual(json.loads(calls[0]["prompt"])["edit_instruction"], p.prompt)
            self.assertEqual(calls[0]["image"].getpixel((0, 0)), (255, 0, 0))
            self.assertEqual(calls[0]["mask"].getpixel((0, 0)), 255)

if __name__ == "__main__":
    unittest.main(verbosity=2)
