from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from types import SimpleNamespace as NS
from unittest.mock import patch
from threading import Event
import unittest
from pi_fibo_lib.progress import LiveProgress
from pi_fibo_lib.generate import generate

class ProgressTests(unittest.TestCase):
    def test_one_second_default_and_no_fake_steps(self):
        state = NS(interrupted=False, sampling_step=2, sampling_steps=6)
        progress = LiveProgress(state)
        self.assertEqual(progress.interval, 1.0)
        with progress:
            progress.phase("Decoding image")
            self.assertIn("Decoding image", state.textinfo)
            self.assertEqual(state.sampling_step, 2)
        self.assertFalse(progress._thread.is_alive())

    def test_background_refresh_stops_on_error(self):
        state = NS(interrupted=False, sampling_step=0, sampling_steps=6)
        refreshed = Event()
        class Probe(LiveProgress):
            def tick(self):
                super().tick()
                if self._thread is not None:
                    refreshed.set()
        progress = Probe(state, interval=0.01)
        with self.assertRaises(RuntimeError):
            with progress:
                self.assertTrue(refreshed.wait(1))
                raise RuntimeError("failed")
        self.assertFalse(progress._thread.is_alive())
        self.assertEqual(state.textinfo, "FIBO failed")

    def test_wrong_tab_is_rejected_before_model_loading(self):
        modules = NS(shared=NS(opts=NS()), processing=NS(), images=NS())
        with patch.dict(sys.modules, {"modules": modules, "backend": NS(memory_management=NS())}), \
             patch("pi_fibo_lib.generate.resolve", return_value={}), \
             patch("pi_fibo_lib.generate.resolve_folder") as load:
            for variant, request in (("edit_turbo", NS()), ("fibo15", NS(init_images=[])),
                                      ("edit", NS(init_images=[]))):
                with self.assertRaises(ValueError):
                    generate(request, variant, NS(), ".")
            load.assert_not_called()
