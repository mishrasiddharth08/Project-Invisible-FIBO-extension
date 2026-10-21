"""Regressions: stock argument forwarding, atomic failure, prompt identity and cache correctness.

The backup pinned host signatures, modified enum internals, silently truncated
captions and changed first-use settings. These tests assert replacement behavior.
"""
import copy
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pi_fibo_lib import bridge, catalog, prompts, runtime, settings, preset

def host():
    calls = []
    sentinel = object()
    class Runner:
        def run(self, p, *args, **kwargs):
            calls.append((self, p, args, kwargs))
            return sentinel
    class CI:
        def __init__(self, filename):
            self.filename = filename
            self.shorthash = None
        def register(self):
            sd.checkpoints_list[self.title] = self
            for key in self.ids:
                sd.checkpoint_aliases[key] = self
    def process(p, *args, **kwargs):
        calls.append((p, args, kwargs))
        return sentinel
    processing = NS(process_images=process, Processed=lambda p, images, seed=0, info="": images)
    sd = NS(checkpoints_list={}, checkpoint_aliases={}, CheckpointInfo=CI, list_models=lambda *a, **k: sentinel)
    shared = NS(opts=NS(sd_model_checkpoint="SDXL"), state=NS(nextjob=lambda: None))
    return NS(ScriptRunner=Runner), processing, sd, shared, calls, sentinel

class Contracts(unittest.TestCase):
    def test_stock_future_arguments_and_return_identity(self):
        scripts, processing, sd, shared, calls, sentinel = host()
        self.assertTrue(bridge.install(scripts, processing, sd, shared, "missing", generate=lambda *a: None, release=lambda: None))
        p = NS(override_settings={})
        runner = scripts.ScriptRunner()
        self.assertIs(runner.run(p, "future", example={"a": 1}), sentinel)
        self.assertEqual(calls[-1], (runner, p, ("future",), {"example": {"a": 1}}))
        self.assertIs(processing.process_images(p, 27, future=True), sentinel)
        self.assertEqual(calls[-1], (p, (27,), {"future": True}))
        self.assertIs(sd.list_models("future", x=2), sentinel)

    def test_changed_host_disables_atomically(self):
        scripts, processing, sd, shared, *_ = host()
        processing.process_images = lambda p, new_required: None
        originals = (scripts.ScriptRunner.run, processing.process_images, sd.list_models)
        self.assertFalse(bridge.install(scripts, processing, sd, shared, ".", generate=lambda *a: None, release=lambda: None))
        self.assertEqual(originals, (scripts.ScriptRunner.run, processing.process_images, sd.list_models))
        self.assertEqual(sd.checkpoints_list, {})

    def test_cleanup_failure_does_not_break_other_models(self):
        scripts, processing, sd, shared, _, sentinel = host()
        def fail():
            raise RuntimeError("simulated cleanup failure")
        bridge.install(scripts, processing, sd, shared, ".", generate=lambda *a: None, release=fail)
        self.assertIs(processing.process_images(NS()), sentinel)

    def test_known_aliases_only(self):
        scripts, processing, sd, shared, *_ = host()
        original = processing.process_images
        api, sibling = NS(process_images=original), NS(process_images=original)
        with patch.dict(sys.modules, {"modules.api.api": api, "other_extension": sibling}):
            bridge.install(scripts, processing, sd, shared, ".", generate=lambda *a: None, release=lambda: None)
        self.assertIs(api.process_images, processing.process_images)
        self.assertIs(sibling.process_images, original)

    def test_api_checkpoint_override_and_native_dispatch(self):
        scripts, processing, sd, shared, *_ = host()
        generated = []
        def generate(*args):
            generated.append(args)
            return "ours"
        bridge.install(scripts, processing, sd, shared, ".", generate=generate, release=lambda: None)
        p = NS(override_settings={"sd_model_checkpoint": "FIBO 1.5"})
        self.assertEqual(processing.process_images(p), "ours")
        self.assertEqual(generated[0][1], "fibo15")
        self.assertEqual(shared.opts.sd_model_checkpoint, "SDXL")
        shared.opts.sd_model_checkpoint = "FIBO 1.5"
        p.override_settings = {}
        self.assertEqual(scripts.ScriptRunner().run(p), "ours")

    def test_similar_filename_is_not_model_identity(self):
        self.assertIsNone(catalog.selected_variant("/models/fibo15.safetensors"))
        self.assertEqual(catalog.selected_variant(catalog.ROOT / "checkpoints/fibo15.json"), "fibo15")
        self.assertEqual(catalog.VARIANTS["fibo15"].repo, "briaai/Fibo-1.5")
        self.assertEqual(catalog.VARIANTS["fibo15"].steps, 6)

    def test_selected_script_runs_before_fibo_takeover(self):
        scripts, processing, sd, shared, calls, sentinel = host()
        def forbidden(*args):
            self.fail("A selected script must not be silently bypassed")
        bridge.install(scripts, processing, sd, shared, ".", generate=forbidden, release=lambda: None)
        shared.opts.sd_model_checkpoint = "FIBO 1.5"
        p = NS(override_settings={})
        runner = scripts.ScriptRunner()
        self.assertIs(runner.run(p, 2, "script-input", future=True), sentinel)
        self.assertEqual(calls[-1], (runner, p, (2, "script-input"), {"future": True}))

    def test_json_is_preserved_and_malformed_json_refused(self):
        text = '{ "short_description": "natural skin", "custom": {"a": 1} }'
        self.assertEqual(prompts.prepare(text), text)
        for bad in ('{"broken":', '["not an object"]', '```json\n{}\n```'):
            with self.assertRaises(ValueError):
                prompts.prepare(bad)

    def test_plain_language_is_not_secretly_embellished(self):
        self.assertEqual(json.loads(prompts.prepare("a paper sculpture")), {"short_description": "a paper sculpture"})
        self.assertEqual(json.loads(prompts.prepare("warmer light", edit=True))["edit_instruction"], "warmer light")

    def test_overlong_caption_fails_without_truncation(self):
        observed = []
        def tokenizer(text, **kw):
            observed.append(kw)
            return {"input_ids": list(range(3001))}
        with self.assertRaisesRegex(ValueError, "Nothing was truncated"):
            prompts.check_budget(tokenizer, "caption")
        self.assertFalse(observed[0]["truncation"])

    def test_preset_choices_leave_enum_and_siblings_intact(self):
        from enum import Enum
        class Arch(Enum):
            sd = 1
            @staticmethod
            def choices(*args, **kwargs):
                return ["sd", "ideogram4"]
        before = dict(Arch.__members__)
        preset._wrap_choices(Arch)
        preset._wrap_choices(Arch)
        self.assertEqual(Arch.choices(), ["sd", "ideogram4", "fibo"])
        self.assertEqual(dict(Arch.__members__), before)

    def test_memory_route_uses_actual_available_space(self):
        gb = 1024**3
        self.assertEqual(runtime.choose_memory(30*gb, 16*gb, "Auto"), "model")
        self.assertEqual(runtime.choose_memory(12*gb, 16*gb, "Auto"), "group")
        self.assertEqual(runtime.choose_memory(30*gb, 16*gb, "Low VRAM"), "group")

    def test_size_change_reuses_offloaded_model(self):
        cached = NS(_pi_memory_route="model", _pi_max_pixels=1024**2)
        key = ("folder", "fibo15", "Auto", ())
        with patch.object(runtime, "_pipe", cached), patch.object(runtime, "_key", key), \
             patch.object(runtime, "file_identity", return_value=()), patch.object(runtime, "release") as release:
            self.assertIs(runtime.acquire("folder", "fibo15", pixels=1536**2), cached)
            release.assert_not_called()

    def test_prompt_cache_exact_bounded_and_independent_of_mutation(self):
        import torch
        calls = []
        class Pipe:
            text_encoder = NS(dtype=torch.float32)
            _execution_device = "cpu"
            def get_prompt_embeds(self, prompt, num_images_per_prompt=1, max_sequence_length=3000, device=None, dtype=None):
                calls.append(prompt)
                return (torch.arange(6).reshape(1, 2, 3).float(), (torch.ones(1, 2, 3),), torch.ones(1, 2))
        pipe = Pipe()
        runtime.install_prompt_cache(pipe, max_bytes=128)
        original = pipe.get_prompt_embeds("one")
        hit = pipe.get_prompt_embeds("one")
        torch.testing.assert_close(original[0], hit[0], rtol=0, atol=0)
        hit[0].zero_()
        torch.testing.assert_close(original[0], pipe.get_prompt_embeds("one")[0], rtol=0, atol=0)
        self.assertEqual(calls, ["one"])
        pipe.get_prompt_embeds("two")
        pipe.get_prompt_embeds("three")
        self.assertLessEqual(sum(size for _, size in pipe._pi_prompt_cache.values()), 128)
        self.assertLessEqual(len(pipe._pi_prompt_cache), 2)

    def test_stop_callback_does_not_retry(self):
        from pi_fibo_lib.generate import step_callback, SkipImage
        state = NS(interrupted=True, skipped=False, sampling_steps=6)
        with self.assertRaises(InterruptedError):
            step_callback(state)(None, 0, 1, {})
        state.interrupted, state.skipped = False, True
        with self.assertRaises(SkipImage):
            step_callback(state)(None, 0, 1, {})

    def test_user_settings_remain_unchanged(self):
        p = NS(width=1024, height=1024, steps=7, cfg_scale=1, batch_size=2, n_iter=3,
               sampler_name="Euler", scheduler="Simple", override_settings={})
        before = copy.deepcopy(vars(p))
        settings.resolve(p, NS())
        self.assertEqual(vars(p), before)

    def test_missing_model_has_actionable_error(self):
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaisesRegex(FileNotFoundError, "No downloads were started"):
                catalog.resolve_folder("fibo15", temp)

if __name__ == "__main__":
    unittest.main(verbosity=2)
