import json
from pathlib import Path
import tempfile
import unittest

import torch
from transformers import Qwen3VLForConditionalGeneration, Qwen3VLProcessor, Qwen3VLConfig
from transformers.modeling_utils import no_init_weights

from forge_fibo2.runtime import RuntimeAssetError, _qwen_config, _strict_key_check, install_hidden_encoder, _BoundedProcessor


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "tests" / "fixtures" / "configs" / "text_encoder"
PROCESSOR = ROOT / "models" / "Fibo-2.0-edit-turbo-bf16" / "processor"


class RuntimeTests(unittest.TestCase):
    def test_overflow_is_rejected_instead_of_silent_truncation(self):
        seen=[]
        def processor(**kwargs):
            seen.append(kwargs)
            return {'input_ids':torch.zeros(1,12,dtype=torch.long)}
        with self.assertRaisesRegex(RuntimeAssetError,'No tokens were silently removed'):
            _BoundedProcessor(processor)(max_length=8,truncation=True)
        self.assertFalse(seen[0]['truncation'])

    def test_hidden_only_optimization_preserves_used_layers_and_skips_vocab_head(self):
        config=Qwen3VLConfig(text_config={'vocab_size':64,'hidden_size':32,'intermediate_size':64,
            'num_hidden_layers':2,'num_attention_heads':2,'num_key_value_heads':1,'head_dim':16,
            'rope_scaling':{'rope_type':'default','mrope_section':[2,3,3]}},
            vision_config={'depth':1,'hidden_size':32,'intermediate_size':64,'num_heads':2,
                'out_hidden_size':32,'deepstack_visual_indexes':[0]})
        model=Qwen3VLForConditionalGeneration(config).eval()
        ids=torch.tensor([[1,2,3,4]])
        calls=[]
        handle=model.lm_head.register_forward_hook(lambda *_:calls.append(True))
        with torch.inference_mode():
            before=model(input_ids=ids,output_hidden_states=True,use_cache=True,return_dict=True)
            self.assertEqual(len(calls),1)
            self.assertTrue(install_hidden_encoder(model,[0,1]))
            after=model(input_ids=ids,output_hidden_states=True,use_cache=True,return_dict=True)
        self.assertEqual(len(calls),1)
        self.assertIsNone(after.past_key_values)
        for left,right in zip(before.hidden_states[:2],after.hidden_states[:2]):
            self.assertTrue(torch.equal(left,right))
        handle.remove()

    @unittest.skipUnless((CONFIG/'config.json').is_file(), 'Run model setup for real configuration compatibility checks')
    def test_transformers_457_qwen_config_compatibility(self):
        config = _qwen_config(CONFIG)
        self.assertEqual(config.text_config.rope_scaling["mrope_section"], [24, 20, 20])
        with no_init_weights(), torch.device("meta"):
            model = Qwen3VLForConditionalGeneration(config)
        self.assertEqual(len(model.model.language_model.layers), 36)

    @unittest.skipUnless((PROCESSOR/'tokenizer.json').is_file(), 'Run model setup for real tokenizer compatibility checks')
    def test_transformers_457_processor_compatibility(self):
        processor = Qwen3VLProcessor.from_pretrained(
            str(PROCESSOR), local_files_only=True, use_fast=True, extra_special_tokens={}
        )
        self.assertEqual(processor.image_token, "<|image_pad|>")

    def test_strict_key_check_rejects_missing(self):
        model = torch.nn.Linear(4, 4)
        with self.assertRaises(RuntimeAssetError):
            _strict_key_check(model, ["weight"], "transformer", False)

    def test_strict_key_check_accepts_exact(self):
        model = torch.nn.Linear(4, 4)
        _strict_key_check(model, list(model.state_dict()), "transformer", False)


if __name__ == "__main__":
    unittest.main()
