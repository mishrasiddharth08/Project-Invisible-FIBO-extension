"""Actual installed FIBO math with tiny RANDOM weights: cache/offload equivalence.

This proves plumbing and numeric preservation, NOT trained-model picture quality,
full-checkpoint loading, memory requirements, or a speed claim.
"""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import PreTrainedTokenizerFast, SmolLM3Config, SmolLM3ForCausalLM
from diffusers import BriaFiboPipeline, BriaFiboTransformer2DModel, AutoencoderKLWan, FlowMatchEulerDiscreteScheduler
from pi_fibo_lib.runtime import install_prompt_cache, install_hidden_state_encoder

def main():
    torch.manual_seed(1024)
    torch.set_num_threads(2)
    tokenizer_core = Tokenizer(WordLevel({"[UNK]": 0, "[PAD]": 1, "a": 2, "portrait": 3, "natural": 4}, unk_token="[UNK]"))
    tokenizer_core.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=tokenizer_core, unk_token="[UNK]", pad_token="[PAD]")
    te = SmolLM3ForCausalLM(SmolLM3Config(vocab_size=32, hidden_size=16, intermediate_size=32,
        num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=2,
        bos_token_id=0, eos_token_id=0, pad_token_id=1))
    transformer = BriaFiboTransformer2DModel(in_channels=4, num_layers=1, num_single_layers=1,
        attention_head_dim=8, num_attention_heads=2, joint_attention_dim=32,
        axes_dims_rope=[2, 2, 4], text_encoder_dim=16)
    vae = AutoencoderKLWan(base_dim=16, z_dim=4, dim_mult=[1, 1, 1, 1], num_res_blocks=1,
                           latents_mean=[0.0] * 4, latents_std=[1.0] * 4)
    pipe = BriaFiboPipeline(transformer=transformer, scheduler=FlowMatchEulerDiscreteScheduler(shift=3.0),
                            vae=vae, text_encoder=te, tokenizer=tokenizer)
    pipe.set_progress_bar_config(disable=True)
    pipe.to("cuda", dtype=torch.bfloat16)
    
    def run(callback=None):
        with torch.inference_mode():
            return pipe(prompt="a natural portrait", width=32, height=32, num_inference_steps=3,
                        guidance_scale=1, generator=torch.Generator(device="cpu").manual_seed(77),
                        output_type="latent", max_sequence_length=32, callback_on_step_end=callback).images.detach().cpu()
    
    reference = run()
    from types import SimpleNamespace
    from pi_fibo_lib.preview import DevelopingPreview
    from pi_fibo_lib.generate import step_callback
    state = SimpleNamespace(interrupted=False, skipped=False, sampling_step=0, sampling_steps=3, id_live_preview=0)
    preview = DevelopingPreview(state, 32, 32, 1)
    handle = pipe.transformer.register_forward_hook(preview.capture)
    try:
        with_preview = run(step_callback(state, preview=preview))
    finally:
        handle.remove()
    assert preview.frames > 0, "Real VAE preview did not decode"
    torch.testing.assert_close(reference, with_preview, rtol=0, atol=0)
    print("PASS: real tiny VAE previews leave final sampled latents bit-identical")
    install_hidden_state_encoder(pipe.text_encoder)
    hidden_only = run()
    torch.testing.assert_close(reference, hidden_only, rtol=0, atol=0)
    install_prompt_cache(pipe)
    cold, warm = run(), run()
    torch.testing.assert_close(reference, cold, rtol=0, atol=0)
    torch.testing.assert_close(reference, warm, rtol=0, atol=0)
    pipe.enable_model_cpu_offload(device="cuda")
    offloaded = run()
    torch.testing.assert_close(reference, offloaded, rtol=0, atol=0)
    pipe.remove_all_hooks()
    pipe.to("cpu")
    pipe.enable_group_offload(onload_device=torch.device("cuda"), offload_device=torch.device("cpu"),
                              offload_type="block_level", num_blocks_per_group=1, use_stream=False)
    grouped = run()
    torch.testing.assert_close(reference, grouped, rtol=0, atol=0)
    print("PASS: tiny official CUDA/BF16 FIBO pipeline; unused vocabulary logits omitted, cold/warm prompt cache and model/group CPU offload are bit-identical")

if __name__ == '__main__':
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cuda-smoke',action='store_true',help='Explicitly authorize this standalone GPU smoke')
    args=parser.parse_args()
    if not args.cuda_smoke:parser.error('Explicit --cuda-smoke is required; this script runs GPU work.')
    main()
