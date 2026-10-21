"""One cached official pipeline; exact CPU offload and bounded prompt caching."""
from collections import OrderedDict
from functools import wraps
from threading import RLock
import gc
import inspect

from .catalog import VARIANTS, file_identity

LOCK = RLock()
_pipe = None
_key = None

def install_hidden_state_encoder(encoder):
    """FIBO consumes hidden states only, so do not allocate unused vocabulary logits.

Install before Accelerate/group-offload hooks. The exact same SmolLM3 backbone
runs; only the unused language prediction head and KV-cache output are omitted.
Other call modes preserve the original language-model forward.
"""
    original = encoder.forward
    if not hasattr(encoder, "model"):
        raise RuntimeError("SmolLM3 backbone interface changed; hidden-state optimization unavailable")
    @wraps(original)
    def hidden_forward(*args, **kwargs):
        if kwargs.get("output_hidden_states") is True and kwargs.get("labels") is None and "logits_to_keep" not in kwargs:
            kwargs = dict(kwargs)
            kwargs.pop("labels", None)
            kwargs["use_cache"] = False
            kwargs["return_dict"] = True
            return encoder.model(*args, **kwargs)
        return original(*args, **kwargs)
    encoder.forward = hidden_forward

def _map_tensors(value, fn):
    if isinstance(value, tuple):
        return tuple(_map_tensors(x, fn) for x in value)
    if isinstance(value, list):
        return [_map_tensors(x, fn) for x in value]
    return fn(value)

def install_prompt_cache(pipe, max_bytes=256 * 1024**2):
    """Cache official embeddings on CPU; weights and all hidden layers are unchanged."""
    original = pipe.get_prompt_embeds
    signature = inspect.signature(original)
    cache = OrderedDict()
    used = 0
    @wraps(original)
    def cached(*args, **kwargs):
        nonlocal used
        call = signature.bind(*args, **kwargs)
        call.apply_defaults()
        values = call.arguments
        prompt = values["prompt"]
        key = (tuple(prompt) if isinstance(prompt, list) else prompt,
               values.get("num_images_per_prompt"), values.get("max_sequence_length"),
               str(values.get("dtype") or pipe.text_encoder.dtype))
        device = values.get("device") or pipe._execution_device
        if key in cache:
            saved, size = cache.pop(key)
            cache[key] = (saved, size)
            return _map_tensors(saved, lambda x: x.to(device=device, copy=True))
        result = original(*args, **kwargs)
        sizes = []
        _map_tensors(result, lambda x: sizes.append(x.numel() * x.element_size()))
        size = sum(sizes)
        if size <= max_bytes:
            while cache and (len(cache) >= 2 or used + size > max_bytes):
                _, (_, old_size) = cache.popitem(last=False)
                used -= old_size
            cache[key] = (_map_tensors(result, lambda x: x.detach().to("cpu", copy=True)), size)
            used += size
        return result
    pipe.get_prompt_embeds = cached
    pipe._pi_prompt_cache = cache

def release():
    global _pipe, _key
    with LOCK:
        old, _pipe, _key = _pipe, None, None
        if old is not None:
            # Removing offload hooks first prevents module-held CUDA references.
            try:
                old.remove_all_hooks()
            finally:
                getattr(old, "_pi_prompt_cache", {}).clear()
                del old
                gc.collect()
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

def choose_memory(free_bytes, transformer_bytes, mode, total_bytes=None, pixels=1024**2):
    if mode not in ("Auto", "Low VRAM"):
        raise ValueError(f"Unknown FIBO memory mode: {mode}")
    # On large GPUs the resident path avoids repeated full-model transfers.
    # Reserve more headroom above 1MP. An OOM never changes resolution or precision.
    headroom = 6 * 1024**3 * max(1, pixels / 1024**2)
    if mode == "Auto" and total_bytes is not None and total_bytes + headroom < free_bytes:
        return "resident"
    return "model" if mode == "Auto" and transformer_bytes + headroom < free_bytes else "group"

def acquire(folder, variant, memory="Auto", pixels=1024**2):
    global _pipe, _key
    import torch
    from diffusers import BriaFiboPipeline, BriaFiboEditPipeline
    key = (str(folder), variant, memory, file_identity(folder))
    if key == _key and _pipe is not None and (
        _pipe._pi_memory_route != "resident" or pixels <= _pipe._pi_max_pixels
    ):
        return _pipe
    release()
    if not torch.cuda.is_available():
        raise RuntimeError("This FIBO build requires a CUDA GPU")
    cls = BriaFiboEditPipeline if VARIANTS[variant].edit else BriaFiboPipeline
    required = {"callback_on_step_end", "max_sequence_length", "generator", "width", "height"}
    if not required.issubset(inspect.signature(cls.__call__).parameters):
        raise RuntimeError("Installed Diffusers FIBO interface changed. Integration stopped; no packages were altered.")
    pipe = cls.from_pretrained(str(folder), torch_dtype=torch.bfloat16,
                               local_files_only=True, use_safetensors=True,
                               low_cpu_mem_usage=True)
    pipe.set_progress_bar_config(disable=True)
    # Populate owned state before installing offload hooks, so failures can be cleaned up.
    pipe._pi_prompt_cache = {}
    _pipe = pipe
    install_hidden_state_encoder(pipe.text_encoder)
    transformer_bytes = sum(p.numel() * p.element_size() for p in pipe.transformer.parameters())
    total_bytes = sum(p.numel() * p.element_size() for component in pipe.components.values()
                      if isinstance(component, torch.nn.Module) for p in component.parameters())
    route = choose_memory(torch.cuda.mem_get_info()[0], transformer_bytes, memory, total_bytes, pixels)
    if route == "resident":
        pipe.to("cuda")
    elif route == "model":
        pipe.enable_model_cpu_offload(device="cuda")
    else:
        # Synchronous group offload avoids stream/pinned-RAM amplification on Windows.
        # It changes residency only, never weight precision or the sample trajectory.
        pipe.enable_group_offload(onload_device=torch.device("cuda"),
                                  offload_device=torch.device("cpu"),
                                  offload_type="block_level", num_blocks_per_group=1,
                                  use_stream=False, non_blocking=False)
    install_prompt_cache(pipe)
    pipe._pi_memory_route = route
    pipe._pi_max_pixels = pixels
    _pipe, _key = pipe, key
    print(f"[Invisible-FIBO] Loaded official {VARIANTS[variant].label}; BF16, memory route {route}; approximation OFF")
    return pipe
