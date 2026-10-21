"""Streaming FIBO 2.0 checkpoint and Qwen3-VL INT8 ConvRot conversion.

Adapted from the installed Project Invisible LLaDA converter.

Never alters the source. No silently failed layers. Output uses Forge's native
per-linear .comfy_quant descriptors and packed weights/scales.
"""
import argparse
from collections import Counter
import json
import os
from pathlib import Path
import re
import shutil
import struct
import tempfile

import torch
from safetensors import safe_open
from safetensors.torch import save

TARGETS = ("bf16", "fp16", "fp8", "fp8_e5m2", "mxfp8", "nvfp4", "int8", "int8_convrot", "int4_convrot", "int4_convrot_mixed", "w4a8_convrot")
STACKED_EXPERT = re.compile(r"^(.*\.experts)\.(gate_proj|up_proj|down_proj)$")
PROTECTED = ("embed", "norm", "bias", "scale", "adaln", "modulation", "final_layer", "lm_head")


class StreamingWriter:
    """Bounded RAM safetensors writer; publish only a complete validated file."""
    def __init__(self, destination, metadata):
        self.destination = Path(destination)
        if self.destination.exists():
            raise FileExistsError(self.destination)
        self.destination.parent.mkdir(parents=True, exist_ok=True)
        self.file = tempfile.NamedTemporaryFile(prefix=".fibo2-payload-", suffix=".tmp",
            dir=self.destination.parent, delete=False)
        self.header = {"__metadata__": metadata}
        self.position = 0
        self.final_temp = None

    def add(self, key, tensor):
        if key in self.header:
            raise ValueError(f"Duplicate output tensor: {key}")
        blob = save({key: tensor.detach().cpu().contiguous()})
        header_size = struct.unpack("<Q", blob[:8])[0]
        entry = json.loads(blob[8:8 + header_size])[key]
        data = memoryview(blob)[8 + header_size:]
        entry["data_offsets"] = [self.position, self.position + len(data)]
        self.header[key] = entry
        self.file.write(data)
        self.position += len(data)

    def finish(self):
        self.file.flush()
        os.fsync(self.file.fileno())
        self.file.close()
        encoded = json.dumps(self.header, separators=(",", ":")).encode("utf-8")
        encoded += b" " * (-len(encoded) % 8)
        with tempfile.NamedTemporaryFile(prefix=".fibo2-complete-", suffix=".tmp",
                dir=self.destination.parent, delete=False) as complete:
            self.final_temp = Path(complete.name)
            complete.write(struct.pack("<Q", len(encoded)))
            complete.write(encoded)
            with open(self.file.name, "rb") as payload:
                shutil.copyfileobj(payload, complete, length=16 * 1024**2)
            complete.flush()
            os.fsync(complete.fileno())
        with safe_open(str(self.final_temp), framework="pt", device="cpu") as result:
            if set(result.keys()) != set(self.header) - {"__metadata__"}:
                raise ValueError("Output header validation failed")
        # Atomic no-clobber publication; hard link fails if destination exists.
        os.link(self.final_temp, self.destination)
        self.abort()

    def abort(self):
        self.file.close()
        Path(self.file.name).unlink(missing_ok=True)
        if self.final_temp is not None:
            self.final_temp.unlink(missing_ok=True)


def eligible(key, tensor, target):
    if not key.endswith(".weight") or tensor.ndim != 2:
        return False
    if any(word in key.lower() for word in PROTECTED) or key.endswith(".gate.weight") or key.startswith(("model.visual.", "visual.")):
        return False
    alignment = 256 if "convrot" in target else 32 if target == "mxfp8" else 16
    return tensor.shape[1] % alignment == 0 and tensor.shape[0] % 16 == 0


def layer_target(key, target, component):
    if target != 'int4_convrot_mixed':
        return target
    # Explicit mixed precision, not a silently re-labelled pure W4 model.
    # Keep refiners and attention at INT8; W4 applies to main DiT MLPs and
    # TE experts. Embeddings/norms/modulation remain protected as before.
    w4 = (component == 'dit' and key.startswith('layers.') and '.feed_forward.' in key)
    w4 = w4 or (component == 'te' and '.experts.experts.' in key)
    return 'int4_convrot' if w4 else 'int8_convrot'


def quantize(key, weight, target, device="cuda"):
    """Serialize using the installed kitchen layout API, not invented scale math."""
    import comfy_kitchen as ck
    from comfy_kitchen.tensor import (TensorWiseINT8Layout, TensorCoreConvRotW4A4Layout,
        TensorCoreNVFP4Layout, TensorCoreMXFP8Layout, AsymW4A8Int8Layout)
    weight = weight.to(device=device).contiguous()
    base = key.removesuffix(".weight")
    if target in ('fp8', 'fp8_e5m2'):
        fp8_dtype = torch.float8_e5m2 if target == 'fp8_e5m2' else torch.float8_e4m3fn
        scale = (weight.abs().max().float() / torch.finfo(fp8_dtype).max).clamp(min=1e-12)
        tensors = {key: ck.quantize_per_tensor_fp8(weight, scale, fp8_dtype), base + ".weight_scale": scale}
        config = {"format": 'float8_e5m2' if target == 'fp8_e5m2' else 'float8_e4m3fn'}
    else:
        layout, config, options = {
            "int8": (TensorWiseINT8Layout, {"format": "int8_tensorwise"}, {}),
            "int8_convrot": (TensorWiseINT8Layout,
                {"format": "int8_tensorwise", "convrot": True, "convrot_groupsize": 256},
                {"per_channel": True, "convrot": True, "convrot_groupsize": 256}),
            "int4_convrot": (TensorCoreConvRotW4A4Layout,
                {"format": "convrot_w4a4", "convrot_groupsize": 256, "quant_group_size": 64},
                {"convrot_groupsize": 256, "quant_group_size": 64}),
            "w4a8_convrot": (AsymW4A8Int8Layout,
                {"format": "asym_w4a8_int8", "group_size": 16, "convrot_groupsize": 256},
                {"group_size": 16, "convrot_groupsize": 256, "scale_dtype": torch.float8_e4m3fn}),
            "mxfp8": (TensorCoreMXFP8Layout, {"format": "mxfp8"}, {}),
            "nvfp4": (TensorCoreNVFP4Layout, {"format": "nvfp4"}, {}),
        }[target]
        qdata, params = layout.quantize(weight, **options)
        tensors = {base + ".weight" + suffix: value for suffix, value in layout.state_dict_tensors(qdata, params).items()}
    tensors = {name: (value.view(torch.uint8) if value.dtype == getattr(torch, "float8_e8m0fnu", None) else value).cpu()
               for name, value in tensors.items()}
    tensors[base + ".comfy_quant"] = torch.tensor(list(json.dumps(config).encode()), dtype=torch.uint8)
    return tensors


@torch.inference_mode()
def convert(source_path, destination, target, component, device="cuda", log=print):
    if target != "int8_convrot" or component not in ("dit", "te"):
        raise ValueError("Unknown conversion target/component")
    source_path, destination = Path(source_path).resolve(), Path(destination).resolve()
    if source_path == destination or destination.exists():
        raise FileExistsError("Choose a new output file; source and existing outputs are never overwritten")
    counts = Counter()
    writer = None
    try:
        with safe_open(str(source_path), framework="pt", device="cpu") as source:
            if any(k.endswith(".comfy_quant") for k in source.keys()):
                raise ValueError("Use the original BF16/FP16 source, not an already-quantized model")
            metadata = {**(source.metadata() or {}), "format": "pt", "pi_fibo2_component": component,
                "pi_fibo2_format": target, "pi_fibo2_expert_layout": "split-linear-v1" if component == "te" else "none"}
            metadata['project_invisible.quantization'] = target
            writer = StreamingWriter(destination, metadata)
            def write_weight(key, value):
                if value.is_floating_point() and value.dtype not in (torch.bfloat16, torch.float16, torch.float32):
                    raise ValueError(f"Conversion requires unquantized source weights: {key}")
                if target in ("bf16", "fp16"):
                    writer.add(key, value.to(torch.bfloat16 if target == "bf16" else torch.float16) if value.is_floating_point() else value)
                    counts["dense"] += 1
                elif eligible(key, value, target):
                    selected_target = layer_target(key, target, component)
                    for name, tensor in quantize(key, value, selected_target, device).items():
                        writer.add(name, tensor)
                    counts[selected_target + '_linears'] += 1
                    counts[selected_target + '_source_parameters'] += value.numel()
                    counts["quantized_linears"] += 1
                    counts["quantized_source_parameters"] += value.numel()
                else:
                    writer.add(key, value)
                    counts["protected_or_unaligned_tensors"] += 1
                    counts["retained_source_parameters"] += value.numel()
            for index, key in enumerate(source.keys()):
                match = STACKED_EXPERT.match(key) if component == "te" else None
                if match:
                    tensor_slice = source.get_slice(key)
                    shape = tensor_slice.get_shape()
                    if len(shape) != 3:
                        raise ValueError(f"Expected [experts,out,in] for {key}")
                    for expert in range(shape[0]):
                        write_weight(f"{match[1]}.experts.{expert}.{match[2]}.weight", tensor_slice[expert])
                    counts["expert_stacks"] += 1
                    log(f"Converted {key} ({shape[0]} experts)", flush=True)
                else:
                    write_weight(key, source.get_tensor(key))
                if index % 100 == 0:
                    log(f"Conversion: {index + 1}/{len(source.keys())} source tensors", flush=True)
            if False:  # FIBO 2 uses a dense Qwen3-VL encoder, not stacked experts.
                raise ValueError("This does not contain the expected LLaDA stacked TE experts")
            if target not in ("bf16", "fp16") and not counts["quantized_linears"]:
                raise ValueError("No compatible weights were quantized")
            writer.header["__metadata__"]["pi_fibo2_conversion_counts"] = json.dumps(counts)
            writer.finish()
        return dict(counts)
    finally:
        if writer is not None:
            writer.abort()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--format", choices=TARGETS, required=True)
    parser.add_argument("--component", choices=("dit", "te"), required=True)
    args = parser.parse_args()
    print(json.dumps(convert(args.source, args.destination, args.format, args.component), indent=2))


if __name__ == "__main__":
    main()
