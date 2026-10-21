"""Lossless structural adaptation of ComfyUI 87ed5a0 FIBO repacks.

Only output rows are split. Padded FFN input columns are retained: cropping
them would invalidate the Hadamard rotation of an INT8 w2 matrix.
"""
import re
import json
import torch


class LayoutError(ValueError):
    pass


def is_repacked(keys):
    return 'layers.0.attention.qkv.weight' in keys


def _rename(key):
    key = re.sub(r'(\.attention)\.q_norm\.', r'\1.norm_q.', key)
    key = re.sub(r'(\.attention)\.k_norm\.', r'\1.norm_k.', key)
    key = key.replace('.attention.out.', '.attention.to_out.0.')
    key = re.sub(r'((?:cross_attention|self_attention)\.to_out)\.(?!0\.)', r'\1.0.', key)
    return key


def unpack_transformer(state):
    """Return ordinary Diffusers keys, preserving quantization tensors exactly."""
    if not is_repacked(state):
        return dict(state)
    result = {}
    consumed = set()
    for key, weight in state.items():
        if not key.endswith('.weight'):
            continue
        base = key[:-7]
        destinations = None
        if base.endswith('.attention.qkv'):
            stem = base[:-3]
            destinations = [stem + suffix for suffix in ('to_q', 'to_k', 'to_v')]
        elif base.endswith('.to_kv'):
            stem = base[:-5]
            destinations = [stem + suffix for suffix in ('to_k', 'to_v')]
        elif base.endswith('.w13'):
            stem = base[:-3]
            destinations = [stem + suffix for suffix in ('w1', 'w3')]
        if destinations is None:
            continue
        count = len(destinations)
        if weight.ndim != 2 or weight.shape[0] % count:
            raise LayoutError(f'Invalid fused output shape: {key}')
        rows = weight.shape[0]
        entries = [(k, v) for k, v in state.items() if k.startswith(base + '.')]
        for source, tensor in entries:
            suffix = source[len(base):]
            if suffix == '.comfy_quant' or tensor.numel() == 1:
                parts = [tensor] * count
            elif tensor.ndim and tensor.shape[0] == rows:
                parts = tensor.chunk(count, dim=0)
            else:
                raise LayoutError(f'Cannot losslessly split fused auxiliary tensor: {source}')
            for destination, part in zip(destinations, parts):
                target = _rename(destination + suffix)
                if target in result:
                    raise LayoutError(f'Duplicate unpacked tensor: {target}')
                result[target] = part.contiguous()
            consumed.add(source)
    for key, tensor in state.items():
        if key not in consumed:
            if key.endswith('.comfy_attention.config'):
                # This is an attention backend hint, not a learned tensor.
                # The official reference processor owns attention execution.
                if not tensor.is_meta:
                    config = json.loads(bytes(tensor.tolist()).decode())
                    if not isinstance(config, dict):
                        raise LayoutError('Invalid attention backend configuration')
                continue
            target = _rename(key)
            if target in result:
                raise LayoutError(f'Duplicate unpacked tensor: {target}')
            result[target] = tensor
    return result


def preserve_ffn_widths(model, state):
    """Construct the repack's padded FFNs; never trim rotated input columns."""
    for name, module in list(model.named_modules()):
        if not all(hasattr(module, attr) for attr in ('w1', 'w2', 'w3')):
            continue
        keys = [name + '.' + attr + '.weight' for attr in ('w1', 'w2', 'w3')]
        if not all(key in state for key in keys):
            continue  # strict key check reports missing weights later
        one, two, three = [state[key].shape for key in keys]
        dim = module.w1.in_features
        width = one[0]
        if one != three or one[1] != dim or tuple(two) != (dim, width) or width < module.w1.out_features:
            raise LayoutError(f'Inconsistent FFN padding: {name}')
        if width != module.w1.out_features:
            # Respect the surrounding constructor's placement policy. Passing
            # device='meta' explicitly makes Forge's quantized load hook move
            # real packed weights back to meta rather than materialize them.
            module.w1 = torch.nn.Linear(dim, width, bias=False, dtype=torch.bfloat16)
            module.w2 = torch.nn.Linear(width, dim, bias=False, dtype=torch.bfloat16)
            module.w3 = torch.nn.Linear(dim, width, bias=False, dtype=torch.bfloat16)
    return model


def unpack_vae(state, config):
    """Use Diffusers' maintained LDM VAE mapping, retaining FLUX2 BN state."""
    if not any(key.startswith('encoder.down.') for key in state):
        return dict(state)
    from diffusers.loaders.single_file_utils import convert_ldm_vae_checkpoint
    prepared = dict(state)
    for old, new in (('encoder.quant_conv.', 'quant_conv.'), ('decoder.post_quant_conv.', 'post_quant_conv.')):
        for key in list(prepared):
            if key.startswith(old):
                prepared[new + key[len(old):]] = prepared.pop(key)
    converted = convert_ldm_vae_checkpoint(prepared, config)
    converted.update({key: value for key, value in state.items() if key.startswith('bn.')})
    if len(converted) != len(state):
        raise LayoutError('VAE conversion did not preserve every source tensor')
    return converted
