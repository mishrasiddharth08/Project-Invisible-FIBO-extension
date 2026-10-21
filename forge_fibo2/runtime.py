"""Strict local Fibo 2.0 runtime. It never downloads or patches shared packages."""

from __future__ import annotations

import gc
import json
from pathlib import Path
import sys
from types import MethodType

import torch
from safetensors import safe_open
from safetensors.torch import load_file
from transformers import Qwen3VLConfig
from transformers.modeling_utils import no_init_weights

from .official import official_classes


class RuntimeAssetError(RuntimeError):
    pass

def _hidden_encoder_forward(encoder, *args, **kwargs):
    if (kwargs.get('output_hidden_states') is True and kwargs.get('labels') is None
            and kwargs.get('past_key_values') is None):
        kwargs = dict(kwargs)
        kwargs.pop('labels',None)
        kwargs.pop('logits_to_keep',None)
        kwargs.update(use_cache=False, return_dict=True)
        return encoder.model(*args, **kwargs)
    return encoder._fibo2_original_forward(*args, **kwargs)

def install_hidden_encoder(encoder, used_layers):
    """Omit logits/cache only for pre-final layers used by this pinned pipeline.

Transformers 4.57's outer capture and backbone differ at the final normalized
hidden state. FIBO reads indices 5..35 of a 36-layer backbone, never that final
state. Refuse the optimization if a future pipeline requests it.
"""
    count=len(encoder.model.language_model.layers)
    if any(layer<0 or layer>=count for layer in used_layers):
        return False
    if not hasattr(encoder,'_fibo2_original_forward'):
        encoder._fibo2_original_forward = encoder.forward
        encoder.forward = MethodType(_hidden_encoder_forward,encoder)
    return True

class _BoundedProcessor:
    def __init__(self,processor):
        self.processor=processor
    def __call__(self,*args,**kwargs):
        limit=kwargs.pop('max_length',None)
        kwargs['truncation']=False
        encoded=self.processor(*args,**kwargs)
        length=encoded['input_ids'].shape[-1]
        if limit is not None and length>limit:
            raise RuntimeAssetError(f'Prompt and image need {length} tokens; FIBO supports {limit}. Shorten the prompt. No tokens were silently removed.')
        return encoded

def install_token_budget(pipe):
    original=pipe._get_qwen_prompt_embeds
    def bounded(text_encoder,processor,*args,**kwargs):
        return original(text_encoder,_BoundedProcessor(processor),*args,**kwargs)
    pipe._get_qwen_prompt_embeds=bounded


def _read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as error:
        raise RuntimeAssetError(f"Cannot read Fibo runtime file: {path}") from error


def _component_path(manifest: dict, name: str, override=None) -> Path:
    if override is not None:
        path = Path(override)
    else:
        value = manifest.get("components", {}).get(name)
        path = Path(value.get("path")) if isinstance(value, dict) and value.get("path") else None
    if path is None or not path.is_file():
        raise RuntimeAssetError(f"Missing local Fibo component: {name}")
    return path.resolve()


def _runtime_dir(manifest: dict, name: str) -> Path:
    runtime = manifest.get("runtime", {})
    component_configs = runtime.get("component_configs", {})
    explicit = runtime.get(name) or runtime.get(f"{name}_config") or component_configs.get(name)
    if explicit:
        if isinstance(explicit, dict):
            explicit = explicit.get("directory")
        if not explicit:
            raise RuntimeAssetError(f"Missing local Fibo config directory: {name}")
        path = Path(explicit)
        path = path.parent if path.is_file() else path
    else:
        model_index = runtime.get("model_index")
        root = Path(model_index).parent if model_index else None
        path = root / "configs" / name if root and (root / "configs" / name).is_dir() else root / name if root else None
    if path is None or not path.is_dir():
        raise RuntimeAssetError(f"Missing local Fibo config directory: {name}")
    return path.resolve()


def _infer_host_root(manifest: dict, checkpoint: Path) -> Path:
    explicit = manifest.get("runtime", {}).get("host_root")
    if explicit:
        return Path(explicit).resolve()
    for parent in checkpoint.parents:
        if (parent / "backend").is_dir() and (parent / "modules_forge").is_dir():
            return parent
    raise RuntimeAssetError("Cannot locate Forge root; add runtime.host_root to the manifest")


def _prepare_forge(host_root: Path):
    for path in (host_root, host_root / "modules_forge" / "packages"):
        value = str(path)
        if value not in sys.path:
            sys.path.insert(0, value)
    try:
        from backend import memory_management
        from backend.operations import using_forge_operations
        from backend.state_dict import detect_quantization, load_state_dict
    except Exception:
        from modules_forge.initialization import initialize_forge

        initialize_forge()
        from backend import memory_management
        from backend.operations import using_forge_operations
        from backend.state_dict import detect_quantization, load_state_dict
    return memory_management, using_forge_operations, detect_quantization, load_state_dict


def _metadata(path: Path) -> tuple[dict, list[str]]:
    with safe_open(str(path), framework="pt", device="cpu") as handle:
        return dict(handle.metadata() or {}), list(handle.keys())


def _validate_metadata(path: Path, role: str) -> tuple[dict, list[str]]:
    metadata, keys = _metadata(path)
    pi_role = metadata.get("project_invisible.role")
    quant_role = metadata.get("pi_fibo2_component")
    expected_quant_role = "dit" if role == "transformer" else "te" if role == "text_encoder" else None
    if pi_role:
        if metadata.get("project_invisible.family") != "fibo2" or pi_role != role:
            raise RuntimeAssetError(f"Wrong Fibo metadata for {role}: {path}")
        from .assets import REVISION, REPOSITORY
        if (metadata.get('project_invisible.revision') != REVISION or
                metadata.get('project_invisible.repository') != REPOSITORY or
                metadata.get('project_invisible.variant') != 'edit-turbo'):
            raise RuntimeAssetError(f'Wrong model revision/variant for {role}: {path}')
    elif expected_quant_role and quant_role == expected_quant_role:
        if metadata.get("pi_fibo2_format") != "int8_convrot":
            raise RuntimeAssetError(f"Unsupported quantization metadata: {path}")
    else:
        from .provenance import identify
        try:
            identity = identify(path, role)
        except ValueError as error:
            raise RuntimeAssetError(str(error)) from error
        metadata = dict(metadata, verified_official_variant=identity['variant'],
                        verified_official_precision=identity['precision'])
    if not keys:
        raise RuntimeAssetError(f"Empty Fibo component: {path}")
    return metadata, keys


def _qwen_config(config_dir: Path) -> Qwen3VLConfig:
    config = Qwen3VLConfig.from_pretrained(str(config_dir), local_files_only=True)
    text = config.text_config
    # Official Fibo was saved by Transformers 5.x. Forge's 4.57 reads the
    # same values under the older rope_scaling field.
    if getattr(text, "rope_scaling", None) is None and getattr(text, "rope_parameters", None):
        text.rope_scaling = dict(text.rope_parameters)
    return config


def _primary_source_keys(keys: set[str]) -> set[str]:
    suffixes = (".comfy_quant", ".weight_scale", ".input_scale")
    return {key for key in keys if not key.endswith(suffixes)}


def _strict_key_check(model, keys: list[str], role: str, quantized: bool) -> None:
    expected = set(model.state_dict())
    supplied = _primary_source_keys(set(keys)) if quantized else set(keys)
    missing = expected - supplied
    unexpected = supplied - expected
    if role == "text_encoder":
        tied = {"model.language_model.embed_tokens.weight", "lm_head.weight"}
        if missing <= tied and supplied & tied:
            missing.clear()
    if missing or unexpected:
        sample = lambda values: sorted(values)[:12]
        raise RuntimeAssetError(
            f"Strict {role} weight mismatch; missing={sample(missing)}, unexpected={sample(unexpected)}"
        )
    if quantized:
        descriptors = {key.removesuffix(".comfy_quant") for key in keys if key.endswith(".comfy_quant")}
        if not descriptors:
            raise RuntimeAssetError(f"Quantized {role} has no per-layer descriptors")
        orphaned = {base for base in descriptors if f"{base}.weight" not in expected}
        if orphaned:
            raise RuntimeAssetError(f"Quantized {role} has unknown layers: {sorted(orphaned)[:12]}")


def _load_model(path: Path, role: str, config_dir: Path, classes: dict, forge):
    memory_management, using_forge_operations, detect_quantization, forge_load_state_dict = forge
    _, keys = _validate_metadata(path, role)
    state = load_file(str(path), device="cpu")
    from .layouts import unpack_transformer, preserve_ffn_widths, unpack_vae
    if role == 'transformer':
        state = unpack_transformer(state)
        keys = list(state)
    descriptors = [k for k in state if k.endswith('.comfy_quant')]
    declared_quant = _metadata(path)[0].get('pi_fibo2_format')
    if declared_quant == 'int8_convrot' and not descriptors:
        raise RuntimeAssetError('INT8_CONVROT metadata has no quantized layer descriptors')
    for key in descriptors:
        descriptor = json.loads(bytes(state[key].tolist()).decode('utf-8'))
        if (descriptor.get('format') != 'int8_tensorwise' or descriptor.get('convrot') is not True
                or descriptor.get('convrot_groupsize') != 256):
            raise RuntimeAssetError(f'Unverified FIBO quantization descriptor: {key}')
        weight = key.removesuffix('.comfy_quant')+'.weight'
        if weight not in state or state[weight].dtype != torch.int8:
            raise RuntimeAssetError(f'INT8_CONVROT layer lacks actual INT8 weights: {key}')
    if role == "text_encoder":
        # The official checkpoint stores one copy of tied token/output weights.
        # Supply the alias before Forge loads, so no missing parameter is ever
        # zero-filled even briefly.
        embed = "model.language_model.embed_tokens.weight"
        head = "lm_head.weight"
        if embed in state and head not in state:
            state[head] = state[embed]
        elif head in state and embed not in state:
            state[embed] = state[head]
        keys = list(state)
    quant_config = detect_quantization(state, is_unet=(role == "transformer"))
    quantized = quant_config is not None
    dtype = torch.bfloat16
    device = torch.device("meta")

    if role == "transformer":
        component_config = classes["transformer"].load_config(str(config_dir))
        construct = lambda: classes["transformer"].from_config(component_config)
    elif role == "text_encoder":
        component_config = _qwen_config(config_dir)
        construct = lambda: classes["text_encoder"](component_config)
    elif role == "vae":
        component_config = classes["vae"].load_config(str(config_dir))
        state = unpack_vae(state, component_config)
        keys = list(state)
        construct = lambda: classes["vae"].from_config(component_config)
    else:
        raise RuntimeAssetError(f"Unknown component role: {role}")

    # Forge quantized modules do not expose ordinary state keys until their
    # load hooks install the packed tensors. Validate against the same official
    # architecture built as a plain meta model first.
    with no_init_weights(), torch.device("meta"):
        reference = construct()
        if role == 'transformer':
            preserve_ffn_widths(reference, state)
    _strict_key_check(reference, keys, role, quantized)
    del reference

    with no_init_weights():
        with using_forge_operations(
            device=device,
            dtype=dtype,
            sd_dtype=dtype,
            manual_cast_enabled=True,
            extra_dtype=dict(quant_config) if quant_config else None,
        ):
            model = construct()
            if role == 'transformer':
                preserve_ffn_widths(model, state)

    forge_load_state_dict(model, state, log_name=f"Fibo2 {role}")
    unloaded=[name for name,parameter in model.named_parameters() if parameter.is_meta]
    if unloaded:
        raise RuntimeAssetError(f'{role} still contains {len(unloaded)} unloaded meta parameters: {unloaded[:20]}')
    if role == "text_encoder":
        model.tie_weights()
    model.eval().requires_grad_(False)
    del state
    return model, quantized


def load(manifest_file, checkpoint=None, text_encoder=None, *, vae=None, device=None):
    """Load local relocated assets and return the official pipeline.

    ``checkpoint`` and ``text_encoder`` may select validated BF16 or converted
    INT8 ConvRot files. No network request or automatic download is performed.
    """
    manifest_path = Path(manifest_file).resolve()
    manifest = _read_json(manifest_path)
    from .assets import REVISION
    from .provenance import PINS
    local_source = manifest.get('revision') == REVISION and manifest.get('variant') == 'edit-turbo'
    official_source = (manifest.get('repository') == PINS['repository'] and
                       manifest.get('revision') == PINS['revision'] and
                       manifest.get('variant') in ('dev', 'edit-turbo'))
    if manifest.get('schema') != 1 or manifest.get('family') != 'fibo2' or not (local_source or official_source):
        raise RuntimeAssetError('Manifest is not a reviewed pinned FIBO 2.0 source')

    checkpoint_path = _component_path(manifest, "checkpoint", checkpoint)
    checkpoint_metadata, _ = _validate_metadata(checkpoint_path, 'transformer')
    variant = checkpoint_metadata.get('verified_official_variant', checkpoint_metadata.get('project_invisible.variant'))
    if variant not in ('dev', 'edit-turbo'):
        raise RuntimeAssetError('Cannot establish Dev versus Edit Turbo checkpoint identity')
    if text_encoder is None:
        # Editing acceptance uses the uploaded workflow's BF16 encoder. A
        # different precision must be selected explicitly, never substituted.
        text_encoder = manifest.get("components", {}).get("text_encoder_bf16", {}).get("path")
    text_encoder_path = _component_path(manifest, "text_encoder_bf16", text_encoder)
    vae_path = _component_path(manifest, "vae", vae)
    host_root = _infer_host_root(manifest, checkpoint_path)
    forge = _prepare_forge(host_root)
    classes = official_classes()

    transformer, transformer_quantized = _load_model(
        checkpoint_path, "transformer", _runtime_dir(manifest, "transformer"), classes, forge
    )
    encoder, encoder_quantized = _load_model(
        text_encoder_path, "text_encoder", _runtime_dir(manifest, "text_encoder"), classes, forge
    )
    vae, _ = _load_model(vae_path, "vae", _runtime_dir(manifest, "vae"), classes, forge)
    processor = classes["processor"].from_pretrained(
        str(_runtime_dir(manifest, "processor")),
        local_files_only=True,
        use_fast=True,
        extra_special_tokens={},
    )
    scheduler = classes["scheduler"].from_pretrained(
        str(_runtime_dir(manifest, "scheduler")), local_files_only=True
    )
    pipe = classes["pipeline"](
        transformer=transformer,
        scheduler=scheduler,
        vae=vae,
        text_encoder=encoder,
        processor=processor,
    )
    pipe._fibo2_variant = variant
    pipe.fibo2_attention_policy = 'Reference Diffusers attention; optional Comfy kernel hints are not applied'
    install_hidden_encoder(encoder,[layer for bundle in pipe.text_encoder_out_layers for layer in bundle])
    install_token_budget(pipe)
    fonts_dir = _runtime_dir(manifest, "fonts")
    pipe.fibo2_fonts = [str(fonts_dir / name) for name in ("Aileron-Regular.otf", "DejaVuSans.ttf")]
    pipe.fibo2_component_names = {'checkpoint': checkpoint_path.name, 'text_encoder': text_encoder_path.name,
                                 'vae': vae_path.name}
    checkpoint_precision = "INT8_CONVROT" if transformer_quantized else "BF16"
    encoder_precision = "INT8_CONVROT" if encoder_quantized else "BF16"
    pipe._fibo2_precision = (
        f"checkpoint={checkpoint_precision}; TE={encoder_precision}; "
        "VAE=BF16 runtime/FP32 disk"
    )

    memory_management = forge[0]
    target_device = torch.device(device) if device is not None else memory_management.get_torch_device()
    try:
        if target_device.type == "cpu":
            pipe.fibo2_memory_policy = "cpu-staged"
        elif (transformer_quantized and encoder_quantized and
                torch.cuda.mem_get_info(target_device)[0] >=
                sum(p.stat().st_size for p in (checkpoint_path,text_encoder_path,vae_path)) + 6*1024**3):
            pipe.to(target_device)
            pipe.fibo2_memory_policy = "resident-quantized"
        else:
            pipe.enable_model_cpu_offload(device=target_device)
            pipe.fibo2_memory_policy = "model-cpu-offload"
    except BaseException:
        release(pipe)
        raise
    return pipe


def release(pipe) -> None:
    if pipe is None:
        return
    try:
        pipe.maybe_free_model_hooks()
    except Exception:
        pass
    try:
        pipe.remove_all_hooks()
    except Exception:
        pass
    for name in ("text_encoder", "transformer", "vae"):
        module = getattr(pipe, name, None)
        if module is not None:
            if name=='text_encoder' and hasattr(module,'_fibo2_original_forward'):
                module.forward=module._fibo2_original_forward
                del module._fibo2_original_forward
            try:
                module.to("cpu")
            except Exception:
                pass
            setattr(pipe, name, None)
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    try:
        from backend import memory_management

        memory_management.soft_empty_cache()
    except Exception:
        pass


__all__ = ["RuntimeAssetError", "load", "release"]
