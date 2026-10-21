import json
import struct
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from forge_fibo2 import assets


def write_safe(path, tensors, metadata=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    header, data, offset = ({"__metadata__": metadata} if metadata else {}), bytearray(), 0
    for name, payload in tensors.items():
        header[name] = {"dtype": "U8", "shape": [len(payload)], "data_offsets": [offset, offset + len(payload)]}
        data.extend(payload)
        offset += len(payload)
    raw = json.dumps(header, separators=(",", ":")).encode()
    raw += b" " * ((8 - len(raw) % 8) % 8)
    path.write_bytes(struct.pack("<Q", len(raw)) + raw + data)


def component(root, role, tensors):
    folder = root / role
    names = list(tensors)
    midpoint = max(1, len(names) // 2)
    groups = (names[:midpoint], names[midpoint:]) if len(names) > 1 else (names,)
    weight_map = {}
    for index, group in enumerate(groups, 1):
        shard = f"model-{index:05d}-of-{len(groups):05d}.safetensors"
        write_safe(folder / shard, {name: tensors[name] for name in group})
        weight_map.update({name: shard for name in group})
    (folder / "model.safetensors.index.json").write_text(json.dumps({"weight_map": weight_map}), encoding="utf-8")


def snapshot(root):
    root.mkdir(parents=True)
    (root / "model_index.json").write_text(json.dumps({
        "_class_name": "BriaFibo2Pipeline",
        "processor": ["transformers", "Qwen3VLProcessor"],
        "scheduler": ["diffusers", "FlowMatchEulerDiscreteScheduler"],
        "text_encoder": ["transformers", "Qwen3VLForConditionalGeneration"],
        "transformer": ["diffusers", "BriaFibo2Transformer2DModel"],
        "vae": ["diffusers", "AutoencoderKLFlux2"],
    }), encoding="utf-8")
    configs = {
        "transformer": {"_class_name": "BriaFibo2Transformer2DModel", "in_channels": 32, "dim": 3584, "cap_feat_dim": 7680, "n_layers": 44},
        "text_encoder": {"model_type": "qwen3_vl", "architectures": ["Qwen3VLForConditionalGeneration"], "text_config": {"hidden_size": 2560}, "vision_config": {"out_hidden_size": 2560}},
        "vae": {"_class_name": "AutoencoderKLFlux2"},
    }
    for role, config in configs.items():
        (root / role).mkdir()
        (root / role / "config.json").write_text(json.dumps(config), encoding="utf-8")
        component(root, role, {f"{role}.a": b"abc", f"{role}.b": b"12345"})
    (root / "scheduler").mkdir()
    (root / "scheduler" / "scheduler_config.json").write_text(json.dumps({"_class_name": "FlowMatchEulerDiscreteScheduler"}), encoding="utf-8")
    (root / "processor").mkdir()
    (root / "processor" / "processor_config.json").write_text(json.dumps({"processor_class": "Qwen3VLProcessor"}), encoding="utf-8")
    (root / "fonts").mkdir()
    (root / "fonts" / "font.txt").write_text("kept", encoding="utf-8")
    return root


def test_relocate_merges_real_safetensors_and_writes_manifest(tmp_path):
    source = snapshot(tmp_path / "snapshot")
    manifest_path = tmp_path / "extension" / "assets.json"
    manifest = assets.relocate(source, tmp_path / "forge", manifest_path)
    found = assets.discover(tmp_path / "forge")
    assert Path(found["checkpoint"]).name == assets.CHECKPOINT_NAME
    assert Path(found["text_encoder_bf16"]).name == assets.TEXT_ENCODER_BF16_NAME
    assert Path(found["vae"]).name == assets.VAE_NAME
    validated = assets.validate_model(found)
    assert validated["text_encoder_preferred"] == str(found["text_encoder_bf16"])
    assert manifest["revision"] == assets.REVISION
    assert json.loads(manifest_path.read_text())["components"]["checkpoint"]["sha256"]
    assert not (source / "transformer").exists()
    assert (source / "configs" / "transformer" / "config.json").is_file()
    assert (source / "configs" / "text_encoder" / "config.json").is_file()
    assert (source / "configs" / "vae" / "config.json").is_file()
    assert manifest["runtime"]["component_configs"]["transformer"]["files"]["config.json"]["sha256"]
    assert (source / "model_index.json").is_file() and (source / "fonts" / "font.txt").is_file()


def test_refuses_different_existing_target_before_overwrite(tmp_path):
    source = snapshot(tmp_path / "snapshot")
    target = Path(assets.discover(tmp_path / "forge")["checkpoint"])
    write_safe(target, {"other": b"different"})
    before = target.read_bytes()
    with pytest.raises(assets.AssetError, match="Refusing to overwrite"):
        assets.relocate(source, tmp_path / "forge", tmp_path / "manifest.json")
    assert target.read_bytes() == before
    assert (source / "transformer").is_dir()


def test_discover_prefers_only_a_real_int8_file(tmp_path):
    found = assets.discover(tmp_path)
    assert found["text_encoder_preferred"] == found["text_encoder_bf16"]
    write_safe(Path(found["text_encoder_int8_convrot"]), {"quantized.weight": b"x"})
    found = assets.discover(tmp_path)
    assert found["text_encoder_preferred"] == found["text_encoder_bf16"]
    write_safe(Path(found["text_encoder_int8_convrot"]), {"layer.comfy_quant": b"x"},
               {"pi_fibo2_format": "int8_convrot", "pi_fibo2_component": "te"})
    found = assets.discover(tmp_path)
    assert found["text_encoder_preferred"] == found["text_encoder_int8_convrot"]


def test_architecture_mismatch_fails_without_moving(tmp_path):
    source = snapshot(tmp_path / "snapshot")
    config = source / "transformer" / "config.json"
    value = json.loads(config.read_text())
    value["dim"] = 1
    config.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(assets.AssetError, match="architecture signature"):
        assets.relocate(source, tmp_path / "forge", tmp_path / "manifest.json")
    assert not Path(assets.discover(tmp_path / "forge")["checkpoint"]).exists()
