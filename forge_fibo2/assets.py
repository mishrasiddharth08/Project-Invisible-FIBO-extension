"""Validated, explicit relocation of the pinned FIBO 2.0 snapshot."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import struct
from pathlib import Path

REPOSITORY = "briaai/Fibo-2.0-edit-turbo"
REVISION = "177f2d4a06b6587144ef5c2b8452a4f1fe2ab4b7"
FAMILY = "fibo2"
FAMILY_DISPLAY = "FIBO-2.0"
VARIANT = "edit-turbo"
CHECKPOINT_NAME = "FIBO-2.0-Edit-Turbo-BF16.safetensors"
TEXT_ENCODER_BF16_NAME = "Qwen3-VL-4B-TextEncoder-BF16--For-FIBO-2.0.safetensors"
TEXT_ENCODER_INT8_NAME = "Qwen3-VL-4B-TextEncoder-int8_convrot--For-FIBO-2.0.safetensors"
VAE_NAME = "LLaDA-Image-Turbo-VAE-FP32--For-FIBO-2.0.safetensors"
_CHUNK = 16 * 1024 * 1024
_DTYPE_BYTES = {
    "BOOL": 1, "U8": 1, "I8": 1, "F8_E4M3": 1, "F8_E5M2": 1,
    "I16": 2, "U16": 2, "F16": 2, "BF16": 2,
    "I32": 4, "U32": 4, "F32": 4,
    "I64": 8, "U64": 8, "F64": 8,
}


class AssetError(RuntimeError):
    pass


def _json(path: Path) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, ValueError) as exc:
        raise AssetError(f"Invalid or missing JSON: {path}") from exc
    if not isinstance(value, dict):
        raise AssetError(f"Expected a JSON object: {path}")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(_CHUNK):
            digest.update(chunk)
    return digest.hexdigest()


def _header(path: Path) -> tuple[dict, int, int]:
    try:
        size = path.stat().st_size
        with path.open("rb") as stream:
            raw = stream.read(8)
            if len(raw) != 8:
                raise AssetError(f"Truncated safetensors file: {path}")
            length = struct.unpack("<Q", raw)[0]
            if length < 2 or length > size - 8:
                raise AssetError(f"Invalid safetensors header length: {path}")
            header = json.loads(stream.read(length).decode("utf-8").rstrip(" \t\r\n\0"))
    except (OSError, UnicodeDecodeError, ValueError, json.JSONDecodeError) as exc:
        raise AssetError(f"Invalid safetensors header: {path}") from exc
    if not isinstance(header, dict):
        raise AssetError(f"Invalid safetensors header object: {path}")
    data_size = size - 8 - length
    tensors = 0
    for name, info in header.items():
        if name == "__metadata__":
            continue
        if not isinstance(info, dict) or info.get("dtype") not in _DTYPE_BYTES:
            raise AssetError(f"Invalid tensor descriptor {name!r}: {path}")
        shape, offsets = info.get("shape"), info.get("data_offsets")
        if not isinstance(shape, list) or not isinstance(offsets, list) or len(offsets) != 2:
            raise AssetError(f"Invalid tensor shape/offsets {name!r}: {path}")
        count = 1
        for dimension in shape:
            if not isinstance(dimension, int) or dimension < 0:
                raise AssetError(f"Invalid tensor shape {name!r}: {path}")
            count *= dimension
        start, end = offsets
        if not all(isinstance(x, int) for x in offsets) or start < 0 or end < start or end > data_size:
            raise AssetError(f"Invalid tensor bounds {name!r}: {path}")
        if end - start != count * _DTYPE_BYTES[info["dtype"]]:
            raise AssetError(f"Tensor byte count mismatch {name!r}: {path}")
        tensors += 1
    if not tensors:
        raise AssetError(f"Safetensors file contains no tensors: {path}")
    return header, 8 + length, data_size


def _component_sources(folder: Path) -> tuple[list[str], dict[str, tuple[Path, dict, int]]]:
    indexes = sorted(folder.glob("*.safetensors.index.json"))
    mapping: dict[str, str] = {}
    if indexes:
        if len(indexes) != 1:
            raise AssetError(f"Expected one shard index in {folder}")
        mapping = _json(indexes[0]).get("weight_map", {})
        if not isinstance(mapping, dict) or not mapping:
            raise AssetError(f"Empty shard index: {indexes[0]}")
        order = list(mapping)
        files = {folder / name for name in mapping.values()}
    else:
        files = set(folder.glob("*.safetensors"))
        if len(files) != 1:
            raise AssetError(f"Expected indexed shards or one safetensors file in {folder}")
        only = next(iter(files))
        header, _, _ = _header(only)
        order = [name for name in header if name != "__metadata__"]
        mapping = {name: only.name for name in order}
    parsed: dict[Path, tuple[dict, int]] = {}
    for path in files:
        if not path.is_file():
            raise AssetError(f"Missing shard: {path}")
        header, data_start, _ = _header(path)
        parsed[path] = (header, data_start)
    sources = {}
    for name in order:
        path = folder / mapping[name]
        header, data_start = parsed[path]
        if name not in header:
            raise AssetError(f"Shard index points to missing tensor {name!r}: {path}")
        sources[name] = (path, header[name], data_start)
    indexed = set(order)
    actual = {name for header, _ in parsed.values() for name in header if name != "__metadata__"}
    if indexed != actual:
        raise AssetError(f"Shard index and tensor headers disagree in {folder}")
    return order, sources


def _merged_plan(folder: Path, metadata: dict[str, str]):
    order, sources = _component_sources(folder)
    header: dict = {"__metadata__": metadata}
    offset = 0
    for name in order:
        _, descriptor, _ = sources[name]
        length = descriptor["data_offsets"][1] - descriptor["data_offsets"][0]
        header[name] = {"dtype": descriptor["dtype"], "shape": descriptor["shape"],
                        "data_offsets": [offset, offset + length]}
        offset += length
    raw = json.dumps(header, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    raw += b" " * ((8 - len(raw) % 8) % 8)
    return raw, order, sources


def _stream_merged(folder: Path, metadata: dict[str, str], output: Path | None) -> str:
    raw, order, sources = _merged_plan(folder, metadata)
    digest = hashlib.sha256()
    prefix = struct.pack("<Q", len(raw)) + raw
    digest.update(prefix)
    destination = output.open("xb") if output is not None else None
    try:
        if destination:
            destination.write(prefix)
        for name in order:
            path, descriptor, data_start = sources[name]
            start, end = descriptor["data_offsets"]
            remaining = end - start
            with path.open("rb") as source:
                source.seek(data_start + start)
                while remaining:
                    chunk = source.read(min(_CHUNK, remaining))
                    if not chunk:
                        raise AssetError(f"Unexpected end of shard: {path}")
                    digest.update(chunk)
                    if destination:
                        destination.write(chunk)
                    remaining -= len(chunk)
        if destination:
            destination.flush()
            os.fsync(destination.fileno())
    finally:
        if destination:
            destination.close()
    return digest.hexdigest()


def _metadata(role: str, quantization: str = "bf16") -> dict[str, str]:
    return {
        "project_invisible.family": FAMILY,
        "project_invisible.family_display": FAMILY_DISPLAY,
        "project_invisible.version": "2.0",
        "project_invisible.variant": VARIANT,
        "project_invisible.role": role,
        "project_invisible.quantization": quantization,
        "project_invisible.repository": REPOSITORY,
        "project_invisible.revision": REVISION,
    }


def _install_component(source: Path, target: Path, metadata: dict[str, str]) -> str:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        expected = _stream_merged(source, metadata, None)
        if _sha256(target) != expected:
            raise AssetError(f"Refusing to overwrite a different file: {target}")
        return expected
    temporary = target.with_name(target.name + ".partial")
    if temporary.exists():
        temporary.unlink()
    try:
        digest = _stream_merged(source, metadata, temporary)
        if _sha256(temporary) != digest:
            raise AssetError(f"Checksum verification failed: {temporary}")
        os.replace(temporary, target)
        return digest
    finally:
        if temporary.exists():
            temporary.unlink()


def _preserve_configs(snapshot: Path, roles: tuple[str, ...]) -> dict:
    plans = []
    for role in roles:
        destination = snapshot / "configs" / role
        for source in sorted((snapshot / role).glob("*.json")):
            target = destination / source.name
            if target.exists() and _sha256(target) != _sha256(source):
                raise AssetError(f"Refusing to overwrite a different config: {target}")
            plans.append((role, source, target))
    result = {role: {"directory": str(snapshot / "configs" / role), "files": {}} for role in roles}
    for role, source, target in plans:
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            temporary = target.with_name(target.name + ".partial")
            shutil.copy2(source, temporary)
            os.replace(temporary, target)
        result[role]["files"][target.name] = {"path": str(target), "sha256": _sha256(target)}
    return result


def _validate_snapshot(snapshot: Path) -> None:
    model = _json(snapshot / "model_index.json")
    expected = {
        "_class_name": "BriaFibo2Pipeline",
        "transformer": ["diffusers", "BriaFibo2Transformer2DModel"],
        "text_encoder": ["transformers", "Qwen3VLForConditionalGeneration"],
        "processor": ["transformers", "Qwen3VLProcessor"],
        "vae": ["diffusers", "AutoencoderKLFlux2"],
        "scheduler": ["diffusers", "FlowMatchEulerDiscreteScheduler"],
    }
    if any(model.get(key) != value for key, value in expected.items()):
        raise AssetError("Snapshot model_index.json is not the pinned FIBO 2.0 architecture")
    transformer = _json(snapshot / "transformer" / "config.json")
    if any(transformer.get(key) != value for key, value in {
        "_class_name": "BriaFibo2Transformer2DModel", "in_channels": 32,
        "dim": 3584, "cap_feat_dim": 7680, "n_layers": 44,
    }.items()):
        raise AssetError("Transformer architecture signature does not match FIBO 2.0")
    encoder = _json(snapshot / "text_encoder" / "config.json")
    if (encoder.get("model_type") != "qwen3_vl" or
            "Qwen3VLForConditionalGeneration" not in encoder.get("architectures", []) or
            encoder.get("text_config", {}).get("hidden_size") != 2560 or
            encoder.get("vision_config", {}).get("out_hidden_size") != 2560):
        raise AssetError("Text encoder is not the required Qwen3-VL-4B architecture")
    vae = _json(snapshot / "vae" / "config.json")
    if vae.get("_class_name") != "AutoencoderKLFlux2":
        raise AssetError("VAE is not the required FLUX.2 VAE")
    scheduler = _json(snapshot / "scheduler" / "scheduler_config.json")
    if scheduler.get("_class_name") != "FlowMatchEulerDiscreteScheduler":
        raise AssetError("Scheduler is not FlowMatch Euler")
    processor = _json(snapshot / "processor" / "processor_config.json")
    if processor.get("processor_class") != "Qwen3VLProcessor":
        raise AssetError("Processor is not Qwen3VLProcessor")
    for role in ("transformer", "text_encoder", "vae"):
        _component_sources(snapshot / role)


def discover(host_root) -> dict:
    host = Path(host_root).resolve()
    checkpoint = host / "models" / "Stable-diffusion" / FAMILY_DISPLAY / CHECKPOINT_NAME
    te_root = host / "models" / "text_encoder" / FAMILY_DISPLAY
    bf16 = te_root / TEXT_ENCODER_BF16_NAME
    int8 = te_root / TEXT_ENCODER_INT8_NAME
    vae = host / "models" / "VAE" / FAMILY_DISPLAY / VAE_NAME
    int8_valid = False
    if int8.is_file():
        try:
            _validate_int8_component(int8, "te")
            int8_valid = True
        except AssetError:
            pass
    return {
        "family": FAMILY, "family_display": FAMILY_DISPLAY, "variant": VARIANT,
        "checkpoint": checkpoint, "text_encoder_bf16": bf16,
        "text_encoder_int8_convrot": int8,
        "text_encoder_preferred": int8 if int8_valid else bf16,
        "vae": vae,
    }


def _validate_int8_component(path: Path, component: str) -> None:
    header, _, _ = _header(path)
    metadata = header.get("__metadata__", {})
    if (metadata.get("pi_fibo2_format") != "int8_convrot" or
            metadata.get("pi_fibo2_component") != component):
        raise AssetError(f"Missing FIBO 2 INT8_CONVROT provenance: {path}")
    if not any(".comfy_quant" in name.lower() for name in header if name != "__metadata__"):
        raise AssetError(f"INT8_CONVROT descriptors are missing: {path}")


def validate_model(component_paths, *, verify_hashes=True) -> dict:
    paths = dict(component_paths)
    required = ("checkpoint", "text_encoder_bf16", "vae")
    for key in required:
        path = Path(paths[key])
        header, _, _ = _header(path)
        metadata = header.get("__metadata__", {})
        if metadata.get("project_invisible.family") != FAMILY or metadata.get("project_invisible.role") != {
                "checkpoint": "transformer", "text_encoder_bf16": "text_encoder", "vae": "vae"}[key]:
            raise AssetError(f"Wrong FIBO 2.0 metadata: {path}")
    int8 = Path(paths["text_encoder_int8_convrot"])
    digest = lambda path: _sha256(path) if verify_hashes else None
    result = {key: {"path": str(Path(paths[key])), "sha256": digest(Path(paths[key]))} for key in required}
    if int8.is_file():
        _validate_int8_component(int8, "te")
        result["text_encoder_int8_convrot"] = {"path": str(int8), "sha256": digest(int8)}
    result["text_encoder_preferred"] = str(int8 if int8.is_file() else Path(paths["text_encoder_bf16"]))
    return result


def relocate(snapshot, host_root, manifest_path) -> dict:
    snapshot, manifest_path = Path(snapshot).resolve(), Path(manifest_path).resolve()
    _validate_snapshot(snapshot)
    paths = discover(host_root)
    installed = {}
    roles = (
        ("checkpoint", "transformer", _metadata("transformer")),
        ("text_encoder_bf16", "text_encoder", _metadata("text_encoder")),
        ("vae", "vae", _metadata("vae", "fp32")),
    )
    # Refuse every conflict before creating any new destination artifact.
    for key, source_role, metadata in roles:
        target = Path(paths[key])
        if target.exists() and _sha256(target) != _stream_merged(snapshot / source_role, metadata, None):
            raise AssetError(f"Refusing to overwrite a different file: {target}")
    configs = _preserve_configs(snapshot, tuple(role for _, role, _ in roles))
    for key, source_role, metadata in roles:
        target = Path(paths[key])
        digest = _install_component(snapshot / source_role, target, metadata)
        installed[key] = {"path": str(target), "directory": str(target.parent),
                          "sha256": digest, "bytes": target.stat().st_size}
    installed["text_encoder_int8_convrot"] = {
        "path": str(paths["text_encoder_int8_convrot"]),
        "available": Path(paths["text_encoder_int8_convrot"]).is_file(),
        "canonical_name": TEXT_ENCODER_INT8_NAME,
    }
    installed["text_encoder_preference"] = ["int8_convrot", "bf16"]
    manifest = {
        "schema": 1, "repository": REPOSITORY, "revision": REVISION,
        "family": FAMILY, "family_display": FAMILY_DISPLAY, "version": "2.0",
        "variant": VARIANT, "components": installed,
        "runtime": {
            "model_index": str(snapshot / "model_index.json"),
            "scheduler": str(snapshot / "scheduler"),
            "processor": str(snapshot / "processor"),
            "fonts": str(snapshot / "fonts"),
            "component_configs": configs,
        },
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = manifest_path.with_name(manifest_path.name + ".partial")
    temporary.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    os.replace(temporary, manifest_path)
    for _, source_role, _ in roles:
        shutil.rmtree(snapshot / source_role)
    return manifest


def _main() -> None:
    parser = argparse.ArgumentParser(description="FIBO 2.0 asset relocation (never downloads models)")
    sub = parser.add_subparsers(dest="command", required=True)
    move = sub.add_parser("relocate")
    move.add_argument("--snapshot", required=True)
    move.add_argument("--host-root", required=True)
    move.add_argument("--manifest", required=True)
    show = sub.add_parser("discover")
    show.add_argument("--host-root", required=True)
    check = sub.add_parser("validate")
    check.add_argument("--host-root", required=True)
    args = parser.parse_args()
    if args.command == "relocate":
        result = relocate(args.snapshot, args.host_root, args.manifest)
    elif args.command == "discover":
        result = {key: str(value) if isinstance(value, Path) else value for key, value in discover(args.host_root).items()}
    else:
        result = validate_model(discover(args.host_root))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    _main()
