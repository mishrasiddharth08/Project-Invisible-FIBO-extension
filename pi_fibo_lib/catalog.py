"""Model identity, local discovery and completeness. Never downloads weights."""
from dataclasses import dataclass
from pathlib import Path
import json
import os

ROOT = Path(__file__).resolve().parent.parent

@dataclass(frozen=True)
class Variant:
    label: str
    repo: str
    steps: int
    guidance: float
    edit: bool = False

VARIANTS = {
    "fibo15": Variant("FIBO 1.5", "briaai/Fibo-1.5", 6, 1.0),
    "base": Variant("FIBO original base", "briaai/FIBO", 50, 5.0),
    "lite": Variant("FIBO Lite", "briaai/Fibo-lite", 8, 1.0),
    "edit": Variant("FIBO Edit 1.5 Base", "briaai/Fibo-Edit-1.5-base", 30, 5.0, True),
    "edit_turbo": Variant("FIBO Edit 1.5 Turbo", "briaai/Fibo-Edit-1.5-turbo", 4, 1.0, True),
}

def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))

def selected_variant(filename):
    """Only our actual manifest files count; a FIBO-like filename never does."""
    path = Path(str(filename)).resolve()
    if path.parent != (ROOT / "checkpoints").resolve() or path.stem not in VARIANTS:
        return None
    try:
        meta = read_json(path)
        return path.stem if meta == {"project_invisible_fibo": 2, "variant": path.stem} else None
    except (OSError, ValueError):
        return None

def _safe_child(root, name):
    p = (root / name).resolve()
    if not p.is_relative_to(root.resolve()):
        raise ValueError(f"Model index path leaves its component folder: {name}")
    return p

def validate_folder(folder, variant):
    """Require complete safetensors shards and the actual architecture configs."""
    folder = Path(folder).resolve()
    manifest = read_json(folder / "model_index.json")
    expected = "BriaFiboEditPipeline" if VARIANTS[variant].edit else "BriaFiboPipeline"
    if manifest.get("_class_name") != expected:
        raise ValueError(f"Expected {expected}; found {manifest.get('_class_name')}")
    for role in ("transformer", "text_encoder", "vae"):
        component = folder / role
        config = read_json(component / "config.json")
        if role == "transformer" and config.get("_class_name") != "BriaFiboTransformer2DModel":
            raise ValueError("The selected transformer is not FIBO")
        if role == "text_encoder" and config.get("model_type") != "smollm3":
            raise ValueError("FIBO requires its SmolLM3 text encoder")
        if role == "vae" and (config.get("_class_name") != "AutoencoderKLWan" or config.get("z_dim") != 48):
            raise ValueError("FIBO requires its 48-channel Wan VAE and saved latent statistics")
        if role == "vae" and any(len(config.get(k, [])) != 48 for k in ("latents_mean", "latents_std")):
            raise ValueError("The VAE is missing its 48-channel latent statistics")
        indices = list(component.glob("*.safetensors.index.json"))
        if indices:
            names = {name for index in indices for name in read_json(index).get("weight_map", {}).values()}
            weights = [_safe_child(component, name) for name in names]
        else:
            weights = list(component.glob("*.safetensors"))
        if not weights or any(not p.is_file() or p.stat().st_size < 16 for p in weights):
            raise FileNotFoundError(f"Missing or incomplete {role} safetensors in {component}")
    if not (folder / "tokenizer" / "tokenizer.json").is_file():
        raise FileNotFoundError("Missing tokenizer/tokenizer.json")
    scheduler = read_json(folder / "scheduler" / "scheduler_config.json")
    if scheduler.get("_class_name") != "FlowMatchEulerDiscreteScheduler":
        raise ValueError("Unverified FIBO scheduler; use the official folder unchanged")
    return folder

def resolve_folder(variant, forge_root):
    cfg = ROOT / "model_locations.json"
    locations = read_json(cfg) if cfg.is_file() else {}
    if locations.get(variant):
        # An explicit selection is authoritative, including its errors.
        return validate_folder(Path(locations[variant]).expanduser(), variant)
    name = VARIANTS[variant].repo.split("/")[-1]
    roots = [Path(forge_root) / "models" / "pi-fibo-15", Path(forge_root) / "models" / "diffusers"]
    candidates = [root / n for root in roots for n in (name, variant)]
    hub = Path(os.environ.get("HF_HUB_CACHE", Path(os.environ.get("HF_HOME", Path.home() / ".cache/huggingface")) / "hub"))
    snapshots = hub / ("models--" + VARIANTS[variant].repo.replace("/", "--")) / "snapshots"
    if snapshots.is_dir():
        candidates += sorted(snapshots.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True)
    for candidate in candidates:
        if (candidate / "model_index.json").is_file():
            return validate_folder(candidate, variant)
    raise FileNotFoundError(
        f"{VARIANTS[variant].label} weights are not installed. Accept the model terms at "
        f"https://huggingface.co/{VARIANTS[variant].repo}, download its full Diffusers folder, "
        f"and set '{variant}' to that absolute folder in {cfg}. No downloads were started."
    )

def file_identity(folder):
    return tuple((str(p.relative_to(folder)), p.stat().st_size, p.stat().st_mtime_ns)
                 for p in sorted(Path(folder).rglob("*"))
                 if p.is_file() and p.suffix in (".json", ".safetensors", ".model"))
