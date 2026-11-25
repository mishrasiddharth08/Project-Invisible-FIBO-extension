"""Explicit pinned BF16 setup. No dependency installation or upgrades."""
import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path
import shutil


def _ready(manifest):
    """Validate local identity and required config files without constructing models."""
    try:
        from forge_fibo2.runtime import _validate_metadata, _runtime_dir
        from forge_fibo2.assets import REPOSITORY, REVISION
        from forge_fibo2.provenance import PINS
        data = json.loads(manifest.read_text(encoding="utf-8"))
        pinned = (data.get("repository"), data.get("revision")) in (
            (REPOSITORY, REVISION), (PINS["repository"], PINS["revision"]))
        if not pinned or data.get("schema") != 1 or data.get("family") != "fibo2":
            return False
        for name, role in (("checkpoint", "transformer"), ("text_encoder_bf16", "text_encoder"), ("vae", "vae")):
            path = Path(data["components"][name]["path"])
            _validate_metadata(path, role)
        for role, filename in (("transformer", "config.json"), ("text_encoder", "config.json"),
                               ("vae", "config.json"), ("processor", "processor_config.json"),
                               ("scheduler", "scheduler_config.json")):
            config = _runtime_dir(data, role) / filename
            if not isinstance(json.loads(config.read_text(encoding="utf-8")), dict):
                return False
        fonts = _runtime_dir(data, "fonts")
        return all((fonts / name).is_file() for name in ("Aileron-Regular.otf", "DejaVuSans.ttf"))
    except Exception:
        return False


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host-root", type=Path)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)
    root = Path(__file__).resolve().parent
    host = args.host_root.resolve() if args.host_root else root.parent.parent
    if not (host / "backend").is_dir() or not (host / "modules_forge").is_dir():
        raise RuntimeError("Cannot locate Forge. Pass --host-root with its installation folder.")
    manifest = root / "model-manifest.json"
    if manifest.exists() and _ready(manifest):
        print("Existing FIBO setup is ready. No downloads or changes.")
        return 0
    if manifest.exists():
        print("Existing manifest is incomplete or refers to unavailable local assets.", flush=True)
    from forge_fibo2.assets import REPOSITORY, REVISION, relocate
    from huggingface_hub import HfApi, get_token, snapshot_download
    token = get_token()
    info = HfApi().model_info(REPOSITORY, revision=REVISION, token=token, files_metadata=True)
    total = sum(item.size or 0 for item in info.siblings)
    if shutil.disk_usage(root).free < total + 2 * 1024**3:
        raise RuntimeError("Not enough space for the snapshot plus 2 GiB.")
    required = total * (2 if root.drive.lower() == host.drive.lower() else 1) + 2 * 1024**3
    if shutil.disk_usage(host).free < required:
        raise RuntimeError("Not enough space for the snapshot and merged model files plus 2 GiB.")
    print(f"Pinned source: {REPOSITORY}@{REVISION}", flush=True)
    if args.check:
        print("Access and space checks passed for pinned BF16 setup. Current setup is not ready. No files downloaded or changed.")
        return 0
    # Unique owned workspace: relocate removes its role directories after merging.
    models = root / "models"
    models.mkdir(parents=True, exist_ok=True)
    workspace = Path(tempfile.mkdtemp(prefix=".fibo2-setup-", dir=models))
    snapshot = workspace / "snapshot"
    staged_manifest = workspace / "model-manifest.json"
    snapshot_download(REPOSITORY, revision=REVISION, local_dir=snapshot, token=token, max_workers=3)
    for item in info.siblings:
        path = snapshot / item.rfilename
        if not path.is_file() or path.stat().st_size != item.size:
            raise RuntimeError(f"Incomplete download: {item.rfilename}")
        expected = getattr(item.lfs, "sha256", None) if item.lfs else None
        if expected:
            with path.open("rb") as stream:
                actual = hashlib.file_digest(stream, "sha256").hexdigest()
            if actual != expected:
                raise RuntimeError(f"SHA256 mismatch: {item.rfilename}")
    relocate(snapshot, host, staged_manifest)
    if not _ready(staged_manifest):
        raise RuntimeError("Relocated setup failed validation. Existing manifest retained.")
    if manifest.exists():
        backup = manifest.with_name("model-manifest.json.backup-" + workspace.name.removeprefix(".fibo2-setup-"))
        shutil.copy2(manifest, backup)
    os.replace(staged_manifest, manifest)
    print("BF16 setup complete. Restart Forge. INT8 conversion is a separate explicit step.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
