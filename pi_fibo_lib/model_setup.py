"""Optional FIBO setup UI; downloads require an explicit button click."""
from pathlib import Path
import hashlib, json, os, shutil, threading
from pi_fibo_lib.catalog import ROOT, VARIANTS, read_json, validate_folder

_LOCK = threading.Lock()
LABEL_TO_KEY = {v.label: k for k, v in VARIANTS.items()}
REVISIONS = {"fibo15": "b91cef5364d81e94a3590fbedd297a5900c6a341"}
UNSAFE = (".bin", ".ckpt", ".pt", ".pth", ".gguf")

def default_destination(key, forge_root):
    return Path(forge_root) / "models" / "pi-fibo-15" / VARIANTS[key].repo.rsplit("/", 1)[-1]

def manual_instructions(label, destination):
    item = VARIANTS[LABEL_TO_KEY[label]]
    return ("### Manual setup (recommended)\n"
            f"1. Open `https://huggingface.co/{item.repo}`, sign in, and accept its terms.\n"
            "2. Download the complete official Diffusers snapshot.\n"
            f"3. Put the complete folder at `{destination}`.\n"
            "4. Refresh Forge's checkpoint list. Use only the normal Hugging Face login.")

def _safe_destination(destination):
    raw = Path(destination).expanduser()
    if not raw.is_absolute():
        raise RuntimeError("Choose an absolute destination folder")
    probe = raw
    while probe != probe.parent:
        if probe.exists() and probe.is_symlink():
            raise RuntimeError("Destination must not use symbolic links")
        probe = probe.parent
    target, extension_root = raw.resolve(), ROOT.resolve()
    if target == extension_root or extension_root in target.parents or target in extension_root.parents:
        raise RuntimeError("Destination must be outside the extension folder and its parents")
    target.mkdir(parents=True, exist_ok=True)
    return target

def _safe_files(info):
    files = list(info.siblings)
    names = [f.rfilename for f in files]
    if not any(n.endswith(".safetensors") for n in names):
        raise RuntimeError("Official snapshot has no safetensors weights")
    if any(n.lower().endswith(UNSAFE) for n in names):
        raise RuntimeError("Official snapshot contains unsupported weight formats")
    if any(Path(n).is_absolute() or ".." in Path(n).parts for n in names):
        raise RuntimeError("Official snapshot contains an unsafe path")
    return files

def _register_location(key, target):
    config = ROOT / "model_locations.json"
    data = read_json(config) if config.is_file() else {}
    if not isinstance(data, dict):
        raise RuntimeError("model_locations.json must contain an object")
    data[key] = str(target)
    temporary = config.with_name(config.name + f".{os.getpid()}.tmp")
    try:
        temporary.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
        os.replace(temporary, config)
    finally:
        temporary.unlink(missing_ok=True)

def download_snapshot(key, destination):
    from huggingface_hub import HfApi, get_token, snapshot_download
    if key not in VARIANTS:
        raise ValueError("Unknown FIBO model variant")
    target, item = _safe_destination(destination), VARIANTS[key]
    token = get_token()
    info = HfApi().model_info(item.repo, revision=REVISIONS.get(key, "main"),
                              token=token, files_metadata=True)
    files = _safe_files(info)
    revision = REVISIONS.get(key) or info.sha
    remaining = sum(max(0, (f.size or 0) - ((target / f.rfilename).stat().st_size
                    if (target / f.rfilename).is_file() else 0)) for f in files)
    if shutil.disk_usage(target).free < remaining + 2 * 1024**3:
        raise RuntimeError("Not enough free space")
    snapshot_download(item.repo, revision=revision, local_dir=target, token=token,
                      allow_patterns=[f.rfilename for f in files], max_workers=3)
    for file in files:
        path = (target / file.rfilename).resolve()
        if target not in path.parents:
            raise RuntimeError("Downloaded file escaped the destination")
        if not path.is_file() or (file.size is not None and path.stat().st_size != file.size):
            raise RuntimeError("Downloaded snapshot is incomplete")
        expected = getattr(file.lfs, "sha256", None) if getattr(file, "lfs", None) else None
        if expected:
            with path.open("rb") as stream:
                actual = hashlib.file_digest(stream, "sha256").hexdigest()
            if actual != expected:
                raise RuntimeError("Downloaded file failed SHA256 verification")
    validate_folder(target, key)
    _register_location(key, target)
    return target

def _is_auth_error(error):
    status = getattr(getattr(error, "response", None), "status_code", None)
    return error.__class__.__name__ in {"GatedRepoError", "RepositoryNotFoundError"} or status in {401, 403}

def _download_from_ui(label, destination):
    if not _LOCK.acquire(blocking=False):
        return "A model download is already running."
    try:
        path = download_snapshot(LABEL_TO_KEY[label], destination)
        return f"Download verified and registered: `{path}`. Refresh the normal checkpoint list."
    except Exception as error:
        if _is_auth_error(error):
            return "Access required. Sign in with the normal Hugging Face CLI and accept the model terms, then try again."
        return "Setup stopped safely. Check the destination, disk space, and official model access."
    finally:
        _LOCK.release()

def _use_existing_from_ui(label, destination):
    try:
        key = LABEL_TO_KEY[label]
        target = _safe_destination(destination)
        validate_folder(target, key)
        _register_location(key, target)
        return f"Existing folder validated and registered: `{target}`. Refresh the normal checkpoint list."
    except Exception:
        return "Folder registration stopped safely. Select a complete official model folder."

def build_ui(is_img2img=False):
    import gradio as gr
    try:
        from modules import paths_internal
        forge_root = Path(paths_internal.script_path)
    except Exception:
        forge_root = ROOT.parents[1]
    first_key = "edit_turbo" if is_img2img else "fibo15"
    first = VARIANTS[first_key].label
    destination = str(default_destination(first_key, forge_root))
    with gr.Accordion("FIBO model setup (optional)", open=False):
        gr.Markdown("Models are never downloaded during startup. Manual setup is the default.")
        variant = gr.Dropdown(list(LABEL_TO_KEY), value=first, label="Official supported model")
        folder = gr.Textbox(value=destination, label="Complete snapshot folder")
        instructions = gr.Markdown(manual_instructions(first, destination))
        use_existing = gr.Button("Use existing complete folder", variant="secondary")
        download = gr.Button("Download official full snapshot", variant="secondary")
        status = gr.Markdown()
        def select_variant(label):
            path = str(default_destination(LABEL_TO_KEY[label], forge_root))
            return path, manual_instructions(label, path)
        variant.change(select_variant, [variant], [folder, instructions], show_progress=False)
        folder.change(manual_instructions, [variant, folder], [instructions], show_progress=False)
        use_existing.click(_use_existing_from_ui, [variant, folder], [status])
        download.click(_download_from_ui, [variant, folder], [status])
    return []
