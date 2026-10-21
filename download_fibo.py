"""User-requested official snapshot download with size and SHA256 verification.

Uses standard HF login or HF_TOKEN, never stores or prints credentials.
Run with --check to test model access without downloading model weights.
"""
from pathlib import Path
import argparse
import hashlib
import json
import shutil
from huggingface_hub import HfApi, get_token, hf_hub_download, snapshot_download
from huggingface_hub.errors import GatedRepoError

REPO = "briaai/Fibo-1.5"
REVISION = "b91cef5364d81e94a3590fbedd297a5900c6a341"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    args.destination.mkdir(parents=True, exist_ok=True)
    token = get_token()
    print(f"Destination: {args.destination}", flush=True)
    print(f"Existing Hugging Face login: {bool(token)}", flush=True)
    info = HfApi().model_info(REPO, revision=REVISION, token=token, files_metadata=True)
    expected = sum(f.size or 0 for f in info.siblings)
    print(f"Official snapshot: {len(info.siblings)} files, {expected / 10**9:.2f} GB", flush=True)
    try:
        hf_hub_download(REPO, "model_index.json", revision=REVISION,
                        local_dir=args.destination, token=token)
    except GatedRepoError:
        print("ACCESS REQUIRED: sign in to Hugging Face, accept access on https://huggingface.co/briaai/Fibo-1.5, then use hf auth login locally. Do not paste tokens into chat.", flush=True)
        return 3
    if args.check:
        print("ACCESS CONFIRMED", flush=True)
        return 0
    remaining = sum(max(0, (f.size or 0) - ((args.destination / f.rfilename).stat().st_size
                    if (args.destination / f.rfilename).is_file() else 0)) for f in info.siblings)
    if shutil.disk_usage(args.destination).free < remaining + 2 * 1024**3:
        raise RuntimeError("Not enough free space for the snapshot plus 2 GiB headroom")
    snapshot_download(REPO, revision=REVISION, local_dir=args.destination, token=token,
                      max_workers=3)
    verified = []
    for item in info.siblings:
        path = args.destination / item.rfilename
        if not path.is_file() or path.stat().st_size != item.size:
            raise RuntimeError(f"Incomplete download: {item.rfilename}")
        expected_hash = getattr(item.lfs, "sha256", None) if item.lfs else None
        with path.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        if expected_hash and digest != expected_hash:
            raise RuntimeError(f"SHA256 mismatch: {item.rfilename}")
        verified.append({"path": item.rfilename, "size": item.size, "sha256": digest})
        print(f"Verified {item.rfilename}", flush=True)
    report = {"repo": REPO, "revision": REVISION, "files": verified}
    (args.destination / "DOWNLOAD-VERIFIED.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"COMPLETE: {len(verified)} files verified in {args.destination}", flush=True)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
