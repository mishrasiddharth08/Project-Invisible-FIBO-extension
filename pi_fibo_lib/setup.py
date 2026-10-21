"""Validate and register an existing local model folder. No download or login."""
import argparse
import json
from pathlib import Path
from .catalog import ROOT, VARIANTS, validate_folder, read_json

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=VARIANTS, default="fibo15")
    parser.add_argument("--folder", required=True)
    args = parser.parse_args()
    folder = validate_folder(Path(args.folder), args.variant)
    target = ROOT / "model_locations.json"
    values = read_json(target) if target.is_file() else {}
    values[args.variant] = str(folder)
    temporary = target.with_suffix(".tmp")
    temporary.write_text(json.dumps(values, indent=2), encoding="utf-8")
    temporary.replace(target)
    print(f"Registered {VARIANTS[args.variant].label}: {folder}. Refresh the Forge checkpoint dropdown.")

if __name__ == "__main__":
    main()
