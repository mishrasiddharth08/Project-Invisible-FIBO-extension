"""Read-only host API and core-file integrity audit against the captured baseline."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
parser = argparse.ArgumentParser()
parser.add_argument("--forge", type=Path, required=True)
parser.add_argument("--baseline", type=Path, required=True)
args = parser.parse_args()
baseline = json.loads(args.baseline.read_text(encoding="utf-8-sig"))
changed = []
for relative, expected in baseline.items():
    path = args.forge / relative
    if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest().upper() != expected:
        changed.append(relative)
if changed:
    raise RuntimeError("Core files differ from initial baseline: " + ", ".join(changed))
expected = {"modules/processing.py": {"process_images", "Processed"},
            "modules/sd_models.py": {"CheckpointInfo", "list_models"},
            "modules/scripts.py": {"ScriptRunner"}}
for relative, symbols in expected.items():
    tree = ast.parse((args.forge / relative).read_text(encoding="utf-8-sig"))
    defined = {node.name for node in tree.body if isinstance(node, (ast.FunctionDef, ast.ClassDef))}
    if not symbols <= defined:
        raise RuntimeError(f"Host interface changed: {relative}: {symbols - defined}")
print(f"PASS: {len(baseline)} tracked Forge files unchanged from baseline; required host entry points present")
