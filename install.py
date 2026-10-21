"""Read-only dependency check. No upgrades, downloads or core modifications."""
import importlib.util
missing=[name for name in ('torch','diffusers','transformers','safetensors','huggingface_hub') if importlib.util.find_spec(name) is None]
if missing:
    print('[FIBO] Missing existing Forge dependencies: '+', '.join(missing))
else:
    print('[FIBO] Existing Forge environment retained. Use Setup-FIBO2.cmd for explicit model setup.')
