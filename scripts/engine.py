"""Sole Forge entry point. No second page or Generate button."""
from pathlib import Path
import sys

_root = str(Path(__file__).resolve().parent.parent)
if _root not in sys.path:
    sys.path.insert(0, _root)

from pi_fibo_lib.bridge import start
start()

try:
    from modules import script_callbacks, shared
    from modules.options import OptionInfo
    import gradio as gr

    def settings():
        shared.opts.add_option("pi_fibo_degrid", OptionInfo(
            "Off", "FIBO 2px grid cleanup (DeGrid; optional)", gr.Dropdown,
            {"choices": ["Off", "Auto"]}, section=("pi_fibo", "Project Invisible FIBO")))
        shared.opts.add_option("pi_fibo_memory", OptionInfo(
            "Auto", "FIBO memory use", gr.Dropdown,
            {"choices": ["Auto", "Low VRAM"]}, section=("pi_fibo", "Project Invisible FIBO")))
    script_callbacks.on_ui_settings(settings)
except Exception as error:
    print(f"[Invisible-FIBO] Settings registration unavailable: {error}")
