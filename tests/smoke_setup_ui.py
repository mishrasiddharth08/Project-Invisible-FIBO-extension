"""Build both setup accordions using the installed Gradio without starting Forge."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import gradio as gr
from pi_fibo_lib.model_setup import build_ui
with gr.Blocks() as page:
    assert build_ui(False) == []
    assert build_ui(True) == []
config = page.get_config_file()
dropdowns = [c for c in config["components"] if c["type"] == "dropdown"]
assert [c["props"]["value"] for c in dropdowns] == ["FIBO 1.5", "FIBO Edit 1.5 Turbo"]
buttons = [c["props"]["value"] for c in config["components"] if c["type"] == "button"]
assert len(buttons) == 4
assert all(b in ("Use existing complete folder", "Download official full snapshot") for b in buttons)
print("PASS: both native setup panels build; no duplicate Generate/edit controls")
