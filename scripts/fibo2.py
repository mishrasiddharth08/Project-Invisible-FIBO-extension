"""No UI controls, tab, accordion or private Generate button."""
from pathlib import Path
import sys
root = str(Path(__file__).resolve().parent.parent)
if root not in sys.path:
    sys.path.insert(0, root)
from modules import scripts
from forge_fibo2.integration import install
install()
from forge_fibo2.unified_ui import install as install_single_fibo_entry
install_single_fibo_entry()

# Explain reference-conditioning semantics on the existing native control.
# This adds no component, tab or accordion and leaves its value/behavior intact.
from modules import script_callbacks
def explain_native_controls(component,**kwargs):
    if getattr(component,'elem_id',None)=='img2img_denoising_strength':
        existing=getattr(component,'info',None) or ''
        component.info=(existing+' FIBO 2.0 reference edits start the target from noise; this denoising-strength control does not apply.').strip()
script_callbacks.on_after_component(explain_native_controls)

class Fibo2Lifecycle(scripts.Script):
    create_group = False
    def title(self):
        return 'FIBO 2.0 engine lifecycle'
    def show(self, is_img2img):
        return scripts.AlwaysVisible
    def ui(self, is_img2img):
        return []
    def process(self, p, *args):
        if getattr(p.sd_model, 'is_fibo2', False):
            p.sd_model.bind(p)

    def postprocess_image_after_composite(self,p,pp,*args):
        prepared=getattr(p,'_fibo2_edit_context',None)
        if prepared is not None:
            from forge_fibo2.editing import restore_masked_alpha
            pp.image=restore_masked_alpha(pp.image,prepared)
