"""Narrow loader hook; no checkpoint manifests or whole-workflow interception."""
from functools import wraps
from pathlib import Path
import inspect
from .contract import is_fibo2_checkpoint

ROOT = Path(__file__).resolve().parent.parent

def install():
    from backend import loader
    from modules import sd_models, script_callbacks, sd_samplers_common, shared
    original = sd_models.forge_loader
    if getattr(original, '_fibo2_loader', False):
        return
    if 'additional_state_dicts' not in inspect.signature(original).parameters:
        raise RuntimeError('Forge loader contract changed; FIBO 2.0 disabled.')
    original_decode = sd_samplers_common.samples_to_images_tensor
    if not callable(original_decode) or not callable(script_callbacks.on_script_unloaded):
        raise RuntimeError('Forge lifecycle contract changed; FIBO 2.0 disabled.')
    try:
        inspect.signature(original_decode).bind(object(), approximation=None, model=None)
    except TypeError as error:
        raise RuntimeError('Forge decoder signature changed; FIBO 2.0 disabled.') from error
    @wraps(original)
    def load(sd, additional_state_dicts=None, **kwargs):
        from .engine import release_current, Fibo2Engine
        if not is_fibo2_checkpoint(sd):
            release_current()
            return original(sd, additional_state_dicts=additional_state_dicts, **kwargs)
        if kwargs:
            raise RuntimeError('Unsupported Forge loader arguments for FIBO 2.0.')
        from .runtime import load as load_pipeline
        import json
        release_current()
        manifest_path = ROOT / 'model-manifest.json'
        if not manifest_path.is_file():
            raise RuntimeError('FIBO 2.0 setup is missing. Run Setup-FIBO2.cmd in the FIBO extension folder, then restart Forge.')
        manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
        selected_te = None
        selected_vae = None
        from safetensors import safe_open
        for filename in additional_state_dicts or []:
            with safe_open(str(filename), framework='pt', device='cpu') as weights:
                metadata = weights.metadata() or {}
            role = metadata.get('project_invisible.role')
            if role is None:
                from .provenance import PINS,identify
                roles={entry['role'] for entry in PINS['assets'] if entry['bytes']==Path(filename).stat().st_size}
                if len(roles)==1:
                    role=next(iter(roles));identify(filename,role)
            if role == 'text_encoder':
                if selected_te is not None:
                    raise ValueError('Select one FIBO 2.0 text encoder.')
                selected_te = filename
            elif role == 'vae':
                if selected_vae is not None:
                    raise ValueError('Select one FIBO 2.0 VAE.')
                selected_vae=filename
            else:
                raise ValueError(f'Unsupported FIBO 2.0 additional module: {Path(filename).name}')
        if selected_te is None or selected_vae is None:
            raise ValueError('Select one FIBO 2.0 text encoder and its FLUX2 VAE in the main model bar.')
        pipe = load_pipeline(manifest_path, checkpoint=sd, text_encoder=selected_te,vae=selected_vae)
        return Fibo2Engine(pipe, manifest)
    load._fibo2_loader = True
    sd_models.forge_loader = load
    backend_original = loader.forge_loader
    if loader.forge_loader is original:
        loader.forge_loader = load
    from .engine import release_current
    script_callbacks.on_script_unloaded(release_current)
    def restore_loader():
        if sd_models.forge_loader is load:
            sd_models.forge_loader = original
        if loader.forge_loader is load:
            loader.forge_loader = backend_original
    script_callbacks.on_script_unloaded(restore_loader)
    from . import preset
    preset.install()
    # Interrupted jobs can otherwise select SD's fast decoder even when Full
    # was requested. Route only this model's real 32-channel latents to its VAE.
    @wraps(original_decode)
    def decode(sample, approximation=None, model=None):
        current = model or shared.sd_model
        if getattr(current, 'is_fibo2', False):
            return current.decode_first_stage(sample)
        return original_decode(sample, approximation=approximation, model=model)
    sd_samplers_common.samples_to_images_tensor = decode
    def restore_decode():
        if sd_samplers_common.samples_to_images_tensor is decode:
            sd_samplers_common.samples_to_images_tensor = original_decode
    script_callbacks.on_script_unloaded(restore_decode)
    try:
        from . import autolink
        handle = autolink.register_runtime_policy(ROOT / 'model-manifest.json')
        script_callbacks.on_script_unloaded(handle.uninstall)
    except Exception as error:
        print('[FIBO 2.0] Autolink unavailable:', error)
    print('[FIBO 2.0] Compatibility engine ready; no accordion; Forge processing retained.')
