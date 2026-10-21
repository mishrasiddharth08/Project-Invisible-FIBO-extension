"""Variant-specific native request contracts; architecture is not model identity."""
from pathlib import Path
import json
import struct

def is_fibo2_checkpoint(filename):
    """Read structure, not a filename; never deserialize pickle."""
    path = Path(filename)
    if path.suffix.lower() != '.safetensors' or not path.is_file():
        return False
    try:
        with path.open('rb') as stream:
            length = struct.unpack('<Q', stream.read(8))[0]
            if length > 32 * 1024 * 1024:
                return False
            header = json.loads(stream.read(length))
        return (header.get('x_embedder.weight', {}).get('shape') == [3584, 128]
                and any(k.startswith('layers.43.') for k in header)
                and any('perceiver' in k for k in header)
                and any('cross_attention' in k for k in header))
    except (OSError, ValueError, struct.error):
        return False

def validate_request(p, variant='edit-turbo'):
    if variant not in ('dev', 'edit-turbo'):
        raise ValueError('Unknown FIBO 2.0 variant; model identity must be verified.')
    sources = getattr(p, 'init_images', None) or []
    if variant == 'edit-turbo' and not sources:
        raise ValueError('FIBO 2.0 Edit Turbo needs a source image in img2img.')
    if len(sources) > 1:
        raise ValueError('This initial Forge adapter supports one source image per request.')
    if getattr(p,'image_mask',None) is not None and getattr(p,'inpainting_fill',0)!=0:
        raise ValueError('FIBO masked reference editing uses neutral Fill. Original/latent noise/latent nothing fill modes are unsupported.')
    if variant == 'edit-turbo' and float(p.cfg_scale) != 1.0:
        raise ValueError('FIBO 2.0 Edit Turbo requires CFG 1.')
    if variant == 'edit-turbo' and int(p.steps) != 4:
        raise ValueError('Select 4 steps for the validated FIBO 2.0 Edit Turbo recipe.')
    if variant == 'dev' and (int(p.steps) != 30 or float(p.cfg_scale) != 5.0):
        raise ValueError('Select 30 steps and CFG 5 for the staged FIBO 2.0 Dev recipe.')
    if p.width <= 0 or p.height <= 0 or p.width % 16 or p.height % 16:
        raise ValueError('Width and height must be multiples of 16.')
    # Forge's ordinary img2img API defaults inpaint_full_res=True even without
    # a mask. That flag only has meaning when a painted mask is present.
    if getattr(p, 'resize_mode', 0) != 0:
        raise ValueError('This FIBO 2.0 build supports Just resize.')
    if getattr(p, 'subseed_strength', 0):
        raise ValueError('Variation seeds are not supported by this FIBO 2.0 engine.')
    if getattr(p, 'enable_hr', False) or getattr(p, 'refiner_checkpoint_info', None):
        raise ValueError('Hires fix and refiner are not supported by this FIBO 2.0 engine.')
    if getattr(p, 'tiling', False):
        raise ValueError('Tiling is not supported by this FIBO 2.0 engine.')
    if getattr(p, 'sampler_name', 'Euler') != 'Euler' or getattr(p, 'scheduler', 'Simple') not in ('Simple', 'Automatic'):
        raise ValueError('Select Euler / Simple; FIBO 2.0 uses its official flow schedule.')
    prompts = p.prompt if isinstance(p.prompt, list) else [p.prompt]
    if any('<lora:' in s.lower() or '<lyco:' in s.lower() for s in prompts):
        raise ValueError('LoRA/LoKr are not validated for FIBO 2.0 in this build.')
