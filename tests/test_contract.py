import json
import struct
from types import SimpleNamespace
import pytest
from forge_fibo2.contract import is_fibo2_checkpoint, validate_request

def request(**changes):
    values = dict(init_images=[object()], cfg_scale=1, steps=4, width=512, height=512,
                  prompt='make the jacket red', sampler_name='Euler', scheduler='Simple')
    values.update(changes)
    return SimpleNamespace(**values)

@pytest.mark.parametrize('changes', [dict(init_images=[]), dict(cfg_scale=7), dict(steps=20),
    dict(width=513), dict(subseed_strength=0.3),
    dict(resize_mode=2), dict(sampler_name='DPM++ 2M'), dict(prompt='<lora:foreign:1>')])
def test_unsupported_requests_are_explicit(changes):
    with pytest.raises(ValueError):
        validate_request(request(**changes))

def test_supported_request():
    validate_request(request())

def test_unsupported_inpaint_fill_is_explicit():
    with pytest.raises(ValueError,match='neutral Fill'):
        validate_request(request(image_mask=object(),inpainting_fill=2))

def test_detector_uses_structure_not_filename(tmp_path):
    path = tmp_path/'anything.safetensors'
    header = {'x_embedder.weight': {'shape':[3584,128]}, 'layers.43.cross_attention.to_k.weight':{}, 'perceiver.weight':{}}
    encoded = json.dumps(header).encode()
    path.write_bytes(struct.pack('<Q',len(encoded))+encoded)
    assert is_fibo2_checkpoint(path)
    path.write_bytes(b'not a model')
    assert not is_fibo2_checkpoint(path)

def test_no_accordion_or_generation_override():
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    script = (root/'scripts/fibo2.py').read_text()
    integration = (root/'forge_fibo2/integration.py').read_text()
    assert 'gr.Accordion' not in script and 'gr.Tab' not in script
    assert 'return []' in script and 'create_group = False' in script
    assert 'processing.process_images =' not in integration
