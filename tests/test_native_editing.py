from types import SimpleNamespace,ModuleType
import sys
import numpy as np
from PIL import Image
import torch
import pytest
from forge_fibo2.editing import prepare_edit,restore_masked_alpha

def images():
    source=Image.new('RGBA',(64,48),(31,57,93,127))
    mask=Image.new('L',source.size,0);mask.paste(255,(20,15,35,30))
    return source,mask

def test_context_crop_and_exact_outside_rgba():
    source,mask=images()
    prepared=prepare_edit(source,mask,(32,32),only_masked=True,padding=4,blur_x=2,blur_y=2)
    assert prepared.reference.size==(32,32) and prepared.mask.size==(32,32)
    assert prepared.paste_to is not None
    result=restore_masked_alpha(Image.new('RGB',source.size,'green'),prepared)
    outside=np.asarray(mask)==0
    assert result.size==source.size and result.mode=='RGBA'
    assert np.array_equal(np.asarray(result)[outside],np.asarray(source)[outside])
    assert np.all(np.asarray(prepared.blend)[outside]==0)
    assert np.any((np.asarray(prepared.blend)>0)&(np.asarray(prepared.blend)<255))

def test_threshold_and_empty_mask_are_explicit():
    source,mask=images();mask.paste(127,(0,0,4,4))
    prepared=prepare_edit(source,mask,(64,48))
    assert prepared.support.getpixel((0,0))==0
    with pytest.raises(ValueError,match='Paint an area'):
        prepare_edit(source,Image.new('L',source.size,0),(32,32))

def test_rgb_mask_inversion_uses_native_control(monkeypatch):
    from forge_fibo2.engine import _init
    module=ModuleType('modules');module.shared=SimpleNamespace(opts=SimpleNamespace(sd_vae_decode_method='Full'))
    monkeypatch.setitem(sys.modules,'modules',module)
    source=Image.new('RGB',(32,32),'red')
    p=SimpleNamespace(init_images=[source],image_mask=Image.new('RGB',source.size,'black'),
        inpainting_mask_invert=1,cfg_scale=1,steps=4,width=32,height=32,prompt='edit',batch_size=1)
    _init(p)
    assert p._fibo2_mask.getextrema()==(255,255)

def test_source_a_b_then_dev_has_no_stale_reference(monkeypatch):
    from forge_fibo2.engine import _init
    module=ModuleType('modules');module.shared=SimpleNamespace(opts=SimpleNamespace(sd_vae_decode_method='Full',samples_format='png'))
    monkeypatch.setitem(sys.modules,'modules',module)
    source,mask=images()
    p=SimpleNamespace(init_images=[source],image_mask=mask,cfg_scale=1,steps=4,width=32,height=32,prompt='edit',batch_size=1,extra_generation_params={})
    _init(p);first=p._fibo2_source.tobytes()
    p.init_images=[Image.new('RGB',source.size,'red')];p.image_mask=None
    _init(p)
    assert p._fibo2_source.tobytes()!=first and p._fibo2_edit_context is None
    p.init_images=[];p._fibo2_variant='dev';p.steps=30;p.cfg_scale=5
    _init(p)
    assert p._fibo2_source is None and p._fibo2_mask is None and p._fibo2_edit_context is None

def test_dev_sampler_preserves_negative_prompt_and_uses_noise(monkeypatch):
    from forge_fibo2.engine import _sample
    state=SimpleNamespace(interrupted=False,stopping_generation=False,skipped=False)
    module=ModuleType('modules');module.shared=SimpleNamespace(state=state)
    samplers=ModuleType('modules.sd_samplers_common');samplers.InterruptedException=type('Interrupted',(BaseException,),{})
    monkeypatch.setitem(sys.modules,'modules',module);monkeypatch.setitem(sys.modules,'modules.sd_samplers_common',samplers)
    calls=[]
    def pipeline(**kwargs):
        calls.append(kwargs);return SimpleNamespace(images=kwargs['latents'])
    p=SimpleNamespace(sd_model=SimpleNamespace(pipe=pipeline),steps=30,cfg_scale=5,width=32,height=32,
        _fibo2_variant='dev',_fibo2_source=None,_fibo2_mask=None,negative_prompt='bad lighting',
        rng=SimpleNamespace(next=lambda:torch.zeros(1,32,4,4)),scripts=None)
    _sample(p,None,None,[64],[],0,['a cube'])
    assert calls[0]['image'] is None and calls[0]['negative_prompt']=='bad lighting'
    assert calls[0]['guidance_scale']==5 and calls[0]['num_inference_steps']==30

def test_no_added_tab_accordion_or_private_generate():
    from pathlib import Path
    scripts=Path(__file__).resolve().parents[1]/'scripts'
    source='\n'.join(path.read_text(encoding='utf-8') for path in scripts.glob('*.py'))
    assert 'gr.Accordion' not in source and 'gr.Tab' not in source and 'Generate edit' not in source
