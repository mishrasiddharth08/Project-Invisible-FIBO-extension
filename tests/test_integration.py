from types import SimpleNamespace, ModuleType
import sys
from forge_fibo2 import integration, preset, autolink
import json
import pytest
import pytest

def test_request_validation_runs_outside_forges_swallowed_script_callback():
    from forge_fibo2.engine import Fibo2Engine
    p=SimpleNamespace(extra_generation_params={},init_images=[])
    engine=SimpleNamespace(pipe=SimpleNamespace(_fibo2_precision='BF16'))
    Fibo2Engine.bind(engine,p)
    with pytest.raises(ValueError,match='source image'):
        p.init([],[],[])

def test_loader_forwards_non_fibo_and_restores_owned_hooks(monkeypatch):
    calls, callbacks = [], []
    def original(sd, additional_state_dicts=None, **kwargs):
        calls.append((sd,additional_state_dicts,kwargs))
        return 'stock result'
    def original_decode(sample, approximation=None, model=None):
        return sample,approximation,model
    loader=SimpleNamespace(forge_loader=original)
    models=SimpleNamespace(forge_loader=original)
    sampler=SimpleNamespace(samples_to_images_tensor=original_decode)
    shared=SimpleNamespace(sd_model=SimpleNamespace(is_fibo2=False))
    callbacks_module=SimpleNamespace(on_script_unloaded=callbacks.append)
    backend=ModuleType('backend')
    backend.loader=loader
    modules=ModuleType('modules')
    modules.sd_models,modules.script_callbacks=models,callbacks_module
    modules.sd_samplers_common,modules.shared=sampler,shared
    monkeypatch.setitem(sys.modules,'backend',backend)
    monkeypatch.setitem(sys.modules,'modules',modules)
    monkeypatch.setattr(integration,'is_fibo2_checkpoint',lambda _:False)
    monkeypatch.setattr(preset,'install',lambda:True)
    monkeypatch.setattr(autolink,'register_runtime_policy',lambda _:SimpleNamespace(uninstall=lambda:None))
    integration.install()
    wrapped=models.forge_loader
    assert wrapped('normal.safetensors',additional_state_dicts=['te'],future_option=True)=='stock result'
    assert calls==[('normal.safetensors',['te'],{'future_option':True})]
    assert sampler.samples_to_images_tensor('latent',approximation=2)==('latent',2,None)
    shared.sd_model=SimpleNamespace(is_fibo2=True,decode_first_stage=lambda x:'fibo '+x)
    assert sampler.samples_to_images_tensor('latent',approximation=2)=='fibo latent'
    integration.install()
    assert models.forge_loader is wrapped
    for callback in callbacks:
        callback()
    assert models.forge_loader is original
    assert loader.forge_loader is original
    assert sampler.samples_to_images_tensor is original_decode


def test_fibo_never_uses_hidden_component_selection(monkeypatch,tmp_path):
    def original(sd,additional_state_dicts=None,**kwargs): return 'stock'
    def decode(sample,approximation=None,model=None): return sample
    loader=SimpleNamespace(forge_loader=original);models=SimpleNamespace(forge_loader=original)
    backend=ModuleType('backend');backend.loader=loader
    modules=ModuleType('modules');modules.sd_models=models
    modules.script_callbacks=SimpleNamespace(on_script_unloaded=lambda _:None)
    modules.sd_samplers_common=SimpleNamespace(samples_to_images_tensor=decode)
    modules.shared=SimpleNamespace(sd_model=None)
    monkeypatch.setitem(sys.modules,'backend',backend);monkeypatch.setitem(sys.modules,'modules',modules)
    monkeypatch.setattr(integration,'is_fibo2_checkpoint',lambda _:True)
    monkeypatch.setattr(integration,'ROOT',tmp_path)
    (tmp_path/'model-manifest.json').write_text(json.dumps({'components':{'vae':{'path':'unused'}}}))
    monkeypatch.setattr(preset,'install',lambda:True)
    monkeypatch.setattr(autolink,'register_runtime_policy',lambda _:SimpleNamespace(uninstall=lambda:None))
    integration.install()
    with pytest.raises(ValueError,match='main model bar'):
        models.forge_loader('fibo.safetensors',additional_state_dicts=[])
