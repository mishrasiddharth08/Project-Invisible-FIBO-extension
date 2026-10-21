from types import SimpleNamespace,ModuleType
import sys
import torch
from forge_fibo2.engine import _sample

def context(monkeypatch, stop=False):
    state=SimpleNamespace(interrupted=False,stopping_generation=False,skipped=False)
    class Interrupted(BaseException):
        pass
    module=ModuleType('modules')
    module.shared=SimpleNamespace(state=state)
    module.sd_samplers_common=SimpleNamespace(InterruptedException=Interrupted)
    monkeypatch.setitem(sys.modules,'modules',module)
    monkeypatch.setitem(sys.modules,'modules.sd_samplers_common',module.sd_samplers_common)
    class Pipeline:
        calls=[]
        fibo2_fonts=[]
        def __call__(self,**kwargs):
            self.calls.append(kwargs)
            for step in range(4):
                if stop and step==1:
                    if stop=='skip':
                        state.skipped=True
                    else:
                        state.interrupted=True
                kwargs['callback_on_step_end'](self,step,step,{'latents':kwargs['latents']+step})
            return SimpleNamespace(images=kwargs['latents']+4)
        def maybe_free_model_hooks(self):
            pass
    pipe=Pipeline()
    noise=torch.zeros(2,32,4,4)
    noise[1]=10
    p=SimpleNamespace(sd_model=SimpleNamespace(pipe=pipe),steps=4,width=32,height=32,
        _fibo2_source=None,_fibo2_mask=None,sampler=SimpleNamespace(last_latent=None),
        scripts=None,rng=SimpleNamespace(next=lambda:noise.clone()))
    return p,state,pipe,noise

def test_native_rng_latents_batch_order_and_progress(monkeypatch):
    p,state,pipe,noise=context(monkeypatch)
    result=_sample(p,None,None,[101,102],[],0,['one','two'])
    assert torch.equal(result,noise+4)
    assert [call['generator'].initial_seed() for call in pipe.calls]==[101,102]
    assert state.sampling_step==state.sampling_steps==8

def test_cancellation_returns_only_real_partial_latents(monkeypatch):
    p,state,pipe,noise=context(monkeypatch,stop=True)
    result=_sample(p,None,None,[101,102],[],0,['one','two'])
    assert state.interrupted
    assert len(pipe.calls)==1
    assert torch.equal(result,noise[:1]+1)

def test_native_sampling_noise_hook_is_honored(monkeypatch):
    p,state,pipe,noise=context(monkeypatch)
    p.scripts=SimpleNamespace(process_before_every_sampling=lambda p,**kw:kw['noise'].add_(3))
    result=_sample(p,None,None,[101,102],[],0,['one','two'])
    assert torch.equal(result,noise+7)

def test_skip_preserves_native_flag_and_returns_real_partial_latents(monkeypatch):
    p,state,pipe,noise=context(monkeypatch,stop='skip')
    result=_sample(p,None,None,[101,102],[],0,['one','two'])
    assert state.skipped and not state.interrupted
    assert torch.equal(result,noise[:1]+1)
