from types import SimpleNamespace as NS
from pathlib import Path
import ast
import torch
from PIL import Image
from forge_fibo2.preview import DevelopingPreview
from test_sampling import context
from forge_fibo2.engine import _sample, Fibo2Engine


def fixture(guidance=1, policy='resident-quantized'):
    state=NS(sampling_step=1,id_live_preview=0)
    decoded=[]
    engine=NS(_decode_latents=lambda x: decoded.append(x.clone()) or x)
    pipe=NS(fibo2_memory_policy=policy,scheduler=NS(sigmas=torch.tensor([1.,.5,0.]),step_index=1),
            image_processor=NS(postprocess=lambda x,output_type:[Image.new('RGB',(32,32))]))
    preview=DevelopingPreview(engine,state,NS(live_previews_enable=True),guidance)
    return preview,pipe,state,decoded


def test_cfg_separate_passes_and_immutable_sampler():
    preview,pipe,state,decoded=fixture(5)
    latents=torch.ones(1,32,4,4)*10
    positive=torch.ones_like(latents)*3
    negative=torch.ones_like(latents)
    preview.capture(None,(),(positive,))
    preview.capture(None,(),(negative,))
    preview.update(pipe,0,latents,4)
    torch.testing.assert_close(decoded[0],torch.ones_like(latents)*4.5)
    assert torch.all(latents==10) and torch.all(positive==3) and torch.all(negative==1)
    assert state.id_live_preview==1 and state.current_latent is None
    preview.capture(None,(),(positive,))
    preview.update(pipe,3,latents,4)
    assert len(decoded)==1


def test_offload_preview_continues_without_vae_transfer(monkeypatch):
    import forge_fibo2.preview as module
    clock=iter(range(0,200,2));monkeypatch.setattr(module,'monotonic',lambda:next(clock))
    preview,pipe,state,decoded=fixture(policy='model-cpu-offload')
    latents=torch.ones(1,32,130,66)
    for step in range(30):
        preview.capture(None,(),(torch.zeros_like(latents),))
        preview.update(pipe,step,latents,30)
    assert not decoded
    assert preview.frames==29
    assert state.id_live_preview==29
    assert state.current_image.size==(128,256)


def test_preview_failure_does_not_abort_sampling():
    preview,pipe,state,decoded=fixture()
    preview.engine._decode_latents=lambda x: (_ for _ in ()).throw(RuntimeError('test'))
    latents=torch.ones(1,32,4,4)
    preview.capture(None,(),(latents,));preview.update(pipe,0,latents,4)
    assert not preview.enabled and state.id_live_preview==0


def test_dev_uses_current_styled_batch_negatives(monkeypatch):
    p,state,pipe,noise=context(monkeypatch)
    p._fibo2_variant='dev';p.negative_prompt=['wrong zero','wrong one']
    p.negative_prompts=['styled batch two A','styled batch two B']
    _sample(p,None,None,[1,2],[],0,['a','b'])
    assert [call['negative_prompt'] for call in pipe.calls]==p.negative_prompts


def test_final_publish_occurs_after_native_compositing(monkeypatch):
    context(monkeypatch)
    import modules
    monkeypatch.setattr(modules,'shared',NS(state=NS(sampling_step=4,id_live_preview=0)),raising=False)
    source=Path(__file__).parents[1]/'scripts/fibo2.py'
    tree=ast.parse(source.read_text())
    method=next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name=='postprocess_image_after_composite')
    ns={};exec(compile(ast.fix_missing_locations(ast.Module(body=[method],type_ignores=[])),str(source),'exec'),ns)
    image=Image.new('RGBA',(50,70),(30,40,50,100))
    pp=NS(image=image);p=NS(sd_model=NS(is_fibo2=True))
    ns[method.name](None,p,pp)
    assert modules.shared.state.current_image is image
    assert modules.shared.state.current_image.mode=='RGBA'


def test_decode_restores_bn_without_changing_input_and_cpu_placement():
    seen=[]
    pipe=NS(_patchify_latents=lambda x:x,_unpatchify_latents=lambda x:x,_execution_device=torch.device('cpu'),
        vae=NS(dtype=torch.float32,bn=NS(running_mean=torch.ones(32),running_var=torch.ones(32)*4),
            config=NS(batch_norm_eps=0),decode=lambda x,return_dict:seen.append(x.clone()) or (x,)))
    engine=NS(pipe=pipe);x=torch.ones(1,32,4,4)
    result=Fibo2Engine._decode_latents(engine,x)
    assert torch.all(result==3) and torch.all(x==1) and seen[0].device.type=='cpu'


def test_mask_preview_restores_original_pixels_and_alpha(monkeypatch):
    context(monkeypatch)
    import modules
    from forge_fibo2.editing import prepare_edit
    original=Image.new('RGBA',(32,32),(20,40,60,100))
    mask=Image.new('L',(32,32));mask.paste(255,(8,8,24,24))
    prepared=prepare_edit(original,mask,(32,32))
    modules.processing=NS(apply_overlay=lambda image,paste,overlay:(Image.alpha_composite(image.convert('RGBA'),overlay),None))
    preview,pipe,state,decoded=fixture()
    preview.request=NS(_fibo2_edit_context=prepared,width=32,height=32)
    latent=torch.ones(1,32,4,4)
    preview.capture(None,(),(latent,));preview.update(pipe,0,latent,4)
    assert state.current_image.getpixel((0,0))==original.getpixel((0,0))
    assert state.current_image.getpixel((16,16))[3]==255


def test_heartbeat_updates_while_encoding_without_advancing_steps(monkeypatch):
    import time
    import pi_fibo_lib.progress as progress
    original=progress.LiveProgress
    monkeypatch.setattr(progress,'LiveProgress',lambda state:original(state,interval=.01))
    p,state,pipe,noise=context(monkeypatch)
    run=type(pipe).__call__
    statuses=[]
    def slow(self,**kwargs):
        time.sleep(.04)
        statuses.append((state.textinfo,state.sampling_step))
        return run(self,**kwargs)
    monkeypatch.setattr(type(pipe),'__call__',slow)
    _sample(p,None,None,[1,2],[],0,['one','two'])
    assert 'Encoding source and prompt' in statuses[0][0]
    assert statuses[0][1]==0
