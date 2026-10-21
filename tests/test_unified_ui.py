from copy import deepcopy
from types import SimpleNamespace as NS
import pytest
from forge_fibo2.unified_ui import single_choices,augment_result,install_for_host

RAW=['sd',('Custom label','custom'),'fibo','fibo2','fibo2_dev','qwen21']

@pytest.mark.parametrize('active',['fibo','fibo2','fibo2_dev'])
def test_exactly_one_fibo_label_and_saved_internal_value(active):
    assert single_choices(RAW,active)==['sd',('Custom label','custom'),('FIBO',active),'qwen21']

def test_unrelated_result_identity_and_arguments_are_preserved():
    target=object();value=({'value':['vae']},{'value':'qwen21'})
    assert augment_result(value,[object(),target],target,RAW) is value

def test_update_carries_fibo_choices_without_changing_other_native_outputs():
    target=object();native={'value':30};value=[native,{'__type__':'update','value':'fibo2_dev'}]
    updated=augment_result(value,[object(),target],target,RAW)
    assert updated[0] is native and updated[1]['value']=='fibo2_dev'
    assert ('FIBO','fibo2_dev') in updated[1]['choices']
    assert 'choices' not in value[1]

def fake_host():
    calls=[];events=[];app_callbacks=[];unload=[];marker=object()
    class Component:
        def __init__(self,value=None):
            self.value=value;self.choices=[(x,x) if isinstance(x,str) else x for x in RAW]
        def change(self,fn,**kwargs):events.append((fn,kwargs))
    target=Component('fibo2_dev');checkpoint=Component('Dev')
    def on_preset_change(preset,*args,**kwargs):calls.append(('preset',preset,args,kwargs));return marker
    def checkpoint_change(ckpt_name,preset,save=True,refresh=True,**kwargs):
        calls.append(('checkpoint',ckpt_name,preset,save,refresh,kwargs));return marker
    main=NS(make_checkpoint_manager_ui=lambda:marker,checkpoint_change=checkpoint_change,
            on_preset_change=on_preset_change,ui_forge_preset=target,ui_checkpoint=checkpoint)
    class Arch:
        @staticmethod
        def choices():return list(RAW)
    opts=NS(forge_preset='fibo2_dev',data={'forge_preset':'fibo2_dev','fibo_t2i_step':9,'fibo2_t2i_step':4,'fibo2_dev_t2i_step':31})
    callbacks=NS(on_app_started=app_callbacks.append,on_script_unloaded=unload.append)
    gr=NS(update=lambda **kw:dict(kw,__type__='update'),skip=lambda:marker)
    root=NS(load=lambda fn,**kwargs:events.append((fn,kwargs)))
    prepare_calls=[]
    handle=install_for_host(main,Arch,opts,callbacks,gr,
        lambda name:{'Dev':'fibo2_dev','Turbo':'fibo2','Legacy':'fibo'}.get(name),
        lambda *args,**kwargs:prepare_calls.append((args,kwargs)),lambda:root)
    return NS(**locals())

def test_registration_and_ui_build_preserve_saved_settings():
    h=fake_host();before=deepcopy(h.opts.data)
    try:
        assert h.main.make_checkpoint_manager_ui() is h.marker
        expected=single_choices([(x,x) if isinstance(x,str) else x for x in RAW],'fibo2_dev')
        assert h.main.ui_forge_preset.choices==expected
        assert ('sd','sd') in expected and ('qwen21','qwen21') in expected
        assert h.opts.data==before
        assert len(h.events)==2 and all(e[1]['outputs']==[h.target] for e in h.events)
    finally:h.handle.uninstall()

def test_checkpoint_route_preserves_native_kwargs_and_nonfibo_calls():
    h=fake_host()
    try:
        assert h.main.checkpoint_change(ckpt_name='Dev',preset='sd',save=False,refresh=False,future='kept') is h.marker
        assert h.calls[-1]==('checkpoint','Dev','fibo2_dev',False,False,{'future':'kept'})
        h.main.checkpoint_change('Other','qwen21',False,False,future='same')
        assert h.calls[-1]==('checkpoint','Other','qwen21',False,False,{'future':'same'})
        h.main.checkpoint_change('Dev',None,False,False)
        assert h.calls[-1][2] is None
    finally:h.handle.uninstall()

def test_existing_native_preset_recipe_and_other_presets_unchanged():
    h=fake_host()
    try:
        assert h.main.on_preset_change('fibo2_dev') is h.marker
        assert h.prepare_calls==[(('fibo2_dev',),{'persist':False})]
        h.main.on_preset_change('qwen21',future=True)
        assert h.calls[-1]==('preset','qwen21',(),{'future':True})
        assert len(h.prepare_calls)==1
    finally:h.handle.uninstall()

def test_ui_route_callback_updates_only_existing_preset():
    h=fake_host()
    try:
        h.main.make_checkpoint_manager_ui();fn,kwargs=h.events[0]
        assert fn('Legacy','fibo2')['value']=='fibo'
        assert fn('Dev','fibo2')['choices']==single_choices(RAW,'fibo2_dev')
        assert fn('Dev','fibo2_dev') is h.marker
        assert fn('Other','sd') is h.marker
        assert kwargs['inputs']==[h.checkpoint,h.target]
    finally:h.handle.uninstall()

def test_sync_and_async_callback_wrappers_and_unload():
    h=fake_host()
    def original(value,**kwargs):return [{'value':['encoder']},{'value':value}]
    async def async_original(value):return {'value':value}
    one=NS(fn=original,outputs=[object(),h.target]);two=NS(fn=async_original,outputs=[h.target])
    try:
        h.app_callbacks[0](NS(fns={0:one,1:two}),None)
        assert one.fn('fibo2')[1]['choices']==single_choices(RAW,'fibo2')
        # This callback is intentionally immediate; exercise its await chain
        # without opening asyncio's Windows loopback socket pair.
        coroutine=two.fn('fibo')
        with pytest.raises(StopIteration) as completed:coroutine.send(None)
        assert completed.value.value['choices']==single_choices(RAW,'fibo')
        unrelated=one.fn('sd');assert unrelated[1]=={'value':'sd'}
    finally:h.handle.uninstall()
    assert one.fn is original and two.fn is async_original
    assert h.Arch.choices()==RAW
