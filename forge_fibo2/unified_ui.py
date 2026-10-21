"""One displayed FIBO preset; retain model-specific saved option namespaces.

Only the existing preset component and its FIBO-valued updates are adapted.
Processing, Generate, scripts and all other preset values pass through unchanged.
"""
from functools import wraps
import inspect
from types import SimpleNamespace

ALIASES=frozenset(('fibo','fibo2','fibo2_dev'))
LABEL='FIBO'

def single_choices(choices,active='fibo2'):
    active=active if active in ALIASES else 'fibo2'
    result=[];inserted=False
    for choice in choices:
        value=choice[1] if isinstance(choice,(tuple,list)) else choice
        if value in ALIASES:
            if not inserted:result.append((LABEL,active));inserted=True
        else:result.append(choice)
    if not inserted:result.append((LABEL,active))
    return result

def augment_result(result,outputs,target,choices):
    """Attach choices to FIBO value updates; preserve every unrelated result."""
    indexes=[i for i,output in enumerate(outputs) if output is target]
    if not indexes:return result
    def update(value):
        selected=value.get('value') if isinstance(value,dict) else value
        if not isinstance(selected,str) or selected not in ALIASES:return value
        payload=dict(value) if isinstance(value,dict) else {'__type__':'update','value':selected}
        payload['choices']=single_choices(choices,selected)
        return payload
    if len(outputs)==1:return update(result)
    if not isinstance(result,(tuple,list)) or len(result)!=len(outputs):return result
    changed=list(result)
    for index in indexes:changed[index]=update(changed[index])
    if all(a is b for a,b in zip(changed,result)):return result
    return tuple(changed) if isinstance(result,tuple) else changed

def checkpoint_route(name,sd_models):
    if not name:return None
    info=sd_models.checkpoint_aliases.get(name)
    if info is None:info=sd_models.get_closet_checkpoint_match(name)
    if info is None:return None
    from pi_fibo_lib.catalog import selected_variant
    if selected_variant(info.filename):return 'fibo'
    from .contract import is_fibo2_checkpoint
    if not is_fibo2_checkpoint(info.filename):return None
    from .runtime import _validate_metadata,RuntimeAssetError
    metadata,_=_validate_metadata(info.filename,'transformer')
    variant=metadata.get('verified_official_variant',metadata.get('project_invisible.variant'))
    if variant=='dev':return 'fibo2_dev'
    if variant=='edit-turbo':return 'fibo2'
    raise RuntimeAssetError('Cannot identify the selected FIBO Dev/Edit variant; select a verified checkpoint.')

def install_for_host(main,arch,opts,callbacks,gr,route,prepare,root_provider):
    existing=getattr(arch,'_fibo_single_ui',None)
    if existing is not None and existing.active:return existing
    originals={name:getattr(main,name) for name in ('make_checkpoint_manager_ui','checkpoint_change','on_preset_change')}
    inspect.signature(originals['make_checkpoint_manager_ui']).bind()
    inspect.signature(originals['checkpoint_change']).bind(ckpt_name='checkpoint',preset='sd',save=False,refresh=False)
    inspect.signature(originals['on_preset_change']).bind('sd')
    if not all(callable(getattr(callbacks,name,None)) for name in ('on_app_started','on_script_unloaded')):
        raise RuntimeError('FIBO preset callback contract changed; single-entry adapter disabled.')
    original_choices=arch.choices
    original_choices()
    handle=SimpleNamespace(active=True,wrapped=[])

    @wraps(original_choices)
    def choices(*args,**kwargs):
        raw=original_choices(*args,**kwargs)
        return single_choices(raw,getattr(opts,'forge_preset',None)) if handle.active else raw

    @wraps(originals['checkpoint_change'])
    def checkpoint_change(*args,**kwargs):
        name=args[0] if args else kwargs.get('ckpt_name')
        preset=args[1] if len(args)>1 else kwargs.get('preset')
        if handle.active and name and preset is not None:
            selected=route(name)
            if selected is not None:
                if len(args)>1:args=(args[0],selected,*args[2:])
                else:kwargs=dict(kwargs,preset=selected)
        return originals['checkpoint_change'](*args,**kwargs)

    @wraps(originals['on_preset_change'])
    def on_preset_change(preset,*args,**kwargs):
        if handle.active and preset in ALIASES and prepare is not None:
            prepare(preset,persist=False)
        return originals['on_preset_change'](preset,*args,**kwargs)

    def route_selection(name,visible_preset):
        if not handle.active:return gr.skip()
        selected=route(name)
        if selected is None or selected==visible_preset:return gr.skip()
        return gr.update(choices=single_choices(original_choices(),selected),value=selected)

    @wraps(originals['make_checkpoint_manager_ui'])
    def make_ui(*args,**kwargs):
        result=originals['make_checkpoint_manager_ui'](*args,**kwargs)
        if not handle.active:return result
        target=main.ui_forge_preset
        # Preserve Gradio's normalized (label,value) pairs for other presets.
        target.choices=single_choices(target.choices,target.value)
        if not getattr(target,'_fibo_route_bound',False):
            main.ui_checkpoint.change(route_selection,inputs=[main.ui_checkpoint,target],outputs=[target],
                queue=False,show_progress=False,api_name='fibo_native_route')
            root_provider().load(route_selection,inputs=[main.ui_checkpoint,target],outputs=[target],
                queue=False,show_progress=False,api_name=False)
            target._fibo_route_bound=True
        return result

    def bind_outputs(demo,app):
        if not handle.active:return
        target=main.ui_forge_preset
        candidates=[]
        for function in demo.fns.values():
            if not any(output is target for output in function.outputs) or function.fn is None:continue
            original=function.fn
            if getattr(original,'_fibo_single_handle',None) is handle:continue
            if inspect.isgeneratorfunction(original) or inspect.isasyncgenfunction(original):
                raise RuntimeError('FIBO preset output contract changed; generator callback needs an adapter.')
            outputs=list(function.outputs)
            def wrap(original,outputs):
                if inspect.iscoroutinefunction(original):
                    @wraps(original)
                    async def wrapped(*args,**kwargs):
                        value=await original(*args,**kwargs)
                        return augment_result(value,outputs,target,original_choices()) if handle.active else value
                else:
                    @wraps(original)
                    def wrapped(*args,**kwargs):
                        value=original(*args,**kwargs)
                        return augment_result(value,outputs,target,original_choices()) if handle.active else value
                wrapped._fibo_single_handle=handle
                return wrapped
            candidates.append((function,original,wrap(original,outputs)))
        try:
            for function,original,wrapped in candidates:
                function.fn=wrapped;handle.wrapped.append((function,original,wrapped))
        except Exception:
            uninstall();raise
        print('[FIBO] Single existing UI Preset entry ready; saved version-specific options retained.')

    def uninstall():
        handle.active=False
        for function,original,wrapped in handle.wrapped:
            if function.fn is wrapped:function.fn=original
        for name,replacement in replacements.items():
            if getattr(main,name) is replacement:setattr(main,name,originals[name])
        if arch.choices is choices:arch.choices=staticmethod(original_choices)
        if getattr(arch,'_fibo_single_ui',None) is handle:delattr(arch,'_fibo_single_ui')

    replacements={'make_checkpoint_manager_ui':make_ui,'checkpoint_change':checkpoint_change,'on_preset_change':on_preset_change}
    handle.uninstall=uninstall;handle.bind_outputs=bind_outputs
    try:
        arch.choices=staticmethod(choices)
        for name,replacement in replacements.items():setattr(main,name,replacement)
        arch._fibo_single_ui=handle
        callbacks.on_app_started(bind_outputs);callbacks.on_script_unloaded(uninstall)
    except Exception:
        uninstall();raise
    return handle

def install():
    from modules import shared,sd_models,script_callbacks
    from modules_forge import main_entry,presets
    from gradio.context import Context
    import gradio as gr
    try:
        from model_autolink.integration import _prepare_preset
    except ImportError:_prepare_preset=None
    return install_for_host(main_entry,presets.PresetArch,shared.opts,script_callbacks,gr,
        lambda name:checkpoint_route(name,sd_models),_prepare_preset,lambda:Context.root_block)
