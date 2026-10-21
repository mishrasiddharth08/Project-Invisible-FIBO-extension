import hashlib,json
from pathlib import Path
from types import SimpleNamespace,ModuleType
import pytest
import torch
from forge_fibo2.layouts import unpack_transformer,preserve_ffn_widths,unpack_vae,LayoutError
from forge_fibo2.contract import validate_request
from forge_fibo2 import provenance

ROOT=Path(__file__).resolve().parents[1]


def test_attention_accepts_official_fp32_norms_with_bf16_projections():
    from forge_fibo2.vendor.diffusers.models.transformers.transformer_bria_fibo2 import BriaFibo2Attention
    layer=BriaFibo2Attention(16,2,8).to(dtype=torch.bfloat16)
    layer.norm_q.float()
    layer.norm_k.float()
    value=layer(torch.randn(1,3,16,dtype=torch.bfloat16),
                encoder_hidden_states=torch.randn(1,5,16,dtype=torch.bfloat16))
    assert value.dtype==torch.bfloat16
    assert torch.isfinite(value).all()
    assert layer.norm_q.weight.dtype==torch.float32

def meta_state(filename):
    header=json.loads((ROOT/'review'/filename).read_text())
    types={'BF16':torch.bfloat16,'F32':torch.float32,'I64':torch.int64,'I8':torch.int8,'U8':torch.uint8}
    return {key:torch.empty(value['shape'],device='meta',dtype=types[value['dtype']])
            for key,value in header.items() if key!='__metadata__'}

def test_official_transformer_header_covers_every_diffusers_weight_and_shape():
    from forge_fibo2.official import official_classes
    state=unpack_transformer(meta_state('fibo2_edit_turbo_int8_convrot-header.json'))
    cls=official_classes()['transformer']
    config=cls.load_config(str(ROOT/'tests/fixtures/configs/transformer'))
    with torch.device('meta'):
        model=preserve_ffn_widths(cls.from_config(config),state)
    primary={k:v for k,v in state.items() if not k.endswith(('.comfy_quant','.weight_scale','.input_scale'))}
    assert set(primary)==set(model.state_dict())
    assert all(value.shape==model.state_dict()[key].shape for key,value in primary.items())
    assert model.layers[0].feed_forward.w2.in_features==9728

def test_official_vae_header_covers_every_weight_and_shape():
    from forge_fibo2.official import official_classes
    cls=official_classes()['vae']
    config=cls.load_config(str(ROOT/'tests/fixtures/configs/vae'))
    state=unpack_vae(meta_state('flux2-vae-header.json'),config)
    with torch.device('meta'): model=cls.from_config(config)
    assert set(state)==set(model.state_dict())
    assert all(v.shape==model.state_dict()[k].shape for k,v in state.items())

def test_fused_quantized_rows_and_scales_are_losslessly_split():
    desc=torch.tensor(list(b'{"format":"int8_tensorwise","convrot":true,"convrot_groupsize":256}'),dtype=torch.uint8)
    source={'layers.0.attention.qkv.weight':torch.arange(48,dtype=torch.int8).reshape(12,4),
        'layers.0.attention.qkv.weight_scale':torch.arange(12).float().view(12,1),
        'layers.0.attention.qkv.comfy_quant':desc,
        'layers.0.feed_forward.w13.weight':torch.arange(64,dtype=torch.int8).reshape(16,4),
        'layers.0.feed_forward.w2.weight':torch.ones(4,8,dtype=torch.int8),
        'perceiver.layers.0.self_attention.to_kv.weight':torch.arange(32).reshape(8,4),
        'perceiver.layers.0.self_attention.to_out.weight':torch.ones(4,4),
        'layers.0.attention.q_norm.weight':torch.ones(2)}
    out=unpack_transformer(source)
    assert torch.equal(torch.cat([out[f'layers.0.attention.to_{k}.weight'] for k in ('q','k','v')]),source['layers.0.attention.qkv.weight'])
    assert torch.equal(torch.cat([out[f'layers.0.attention.to_{k}.weight_scale'] for k in ('q','k','v')]),source['layers.0.attention.qkv.weight_scale'])
    assert torch.equal(out['layers.0.attention.to_q.comfy_quant'],desc)
    assert out['layers.0.feed_forward.w2.weight'].shape==(4,8)
    assert 'perceiver.layers.0.self_attention.to_out.0.weight' in out
    assert 'layers.0.attention.norm_q.weight' in out

def test_unknown_fused_auxiliary_is_rejected():
    with pytest.raises(LayoutError,match='auxiliary'):
        unpack_transformer({'layers.0.attention.qkv.weight':torch.zeros(12,4),
                            'layers.0.attention.qkv.input_scale':torch.ones(4)})

def test_padded_ffn_preserves_dense_output_and_never_crops_columns():
    from forge_fibo2.vendor.diffusers.models.transformers.transformer_bria_fibo2 import FeedForward
    torch.manual_seed(2)
    original=FeedForward(4,3).eval()
    root=torch.nn.Module();root.ffn=FeedForward(4,3)
    state={'ffn.w1.weight':torch.nn.functional.pad(original.w1.weight,(0,0,0,5)),
           'ffn.w3.weight':torch.nn.functional.pad(original.w3.weight,(0,0,0,5)),
           'ffn.w2.weight':torch.nn.functional.pad(original.w2.weight,(0,5))}
    preserve_ffn_widths(root,state)
    root.load_state_dict(state,assign=True)
    sample=torch.randn(2,4)
    assert torch.allclose(root.ffn(sample),original(sample),atol=1e-7,rtol=1e-6)
    assert root.ffn.w2.in_features==8

def test_padded_quantized_constructor_does_not_force_real_weights_to_meta(monkeypatch):
    from forge_fibo2.vendor.diffusers.models.transformers.transformer_bria_fibo2 import FeedForward
    root=torch.nn.Module();root.ffn=FeedForward(4,3)
    class PackedLinear(torch.nn.Module):
        def __init__(self,in_features,out_features,bias=False,device=None,dtype=None):
            super().__init__();self.in_features=in_features;self.out_features=out_features
            self.factory_device=device
        def _load_from_state_dict(self,state,prefix,*args):
            self.weight=torch.nn.Parameter(state[prefix+'weight'].float().to(self.factory_device or 'cpu'),requires_grad=False)
    monkeypatch.setattr(torch.nn,'Linear',PackedLinear)
    state={'ffn.w1.weight':torch.ones(8,4,dtype=torch.int8),
           'ffn.w3.weight':torch.ones(8,4,dtype=torch.int8),
           'ffn.w2.weight':torch.ones(4,8,dtype=torch.int8)}
    preserve_ffn_widths(root,state);root.load_state_dict(state,strict=False)
    assert not any(p.is_meta for p in root.parameters())

def request(**changes):
    data=dict(init_images=[],cfg_scale=5,steps=30,width=32,height=32,prompt='a red cube')
    data.update(changes);return SimpleNamespace(**data)

def test_dev_is_true_text_to_image_and_turbo_still_requires_image():
    validate_request(request(),'dev')
    with pytest.raises(ValueError,match='source image'): validate_request(request())
    with pytest.raises(ValueError,match='30 steps'): validate_request(request(steps=4),'dev')


def test_native_img2img_default_only_masked_flag_does_not_require_a_mask():
    validate_request(request(init_images=[object()],cfg_scale=1,steps=4,
                             inpaint_full_res=True,image_mask=None),'edit-turbo')

def test_dev_initialization_does_not_create_a_noised_source(monkeypatch):
    from forge_fibo2.engine import _init
    module=ModuleType('modules');module.shared=SimpleNamespace(opts=SimpleNamespace(sd_vae_decode_method='Full'))
    monkeypatch.setitem(__import__('sys').modules,'modules',module)
    p=request(_fibo2_variant='dev')
    _init(p)
    assert p._fibo2_source is None and p._fibo2_mask is None and p.overlay_images is None

def test_official_identity_is_hash_based_not_filename(tmp_path,monkeypatch):
    path=tmp_path/'misleading-edit-turbo.safetensors';path.write_bytes(b'dev fixture')
    monkeypatch.setitem(provenance.PINS,'assets',[{'bytes':11,'sha256':hashlib.sha256(b'dev fixture').hexdigest(),'role':'transformer','variant':'dev'}])
    provenance._identify.cache_clear()
    assert provenance.identify(path,'transformer')['variant']=='dev'
    path.write_bytes(b'bad fixture')
    with pytest.raises(ValueError,match='SHA256'): provenance.identify(path,'transformer')
