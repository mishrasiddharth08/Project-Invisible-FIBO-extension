import torch
from forge_fibo2.official import official_classes

def test_vendored_official_classes_without_shared_replacement():
    import diffusers
    classes = official_classes()
    assert classes['transformer'].__module__.startswith('forge_fibo2.vendor.')
    assert classes['pipeline'].__module__.startswith('forge_fibo2.vendor.')
    assert diffusers.__name__ == 'diffusers'

def test_official_transformer_meta_architecture():
    classes = official_classes()
    with torch.device('meta'):
        transformer = classes['transformer'](dim=64, n_heads=2, n_layers=1,
            n_refiner_layers=1, cap_feat_dim=32, axes_dims=(8,12,12),
            min_num_gist_tokens=2, max_num_gist_tokens=4, injection_layer_ids=(0,))
    assert transformer.x_embedder.weight.shape == (64,128)
