"""Collecting CPU regressions must never launch the standalone CUDA smoke."""
from pathlib import Path
import runpy
import torch

def test_cuda_smoke_is_import_safe(monkeypatch):
    def forbidden(*args,**kwargs):
        raise AssertionError('GPU initialization during CPU test collection')
    monkeypatch.setattr(torch.cuda,'_lazy_init',forbidden)
    namespace=runpy.run_path(str(Path(__file__).with_name('test_tiny_pipeline.py')),
                             run_name='import_only_cuda_smoke')
    assert callable(namespace['main'])
