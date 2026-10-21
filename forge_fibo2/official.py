"""Load Fibo 2.0 with pinned upstream classes and Forge's installed libraries.

The loader deliberately constructs every component itself.  Diffusers 0.37.1
already has the required generic VAE/scheduler infrastructure, but its public
registry predates BriaFibo2Pipeline and BriaFibo2Transformer2DModel.
"""

from __future__ import annotations

import torch
from diffusers import AutoencoderKLFlux2, FlowMatchEulerDiscreteScheduler
from transformers import Qwen3VLForConditionalGeneration, Qwen3VLProcessor

from .vendor.diffusers.models.transformers import BriaFibo2Transformer2DModel
from .vendor.diffusers.pipelines.bria_fibo2 import BriaFibo2Pipeline


MODEL_DEV = "briaai/Fibo-2.0-dev"
MODEL_EDIT_TURBO = "briaai/Fibo-2.0-edit-turbo"
SUPPORTED_MODELS = (MODEL_DEV, MODEL_EDIT_TURBO)


def official_classes() -> dict[str, type]:
    """Classes needed by Forge's native loader without touching registries."""
    return {
        "pipeline": BriaFibo2Pipeline,
        "transformer": BriaFibo2Transformer2DModel,
        "vae": AutoencoderKLFlux2,
        "scheduler": FlowMatchEulerDiscreteScheduler,
        "text_encoder": Qwen3VLForConditionalGeneration,
        "processor": Qwen3VLProcessor,
    }


def load_official_pipeline(
    model_id: str,
    *,
    dtype: torch.dtype = torch.bfloat16,
    revision: str | None = None,
    token: str | bool | None = None,
    cache_dir: str | None = None,
    local_files_only: bool = False,
) -> BriaFibo2Pipeline:
    """Build the official pipeline without modifying the installed Diffusers package.

    The caller owns device placement/offload.  This keeps loading safe for the
    full BF16 model and lets Forge choose its memory policy.
    """
    if model_id not in SUPPORTED_MODELS:
        raise ValueError(f"Unsupported official Fibo 2.0 model: {model_id}")

    common = {
        "revision": revision,
        "token": token,
        "cache_dir": cache_dir,
        "local_files_only": local_files_only,
    }
    common = {key: value for key, value in common.items() if value is not None}
    weight_args = {**common, "torch_dtype": dtype, "low_cpu_mem_usage": True}

    transformer = BriaFibo2Transformer2DModel.from_pretrained(
        model_id, subfolder="transformer", **weight_args
    )
    vae = AutoencoderKLFlux2.from_pretrained(model_id, subfolder="vae", **weight_args)
    text_encoder = Qwen3VLForConditionalGeneration.from_pretrained(
        model_id, subfolder="text_encoder", **weight_args
    )
    processor = Qwen3VLProcessor.from_pretrained(model_id, subfolder="processor", **common)
    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(model_id, subfolder="scheduler", **common)

    return BriaFibo2Pipeline(
        transformer=transformer,
        scheduler=scheduler,
        vae=vae,
        text_encoder=text_encoder,
        processor=processor,
    )


__all__ = [
    "MODEL_DEV",
    "MODEL_EDIT_TURBO",
    "SUPPORTED_MODELS",
    "BriaFibo2Pipeline",
    "BriaFibo2Transformer2DModel",
    "official_classes",
    "load_official_pipeline",
]
