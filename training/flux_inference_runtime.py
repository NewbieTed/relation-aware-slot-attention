"""Runtime helpers for FLUX + SeeThrough3D inference.

This module contains the small amount of SeeThrough3D integration needed at
generation time: loading the vendored FLUX pipeline, installing condition LoRA
processors, configuring optional bitsandbytes quantization, and keeping
quantized pipeline execution devices consistent. It intentionally has no
fine-tuning loop.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path
from typing import Any

import torch


def import_seethrough3d_flux() -> tuple[Any, Any, Any, Any, Any]:
    """Load SeeThrough3D's FLUX fork and condition-LoRA processors."""

    repo_root = Path(__file__).resolve().parents[1]
    seethrough_root = repo_root / "external" / "seethrough3d"
    if not seethrough_root.exists():
        raise FileNotFoundError(
            "Missing external/seethrough3d. Run scripts/setup/setup_seethrough3d_checkout.sh "
            "before FLUX relation inference."
        )
    sys.path.insert(0, str(seethrough_root))
    if "einops" not in sys.modules:
        try:
            __import__("einops")
        except ModuleNotFoundError:
            einops_stub = types.ModuleType("einops")

            def _missing_rearrange(*_args: Any, **_kwargs: Any) -> None:
                raise ModuleNotFoundError("einops is required for rearrange")

            einops_stub.rearrange = _missing_rearrange
            sys.modules["einops"] = einops_stub

    from diffusers.models.attention_processor import FluxAttnProcessor2_0
    from train.src.layers import MultiDoubleStreamBlockLoraProcessor, MultiSingleStreamBlockLoraProcessor
    from train.src.pipeline import FluxPipeline
    from train.src.transformer_flux import FluxTransformer2DModel

    return (
        FluxPipeline,
        FluxTransformer2DModel,
        MultiDoubleStreamBlockLoraProcessor,
        MultiSingleStreamBlockLoraProcessor,
        FluxAttnProcessor2_0,
    )


def install_condition_lora_processors(
    *,
    transformer: Any,
    rank: int,
    alpha: float,
    cond_size: int,
    device: str,
    dtype: torch.dtype,
    double_processor_cls: Any,
    single_processor_cls: Any,
    base_processor_cls: Any,
) -> list[str]:
    """Attach SeeThrough3D condition LoRA processors to FLUX attention blocks."""

    processors = {}
    installed: list[str] = []
    for name, attn_processor in transformer.attn_processors.items():
        layer_index = None
        for part in name.split("."):
            if part.isdigit():
                layer_index = int(part)
                break

        if name.startswith("transformer_blocks") and layer_index is not None:
            processors[name] = double_processor_cls(
                dim=transformer.inner_dim,
                ranks=[rank],
                network_alphas=[alpha],
                lora_weights=[1.0],
                device=device,
                dtype=dtype,
                cond_width=cond_size,
                cond_height=cond_size,
                n_loras=1,
            )
            installed.append(name)
        elif name.startswith("single_transformer_blocks") and layer_index is not None:
            processors[name] = single_processor_cls(
                dim=transformer.inner_dim,
                ranks=[rank],
                network_alphas=[alpha],
                lora_weights=[1.0],
                device=device,
                dtype=dtype,
                cond_width=cond_size,
                cond_height=cond_size,
                n_loras=1,
            )
            installed.append(name)
        else:
            processors[name] = attn_processor if attn_processor is not None else base_processor_cls()

    transformer.set_attn_processor(processors)
    transformer.requires_grad_(False)
    return installed


def build_flux_quantization_config(mode: str, dtype: torch.dtype) -> Any | None:
    """Create a bitsandbytes config for frozen FLUX inference weights."""

    if mode == "none":
        return None
    try:
        from diffusers import BitsAndBytesConfig
    except ImportError as exc:
        raise ImportError("FLUX quantization requires a recent diffusers with BitsAndBytesConfig.") from exc
    try:
        __import__("bitsandbytes")
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("FLUX quantization requires bitsandbytes in .venv-flux.") from exc

    if mode == "8bit":
        return BitsAndBytesConfig(load_in_8bit=True)
    compute_dtype = torch.bfloat16 if dtype == torch.bfloat16 else torch.float16
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=True,
    )


def set_pipeline_execution_device(pipeline: Any, device: str) -> None:
    """Force SeeThrough3D FLUX to prepare latents on the transformer device."""

    forced_device = torch.device(device)
    object.__setattr__(pipeline, "_forced_execution_device", forced_device)
    if getattr(pipeline.__class__, "_relation_forced_execution_device", False):
        return

    base_cls = pipeline.__class__

    class ForcedExecutionDevicePipeline(base_cls):  # type: ignore[misc, valid-type]
        _relation_forced_execution_device = True

        @property
        def _execution_device(self) -> torch.device:  # type: ignore[override]
            forced = getattr(self, "_forced_execution_device", None)
            if forced is not None:
                return torch.device(forced)
            return super()._execution_device

    object.__setattr__(pipeline, "__class__", ForcedExecutionDevicePipeline)


def pipeline_execution_device(pipeline: Any, fallback: str) -> torch.device:
    """Return the device SeeThrough3D will use for sampling latents."""

    execution_device = getattr(pipeline, "_execution_device", None)
    if execution_device is None:
        return torch.device(fallback)
    return torch.device(execution_device)


def text_encoder_device(pipeline: Any) -> torch.device:
    """Return the actual text encoder device under low-VRAM inference."""

    try:
        return next(pipeline.text_encoder.parameters()).device
    except StopIteration:
        return torch.device("cpu")
