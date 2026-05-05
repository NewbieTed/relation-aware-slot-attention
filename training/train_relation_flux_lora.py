"""Train FLUX.1-dev LoRA with GNN-predicted 3D box OSCR conditions.

This is the FLUX replacement path for the earlier SD1.5 experiments. The
training loop keeps the base FLUX model frozen, renders GNN-predicted 3D boxes
into lightweight OSCR images, VAE-encodes those OSCR images into packed latent
tokens, and feeds them through the SeeThrough3D-style FLUX condition stream.

Only rank-128 LoRA adapters on FLUX self-attention projections are trainable by
default. The GNN, text encoders, VAE, and base FLUX transformer remain frozen.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import types
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from .dataset import build_dataset_splits, collate_training_items
from .graph_modules import GraphSlotEncoder, build_slot_conditioning
from .graph_targets import bbox_centers_after_crop
from .metrics import MetricsLogger, write_split_manifest
from .oscr_renderer import render_oscr_boxes
from .runtime import (
    choose_weight_dtype,
    is_tqdm_disabled,
    normalize_graph_encoder_state_dict,
    resolve_torch_device,
    set_seed,
)
from .scene_graph import build_batched_scene_graphs
from .seethrough_condition import (
    build_binding_prompt,
    call_ids_from_binding_prompt,
    render_seethrough_oscr_and_masks,
)

DEFAULT_FLUX_MODEL_ID = "black-forest-labs/FLUX.1-dev"


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train relation-aware FLUX.1-dev LoRA with OSCR condition latents.")
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-id", type=str, default=DEFAULT_FLUX_MODEL_ID)
    parser.add_argument("--init-graph-encoder", type=Path, required=True)
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    parser.add_argument("--mixed-precision", choices=("no", "fp16", "bf16"), default="bf16")
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--oscr-size", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--max-train-steps", type=int, default=24000)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--slot-dim", type=int, default=512)
    parser.add_argument("--gnn-layers", type=int, default=2)
    parser.add_argument("--lora-rank", type=int, default=128)
    parser.add_argument("--lora-alpha", type=float, default=128.0)
    parser.add_argument(
        "--flux-quantization",
        choices=("none", "8bit", "4bit"),
        default="none",
        help="Optionally load the frozen FLUX transformer with bitsandbytes quantization.",
    )
    parser.add_argument("--max-sequence-length", type=int, default=512)
    parser.add_argument("--guidance-scale", type=float, default=3.5)
    parser.add_argument(
        "--low-vram",
        action="store_true",
        help="Keep frozen text encoders/GNN on CPU and offload the VAE between encodes.",
    )
    parser.add_argument(
        "--gradient-checkpointing",
        action="store_true",
        help="Enable transformer gradient checkpointing. Disabled by default because SeeThrough3D checkpointing currently does not support condition kwargs on some versions.",
    )
    parser.add_argument("--prompt-prefix", type=str, default="a photo of")
    parser.add_argument("--condition-renderer", choices=("seethrough", "legacy"), default="seethrough")
    parser.add_argument("--oscr-face-alpha", type=float, default=0.10)
    parser.add_argument("--oscr-azimuth-degrees", type=float, default=180.0)
    parser.add_argument("--limit-rows", type=int, default=None)
    parser.add_argument("--eval-fraction", type=float, default=0.1)
    parser.add_argument("--test-fraction", type=float, default=0.1)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--eval-every", type=int, default=1000)
    parser.add_argument("--eval-sample-count", type=int, default=0)
    parser.add_argument("--eval-inference-steps", type=int, default=12)
    parser.add_argument("--eval-sample-seed", type=int, default=1234)
    parser.add_argument("--save-every", type=int, default=4000)
    parser.add_argument("--disable-tqdm", action="store_true")
    return parser


def _save_state(output_dir: Path, *, step: int, args: argparse.Namespace, lora_layers: list[str]) -> None:
    payload = {
        "step": step,
        "dataset_dir": str(args.dataset_dir),
        "model_id": args.model_id,
        "init_graph_encoder": str(args.init_graph_encoder),
        "lora_rank": args.lora_rank,
        "lora_alpha": args.lora_alpha,
        "flux_quantization": args.flux_quantization,
        "lora_layers": lora_layers,
        "oscr_size": args.oscr_size,
        "condition_renderer": args.condition_renderer,
        "oscr_face_alpha": args.oscr_face_alpha,
        "oscr_azimuth_degrees": args.oscr_azimuth_degrees,
    }
    (output_dir / "training_state.json").write_text(json.dumps(payload, indent=2))


def _import_seethrough3d_flux() -> tuple[Any, Any, Any, Any, Any]:
    """Load the FLUX fork and LoRA attention processors from external/seethrough3d."""

    repo_root = Path(__file__).resolve().parents[1]
    seethrough_root = repo_root / "external" / "seethrough3d"
    if not seethrough_root.exists():
        raise FileNotFoundError(
            "Missing external/seethrough3d. Clone https://github.com/va1bhavagrawal/seethrough3d "
            "there before running this FLUX condition-stream trainer."
        )
    sys.path.insert(0, str(seethrough_root))
    if "einops" not in sys.modules:
        try:
            __import__("einops")
        except ModuleNotFoundError:
            # The reference layers import rearrange but do not use it in the
            # LoRA processors we need. Keep the integration lightweight when
            # einops is not installed in the current environment.
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


def _install_condition_lora_processors(
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
    """Attach SeeThrough3D LoRA processors to every FLUX self-attention block."""

    processors = {}
    installed: list[str] = []
    for name, attn_processor in transformer.attn_processors.items():
        layer_index = None
        parts = name.split(".")
        for part in parts:
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
    for name, parameter in transformer.named_parameters():
        parameter.requires_grad = "_loras" in name
    return installed


def _save_condition_lora_state(transformer: Any, path: Path) -> None:
    state = {
        key: value.detach().cpu()
        for key, value in transformer.state_dict().items()
        if "_loras" in key
    }
    torch.save(state, path)


def _enable_gradient_checkpointing_compat(transformer: Any) -> None:
    """Enable checkpointing across diffusers and SeeThrough3D API variants."""

    if hasattr(transformer, "enable_gradient_checkpointing"):
        try:
            transformer.enable_gradient_checkpointing()
            return
        except TypeError:
            pass

    if hasattr(transformer, "_set_gradient_checkpointing"):
        for module in transformer.modules():
            transformer._set_gradient_checkpointing(module, enable=True)
        return

    if hasattr(transformer, "gradient_checkpointing"):
        transformer.gradient_checkpointing = True


def _build_flux_quantization_config(mode: str, dtype: torch.dtype) -> Any | None:
    """Create a bitsandbytes config for quantized frozen FLUX weights."""

    if mode == "none":
        return None
    try:
        from diffusers import BitsAndBytesConfig
    except ImportError as exc:
        raise ImportError(
            "FLUX quantization requires a recent diffusers with BitsAndBytesConfig."
        ) from exc
    try:
        __import__("bitsandbytes")
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "FLUX quantization requires bitsandbytes. Install with "
            "`python -m pip install bitsandbytes` in .venv-flux."
        ) from exc

    if mode == "8bit":
        return BitsAndBytesConfig(load_in_8bit=True)
    compute_dtype = torch.bfloat16 if dtype == torch.bfloat16 else torch.float16
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=True,
    )


def _set_pipeline_execution_device(pipeline: Any, device: str) -> None:
    """Keep quantized FLUX inference from preparing CPU latents for a CUDA transformer."""

    try:
        object.__setattr__(pipeline, "_execution_device", torch.device(device))
    except Exception:
        pipeline._execution_device = torch.device(device)


def _load_graph_encoder(
    *,
    path: Path,
    text_hidden_dim: int,
    slot_dim: int,
    gnn_layers: int,
    device: str,
) -> GraphSlotEncoder:
    graph_encoder = GraphSlotEncoder(
        text_hidden_dim=text_hidden_dim,
        slot_dim=slot_dim,
        num_layers=gnn_layers,
    ).to(device)
    state_dict = normalize_graph_encoder_state_dict(torch.load(path, map_location=device))
    incompatible = graph_encoder.load_state_dict(state_dict, strict=False)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        print(
            "Loaded graph encoder with "
            f"{len(incompatible.missing_keys)} missing and "
            f"{len(incompatible.unexpected_keys)} unexpected keys."
        )
    graph_encoder.requires_grad_(False)
    graph_encoder.eval()
    return graph_encoder


@torch.no_grad()
def _graph_3d_boxes(
    *,
    batch: dict[str, Any],
    pipeline: Any,
    graph_encoder: GraphSlotEncoder,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    graph_device = getattr(pipeline, "_graph_device", device)
    max_nodes = max(len(graph["nodes"]) for graph in batch["scene_graphs"])
    centers, slot_mask = bbox_centers_after_crop(
        batch["metadata"],
        batch["image_sizes"],
        max_nodes=max_nodes,
        device=torch.device(graph_device),
    )
    scene_graph_batch = build_batched_scene_graphs(
        batch["scene_graphs"],
        slot_targets=centers,
        slot_mask=slot_mask,
    )
    conditioning = build_slot_conditioning(
        tokenizer=pipeline.tokenizer,
        text_encoder=pipeline.text_encoder,
        scene_graph_batch=scene_graph_batch,
        graph_encoder=graph_encoder,
        device=graph_device,
    )
    return (
        conditioning.slot_positions.to(device),
        conditioning.slot_log_sizes_3d.to(device),
        conditioning.slot_mask.to(device),
    )


def _encode_packed_latents(
    *,
    pipeline: Any,
    images: torch.Tensor,
    image_size: int,
    device: str,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor, tuple[int, int]]:
    vae_device = next(pipeline.vae.parameters()).device
    if str(vae_device) != device:
        pipeline.vae.to(device=device, dtype=dtype)
    latents = pipeline.vae.encode(images.to(device=device, dtype=dtype)).latent_dist.sample()
    if getattr(pipeline, "_low_vram", False):
        pipeline.vae.to("cpu")
        if device == "cuda":
            torch.cuda.empty_cache()
    latents = (latents - pipeline.vae.config.shift_factor) * pipeline.vae.config.scaling_factor
    batch_size, channels, height, width = latents.shape
    packed = pipeline._pack_latents(latents, batch_size, channels, height, width)
    ids = pipeline._prepare_latent_image_ids(
        batch_size,
        height,
        width,
        torch.device(device),
        dtype,
    )
    return packed, ids, (height // 2, width // 2)


def _resize_condition_ids(
    *,
    cond_grid: tuple[int, int],
    image_grid: tuple[int, int],
    device: str,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Create FLUX image ids for condition tokens in the same coordinate frame as image tokens."""

    cond_h, cond_w = cond_grid
    image_h, image_w = image_grid
    ids = torch.zeros(cond_h, cond_w, 3, device=device, dtype=dtype)
    scale_h = image_h / max(cond_h, 1)
    scale_w = image_w / max(cond_w, 1)
    ids[..., 1] = torch.arange(cond_h, device=device, dtype=dtype)[:, None] * scale_h
    ids[..., 2] = torch.arange(cond_w, device=device, dtype=dtype)[None, :] * scale_w
    return ids.reshape(cond_h * cond_w, 3)


def _build_binding_inputs(
    *,
    batch: dict[str, Any],
    pipeline: Any,
    slot_mask: torch.Tensor,
    cuboid_masks: torch.Tensor,
    max_sequence_length: int,
    device: str,
    prompt_prefix: str,
) -> tuple[list[str], list[list[torch.Tensor]], torch.Tensor]:
    """Build SeeThrough3D-style prompts, call ids, and cuboid masks."""

    binding_prompts: list[str] = []
    batch_call_ids: list[list[torch.Tensor]] = []
    for batch_index, (prompt, scene_graph) in enumerate(zip(batch["prompts"], batch["scene_graphs"])):
        binding_prompt = build_binding_prompt(
            original_prompt=str(prompt),
            scene_graph=scene_graph,
            prefix=prompt_prefix,
        )
        binding_prompts.append(binding_prompt.prompt)
        span_call_ids = call_ids_from_binding_prompt(
            tokenizer=pipeline.tokenizer_2,
            binding_prompt=binding_prompt,
            max_sequence_length=max_sequence_length,
            device=device,
        )
        sample_call_ids: list[torch.Tensor] = []
        for node_index, token_ids in enumerate(span_call_ids):
            if node_index >= slot_mask.shape[1] or not bool(slot_mask[batch_index, node_index].item()):
                continue
            sample_call_ids.append(token_ids)
        batch_call_ids.append(sample_call_ids)
    return binding_prompts, batch_call_ids, cuboid_masks.to(device=device, dtype=torch.uint8)


@torch.no_grad()
def _build_condition_latents(
    *,
    batch: dict[str, Any],
    pipeline: Any,
    graph_encoder: GraphSlotEncoder,
    device: str,
    dtype: torch.dtype,
    oscr_size: int,
    condition_renderer: str,
    oscr_face_alpha: float,
    oscr_azimuth_degrees: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    centers, log_sizes, slot_mask = _graph_3d_boxes(
        batch=batch,
        pipeline=pipeline,
        graph_encoder=graph_encoder,
        device=device,
    )
    cond_token_grid = (oscr_size // 16, oscr_size // 16)
    if condition_renderer == "seethrough":
        oscr, cuboid_masks = render_seethrough_oscr_and_masks(
            centers=centers,
            log_sizes=log_sizes,
            slot_mask=slot_mask,
            image_size=oscr_size,
            mask_size=cond_token_grid,
            face_alpha=oscr_face_alpha,
            azimuth_degrees=oscr_azimuth_degrees,
        )
    else:
        oscr = render_oscr_boxes(
            centers=centers,
            log_sizes=log_sizes,
            slot_mask=slot_mask,
            image_size=oscr_size,
        )
        _, cuboid_masks = render_seethrough_oscr_and_masks(
            centers=centers,
            log_sizes=log_sizes,
            slot_mask=slot_mask,
            image_size=oscr_size,
            mask_size=cond_token_grid,
            face_alpha=oscr_face_alpha,
            azimuth_degrees=oscr_azimuth_degrees,
        )
    cond_latents, cond_ids, cond_grid = _encode_packed_latents(
        pipeline=pipeline,
        images=oscr,
        image_size=oscr_size,
        device=device,
        dtype=dtype,
    )
    return cond_latents, cond_ids, torch.tensor(cond_grid, device=device), centers, log_sizes, slot_mask, cuboid_masks


def _compute_loss(
    *,
    batch: dict[str, Any],
    pipeline: Any,
    graph_encoder: GraphSlotEncoder,
    device: str,
    dtype: torch.dtype,
    image_size: int,
    oscr_size: int,
    guidance_scale: float,
    max_sequence_length: int,
    condition_renderer: str,
    oscr_face_alpha: float,
    oscr_azimuth_degrees: float,
    prompt_prefix: str,
) -> dict[str, torch.Tensor]:
    pixel_values = batch["pixel_values"].to(device=device, dtype=dtype)
    with torch.no_grad():
        clean_latents, image_ids, image_grid = _encode_packed_latents(
            pipeline=pipeline,
            images=pixel_values,
            image_size=image_size,
            device=device,
            dtype=dtype,
        )
        encoder_device = getattr(pipeline, "_encoder_device", device)
        cond_latents, _cond_ids, cond_grid_tensor, centers, log_sizes, slot_mask, cuboid_masks = _build_condition_latents(
            batch=batch,
            pipeline=pipeline,
            graph_encoder=graph_encoder,
            device=device,
            dtype=dtype,
            oscr_size=oscr_size,
            condition_renderer=condition_renderer,
            oscr_face_alpha=oscr_face_alpha,
            oscr_azimuth_degrees=oscr_azimuth_degrees,
        )
        binding_prompts, call_ids, cuboids_segmasks = _build_binding_inputs(
            batch=batch,
            pipeline=pipeline,
            slot_mask=slot_mask,
            cuboid_masks=cuboid_masks,
            max_sequence_length=max_sequence_length,
            device=device,
            prompt_prefix=prompt_prefix,
        )
        prompt_embeds, pooled_prompt_embeds, text_ids = pipeline.encode_prompt(
            prompt=binding_prompts,
            prompt_2=binding_prompts,
            device=torch.device(encoder_device),
            num_images_per_prompt=1,
            max_sequence_length=max_sequence_length,
        )
        prompt_embeds = prompt_embeds.to(device=device, dtype=dtype)
        pooled_prompt_embeds = pooled_prompt_embeds.to(device=device, dtype=dtype)
        text_ids = text_ids.to(device=device)
        cond_grid = (int(cond_grid_tensor[0].item()), int(cond_grid_tensor[1].item()))
        cond_ids = _resize_condition_ids(
            cond_grid=cond_grid,
            image_grid=image_grid,
            device=device,
            dtype=image_ids.dtype,
        )
        image_and_condition_ids = torch.cat([image_ids, cond_ids], dim=0)

    noise = torch.randn_like(clean_latents)
    sigmas = torch.rand(clean_latents.shape[0], device=clean_latents.device, dtype=clean_latents.dtype)
    noisy_latents = (1.0 - sigmas[:, None, None]) * clean_latents + sigmas[:, None, None] * noise
    target = noise - clean_latents
    guidance = None
    if pipeline.transformer.config.guidance_embeds:
        guidance = torch.full(
            (clean_latents.shape[0],),
            guidance_scale,
            device=clean_latents.device,
            dtype=torch.float32,
        )
    model_pred = pipeline.transformer(
        hidden_states=noisy_latents,
        cond_hidden_states=cond_latents,
        timestep=sigmas,
        guidance=guidance,
        pooled_projections=pooled_prompt_embeds,
        encoder_hidden_states=prompt_embeds,
        txt_ids=text_ids,
        img_ids=image_and_condition_ids,
        return_dict=False,
        call_ids=call_ids,
        cuboids_segmasks=cuboids_segmasks,
    )[0]
    loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
    return {
        "loss": loss,
        "flow_loss": loss.detach(),
        "condition_latent_norm": cond_latents.float().pow(2).mean().sqrt().detach(),
        "pred_center_abs_mean": centers.float().abs().mean().detach(),
        "pred_log_size_mean": log_sizes.float().mean().detach(),
        "binding_token_count": torch.tensor(
            sum(len(token_ids) for sample in call_ids for token_ids in sample),
            device=clean_latents.device,
            dtype=torch.float32,
        ),
        "binding_mask_pct": cuboids_segmasks.float().mean().mul(100.0).detach(),
    }


def _tensor_to_pil(image: torch.Tensor) -> Any:
    from PIL import Image

    array = image.detach().float().add(1.0).mul(127.5).clamp(0, 255)
    array = array.permute(1, 2, 0).to(torch.uint8).cpu().numpy()
    return Image.fromarray(array)


@torch.no_grad()
def _run_eval_samples(
    *,
    step: int,
    eval_dataset: Any,
    pipeline: Any,
    graph_encoder: GraphSlotEncoder,
    output_dir: Path,
    device: str,
    dtype: torch.dtype,
    image_size: int,
    oscr_size: int,
    guidance_scale: float,
    max_sequence_length: int,
    condition_renderer: str,
    oscr_face_alpha: float,
    oscr_azimuth_degrees: float,
    prompt_prefix: str,
    sample_count: int,
    inference_steps: int,
    seed: int,
) -> None:
    """Generate small qualitative samples using the already-loaded pipeline."""

    if sample_count <= 0 or len(eval_dataset) == 0:
        return
    was_training = pipeline.transformer.training
    pipeline.transformer.eval()
    sample_dir = output_dir / "eval_samples" / f"step-{step:06d}"
    sample_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []

    for sample_index in range(min(sample_count, len(eval_dataset))):
        item = eval_dataset[sample_index]
        graph_device = getattr(pipeline, "_graph_device", device)
        slot_targets = torch.zeros(
            1,
            len(item.scene_graph["nodes"]),
            3,
            device=torch.device(graph_device),
        )
        slot_mask = torch.ones(
            1,
            len(item.scene_graph["nodes"]),
            device=torch.device(graph_device),
            dtype=torch.bool,
        )
        scene_graph_batch = build_batched_scene_graphs(
            [item.scene_graph],
            slot_targets=slot_targets,
            slot_mask=slot_mask,
        )
        conditioning = build_slot_conditioning(
            tokenizer=pipeline.tokenizer,
            text_encoder=pipeline.text_encoder,
            scene_graph_batch=scene_graph_batch,
            graph_encoder=graph_encoder,
            device=graph_device,
        )
        centers = conditioning.slot_positions.to(device)
        log_sizes = conditioning.slot_log_sizes_3d.to(device)
        slot_mask = conditioning.slot_mask.to(device)
        cond_grid = (oscr_size // 16, oscr_size // 16)
        if condition_renderer == "seethrough":
            oscr, cuboids_segmasks = render_seethrough_oscr_and_masks(
                centers=centers,
                log_sizes=log_sizes,
                slot_mask=slot_mask,
                image_size=oscr_size,
                mask_size=cond_grid,
                face_alpha=oscr_face_alpha,
                azimuth_degrees=oscr_azimuth_degrees,
            )
            oscr_viz, _ = render_seethrough_oscr_and_masks(
                centers=centers,
                log_sizes=log_sizes,
                slot_mask=slot_mask,
                image_size=oscr_size,
                mask_size=cond_grid,
                face_alpha=max(oscr_face_alpha, 0.25),
                azimuth_degrees=oscr_azimuth_degrees,
            )
        else:
            oscr = render_oscr_boxes(
                centers=centers,
                log_sizes=log_sizes,
                slot_mask=slot_mask,
                image_size=oscr_size,
            )
            _, cuboids_segmasks = render_seethrough_oscr_and_masks(
                centers=centers,
                log_sizes=log_sizes,
                slot_mask=slot_mask,
                image_size=oscr_size,
                mask_size=cond_grid,
                face_alpha=oscr_face_alpha,
                azimuth_degrees=oscr_azimuth_degrees,
            )
            oscr_viz, _ = render_seethrough_oscr_and_masks(
                centers=centers,
                log_sizes=log_sizes,
                slot_mask=slot_mask,
                image_size=oscr_size,
                mask_size=cond_grid,
                face_alpha=0.25,
                azimuth_degrees=oscr_azimuth_degrees,
            )
        binding_prompt = build_binding_prompt(
            original_prompt=item.prompt,
            scene_graph=item.scene_graph,
            prefix=prompt_prefix,
        )
        call_ids = [
            call_ids_from_binding_prompt(
                tokenizer=pipeline.tokenizer_2,
                binding_prompt=binding_prompt,
                max_sequence_length=max_sequence_length,
                device=device,
            )
        ]
        generator_device = getattr(pipeline, "_execution_device", torch.device(device))
        generator_device = torch.device(generator_device)
        generator = (
            torch.Generator(device=generator_device).manual_seed(seed + step * 1000 + sample_index)
            if generator_device.type != "mps"
            else None
        )
        image = pipeline(
            prompt=binding_prompt.prompt,
            prompt_2=binding_prompt.prompt,
            height=image_size,
            width=image_size,
            num_inference_steps=inference_steps,
            guidance_scale=guidance_scale,
            generator=generator,
            max_sequence_length=max_sequence_length,
            spatial_images=[_tensor_to_pil(oscr[0])],
            subject_images=[],
            cond_size=oscr_size,
            call_ids=call_ids,
            cuboids_segmasks=cuboids_segmasks.to(device=device, dtype=torch.uint8),
        ).images[0]
        image_path = sample_dir / f"sample_{sample_index:02d}.png"
        oscr_path = sample_dir / f"sample_{sample_index:02d}_oscr.png"
        oscr_viz_path = sample_dir / f"sample_{sample_index:02d}_oscr_viz.png"
        image.save(image_path)
        _tensor_to_pil(oscr[0]).save(oscr_path)
        _tensor_to_pil(oscr_viz[0]).save(oscr_viz_path)
        records.append(
            {
                "prompt": item.prompt,
                "binding_prompt": binding_prompt.prompt,
                "image": str(image_path),
                "oscr": str(oscr_path),
                "oscr_viz": str(oscr_viz_path),
                "predicted_centers": centers[0].detach().cpu().tolist(),
                "predicted_sizes_3d": log_sizes[0].exp().detach().cpu().tolist(),
                "binding_mask_pct": float(cuboids_segmasks.float().mean().mul(100.0).item()),
            }
        )
    (sample_dir / "samples.json").write_text(json.dumps(records, indent=2))
    if was_training:
        pipeline.transformer.train()


def _save_checkpoint(
    *,
    output_dir: Path,
    step: int,
    pipeline: Any,
    graph_encoder: GraphSlotEncoder,
    optimizer: torch.optim.Optimizer,
) -> Path:
    checkpoint_dir = output_dir / f"checkpoint-{step:06d}"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    _save_condition_lora_state(pipeline.transformer, checkpoint_dir / "flux_lora.pt")
    torch.save(graph_encoder.state_dict(), checkpoint_dir / "graph_encoder.pt")
    torch.save(optimizer.state_dict(), checkpoint_dir / "optimizer.pt")
    return checkpoint_dir


def main() -> int:
    args = make_parser().parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    device = resolve_torch_device(args.device)
    dtype = choose_weight_dtype(device, args.mixed_precision)
    set_seed(args.seed)

    datasets = build_dataset_splits(
        args.dataset_dir,
        image_size=args.image_size,
        prompt_prefix=args.prompt_prefix,
        limit_rows=args.limit_rows,
        seed=args.seed,
        eval_fraction=args.eval_fraction,
        test_fraction=args.test_fraction,
    )
    write_split_manifest(
        args.output_dir,
        train_rows=datasets["train"].rows,
        eval_rows=datasets["eval"].rows,
        test_rows=datasets["test"].rows,
        seed=args.seed,
        eval_fraction=args.eval_fraction,
        test_fraction=args.test_fraction,
    )
    train_dataloader = DataLoader(
        datasets["train"],
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_training_items,
    )

    (
        FluxPipeline,
        FluxTransformer2DModel,
        MultiDoubleStreamBlockLoraProcessor,
        MultiSingleStreamBlockLoraProcessor,
        FluxAttnProcessor2_0,
    ) = _import_seethrough3d_flux()

    pipeline = FluxPipeline.from_pretrained(args.model_id, transformer=None, torch_dtype=dtype)
    quantization_config = _build_flux_quantization_config(args.flux_quantization, dtype)
    transformer_kwargs: dict[str, Any] = {
        "subfolder": "transformer",
        "torch_dtype": dtype,
    }
    if quantization_config is not None:
        transformer_kwargs["quantization_config"] = quantization_config
        transformer_kwargs["device_map"] = {"": device}
    pipeline.transformer = FluxTransformer2DModel.from_pretrained(
        args.model_id,
        **transformer_kwargs,
    )
    pipeline._low_vram = args.low_vram
    pipeline._encoder_device = "cpu" if args.low_vram else device
    pipeline._graph_device = "cpu" if args.low_vram else device
    if args.low_vram:
        pipeline.text_encoder.to("cpu")
        pipeline.text_encoder_2.to("cpu")
        pipeline.vae.to("cpu")
        if quantization_config is None:
            pipeline.transformer.to(device=device, dtype=dtype)
    else:
        if quantization_config is None:
            pipeline.to(device)
        else:
            pipeline.vae.to(device=device, dtype=dtype)
            pipeline.text_encoder.to(device)
            pipeline.text_encoder_2.to(device)
    _set_pipeline_execution_device(pipeline, device)
    if args.gradient_checkpointing:
        _enable_gradient_checkpointing_compat(pipeline.transformer)
    pipeline.vae.requires_grad_(False)
    pipeline.text_encoder.requires_grad_(False)
    pipeline.text_encoder_2.requires_grad_(False)
    pipeline.transformer.requires_grad_(False)

    lora_layers = _install_condition_lora_processors(
        transformer=pipeline.transformer,
        rank=args.lora_rank,
        alpha=args.lora_alpha,
        cond_size=args.oscr_size,
        device=device,
        dtype=dtype,
        double_processor_cls=MultiDoubleStreamBlockLoraProcessor,
        single_processor_cls=MultiSingleStreamBlockLoraProcessor,
        base_processor_cls=FluxAttnProcessor2_0,
    )
    print(f"Installed SeeThrough3D condition LoRA processors on {len(lora_layers)} FLUX attention blocks.")

    graph_encoder = _load_graph_encoder(
        path=args.init_graph_encoder,
        text_hidden_dim=pipeline.text_encoder.config.hidden_size,
        slot_dim=args.slot_dim,
        gnn_layers=args.gnn_layers,
        device=pipeline._graph_device,
    )
    trainable_params = [p for p in pipeline.transformer.parameters() if p.requires_grad]
    print(f"Trainable FLUX LoRA parameters: {sum(p.numel() for p in trainable_params) / 1_000_000:.2f}M")
    optimizer = torch.optim.AdamW(trainable_params, lr=args.learning_rate)
    scaler = torch.amp.GradScaler("cuda", enabled=device == "cuda" and args.mixed_precision == "fp16")
    metrics_logger = MetricsLogger(
        args.output_dir,
        fieldnames=[
            "step",
            "split",
            "loss",
            "flow_loss",
            "condition_latent_norm",
            "pred_center_abs_mean",
            "pred_log_size_mean",
            "binding_token_count",
            "binding_mask_pct",
        ],
    )
    _save_state(args.output_dir, step=0, args=args, lora_layers=lora_layers)

    progress = tqdm(total=args.max_train_steps, disable=is_tqdm_disabled(args), desc="RelationFluxLoRA")
    global_step = 0
    micro_step = 0
    running = {
        "loss": 0.0,
        "flow_loss": 0.0,
        "condition_latent_norm": 0.0,
        "pred_center_abs_mean": 0.0,
        "pred_log_size_mean": 0.0,
        "binding_token_count": 0.0,
        "binding_mask_pct": 0.0,
    }
    running_steps = 0
    optimizer.zero_grad(set_to_none=True)
    while global_step < args.max_train_steps:
        for batch in train_dataloader:
            metrics = _compute_loss(
                batch=batch,
                pipeline=pipeline,
                graph_encoder=graph_encoder,
                device=device,
                dtype=dtype,
                image_size=args.image_size,
                oscr_size=args.oscr_size,
                guidance_scale=args.guidance_scale,
                max_sequence_length=args.max_sequence_length,
                condition_renderer=args.condition_renderer,
                oscr_face_alpha=args.oscr_face_alpha,
                oscr_azimuth_degrees=args.oscr_azimuth_degrees,
                prompt_prefix=args.prompt_prefix,
            )
            loss = metrics["loss"] / args.gradient_accumulation_steps
            if scaler.is_enabled():
                scaler.scale(loss).backward()
            else:
                loss.backward()
            micro_step += 1
            running_steps += 1
            for key in running:
                running[key] += float(metrics[key].item())
            if micro_step % args.gradient_accumulation_steps == 0:
                if scaler.is_enabled():
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
                progress.update(1)
                if global_step % args.log_every == 0:
                    train_log = {
                        "step": global_step,
                        "split": "train",
                        **{key: value / running_steps for key, value in running.items()},
                    }
                    metrics_logger.log(train_log)
                    progress.set_postfix(loss=f"{train_log['loss']:.4f}")
                    running = {key: 0.0 for key in running}
                    running_steps = 0
                if global_step % args.save_every == 0:
                    checkpoint = _save_checkpoint(
                        output_dir=args.output_dir,
                        step=global_step,
                        pipeline=pipeline,
                        graph_encoder=graph_encoder,
                        optimizer=optimizer,
                    )
                    _save_state(args.output_dir, step=global_step, args=args, lora_layers=lora_layers)
                    print(f"Saved FLUX LoRA checkpoint to {checkpoint}")
                if args.eval_sample_count > 0 and global_step % args.eval_every == 0:
                    _run_eval_samples(
                        step=global_step,
                        eval_dataset=datasets["eval"],
                        pipeline=pipeline,
                        graph_encoder=graph_encoder,
                        output_dir=args.output_dir,
                        device=device,
                        dtype=dtype,
                        image_size=args.image_size,
                        oscr_size=args.oscr_size,
                        guidance_scale=args.guidance_scale,
                        max_sequence_length=args.max_sequence_length,
                        condition_renderer=args.condition_renderer,
                        oscr_face_alpha=args.oscr_face_alpha,
                        oscr_azimuth_degrees=args.oscr_azimuth_degrees,
                        prompt_prefix=args.prompt_prefix,
                        sample_count=args.eval_sample_count,
                        inference_steps=args.eval_inference_steps,
                        seed=args.eval_sample_seed,
                    )
                    print(
                        "Saved eval samples to "
                        f"{args.output_dir / 'eval_samples' / f'step-{global_step:06d}'}"
                    )
                if global_step >= args.max_train_steps:
                    break
        else:
            continue
        break

    final_dir = args.output_dir / "final"
    final_dir.mkdir(parents=True, exist_ok=True)
    _save_condition_lora_state(pipeline.transformer, final_dir / "flux_lora.pt")
    torch.save(graph_encoder.state_dict(), final_dir / "graph_encoder.pt")
    _save_state(args.output_dir, step=global_step, args=args, lora_layers=lora_layers)
    print(f"Relation FLUX LoRA training finished at step {global_step}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
