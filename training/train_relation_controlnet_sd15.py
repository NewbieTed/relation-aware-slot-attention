"""Train a ControlNet adapter from relation-aware GNN layout maps.

This trainer is intentionally separate from ``train_relation_aware_sd15.py``.
The older path appends learned slot embeddings to the CLIP token sequence and
modifies cross-attention. This path keeps the base SD1.5 U-Net, VAE, CLIP text
encoder, and GNN frozen, then trains only a ControlNet branch that receives an
explicit spatial condition image built from the GNN's predicted object ellipses.

High-level data flow:
    prompt/image row -> frozen GNN -> Gaussian layout map -> trainable ControlNet
    normal CLIP prompt -> frozen U-Net cross-attention
    ControlNet residuals + noisy latents -> frozen U-Net denoising prediction
"""

from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from evaluation.generate import save_gnn_overlay
from evaluation.prompt_parser import parse_prompt_to_scene_graph
from .dataset import (
    SCOPDepthTextToImageDataset,
    build_dataset_splits,
    collate_training_items,
)
from .graph_modules import GraphSlotEncoder, build_slot_conditioning, pooled_label_embeddings
from .graph_targets import bbox_centers_after_crop, bbox_log_sigmas_after_crop
from .layout_conditioning import build_gaussian_layout_maps
from .metrics import MetricsLogger, write_split_manifest
from .prompts import prompt_from_scop_depth_row
from .scene_graph import build_batched_scene_graphs
from .train_sd15_lora import (
    DEFAULT_MODEL_ID,
    _is_tqdm_disabled,
    build_autocast_context,
    choose_weight_dtype,
    resolve_torch_device,
    set_seed,
)


def make_parser() -> argparse.ArgumentParser:
    """Create CLI options for the ControlNet layout experiment.

    The most important switches are ``--layout-source`` and
    ``--controlnet-conditioning-scale``. ``layout-source=gnn`` tests the real
    pipeline; ``layout-source=gt`` is a debugging upper bound using dataset
    boxes instead of GNN predictions.
    """

    parser = argparse.ArgumentParser(
        description="Train a ControlNet layout adapter for relation-aware SD1.5."
    )
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-id", type=str, default=DEFAULT_MODEL_ID)
    parser.add_argument("--init-graph-encoder", type=Path, required=True)
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    parser.add_argument("--mixed-precision", choices=("no", "fp16", "bf16"), default="fp16")
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--max-train-steps", type=int, default=24000)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--slot-dim", type=int, default=512)
    parser.add_argument("--gnn-layers", type=int, default=2)
    parser.add_argument("--limit-rows", type=int, default=None)
    parser.add_argument("--prompt-prefix", type=str, default="a photo of")
    parser.add_argument("--eval-fraction", type=float, default=0.1)
    parser.add_argument("--test-fraction", type=float, default=0.1)
    parser.add_argument("--layout-source", choices=("gnn", "gt"), default="gnn")
    parser.add_argument("--layout-sigma-scale", type=float, default=1.0)
    parser.add_argument("--layout-semantic-channels", type=int, default=8)
    parser.add_argument("--controlnet-conditioning-scale", type=float, default=1.0)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--eval-every", type=int, default=1000)
    parser.add_argument("--save-every", type=int, default=4000)
    parser.add_argument("--eval-sample-prompts", type=int, default=4)
    parser.add_argument("--eval-sample-images", type=int, default=2)
    parser.add_argument("--eval-sample-inference-steps", type=int, default=30)
    parser.add_argument("--eval-sample-guidance-scale", type=float, default=7.5)
    parser.add_argument("--disable-tqdm", action="store_true")
    return parser


def _save_state(output_dir: Path, *, step: int, args: argparse.Namespace) -> None:
    """Write resumability/debug metadata for the current training run."""

    payload = {
        "step": step,
        "dataset_dir": str(args.dataset_dir),
        "model_id": args.model_id,
        "init_graph_encoder": str(args.init_graph_encoder),
        "layout_source": args.layout_source,
        "layout_sigma_scale": args.layout_sigma_scale,
        "layout_semantic_channels": args.layout_semantic_channels,
        "controlnet_conditioning_scale": args.controlnet_conditioning_scale,
        "learning_rate": args.learning_rate,
        "batch_size": args.batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
    }
    (output_dir / "training_state.json").write_text(json.dumps(payload, indent=2))


def _save_controlnet(
    *,
    output_dir: Path,
    step: int,
    controlnet: Any,
    graph_encoder: GraphSlotEncoder,
    optimizer: torch.optim.Optimizer,
) -> Path:
    """Save the trainable ControlNet plus frozen GNN reference checkpoint.

    The graph encoder is frozen during training, but saving it beside the
    ControlNet makes evaluation self-contained through ``--relation-aware-dir``.
    """

    checkpoint_dir = output_dir / f"checkpoint-{step:06d}"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    controlnet.save_pretrained(checkpoint_dir / "controlnet")
    torch.save(graph_encoder.state_dict(), checkpoint_dir / "graph_encoder.pt")
    torch.save(optimizer.state_dict(), checkpoint_dir / "optimizer.pt")
    return checkpoint_dir


def _sample_eval_prompts(
    dataset: SCOPDepthTextToImageDataset,
    *,
    prompt_count: int,
    step: int,
    seed: int,
) -> list[str]:
    """Select deterministic validation prompts for periodic image snapshots."""

    rows = list(dataset.rows)
    if prompt_count <= 0 or not rows:
        return []
    rng = random.Random(seed + step * 7919)
    if prompt_count >= len(rows):
        selected = rows
        rng.shuffle(selected)
    else:
        selected = rng.sample(rows, prompt_count)
    return [prompt_from_scop_depth_row(row, prefix=dataset.prompt_prefix) for row in selected]


def _layout_from_batch(
    *,
    batch: dict[str, Any],
    tokenizer: Any,
    text_encoder: Any,
    graph_encoder: GraphSlotEncoder,
    device: str,
    image_size: int,
    layout_source: str,
    sigma_scale: float,
    semantic_channels: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build the image-like ControlNet condition for a training batch.

    Returns:
        layout: The rasterized Gaussian condition image.
        slot_positions: GNN-predicted x/y/z centers, used for overlays.
        slot_log_sigmas: GNN-predicted x/y log sigmas, used for overlays.
        slot_mask: Boolean object-slot mask.
    """

    # Dataset rows may contain fewer objects in future variants, so derive the
    # padded slot count from the current mini-batch rather than hard-coding two.
    max_nodes = max(len(graph["nodes"]) for graph in batch["scene_graphs"])

    # Ground-truth centers/sigmas are still useful for debugging with
    # --layout-source gt, and they provide the slot mask shape in both modes.
    gt_centers, slot_mask = bbox_centers_after_crop(
        batch["metadata"],
        batch["image_sizes"],
        max_nodes=max_nodes,
        device=torch.device(device),
    )
    gt_log_sigmas, _ = bbox_log_sigmas_after_crop(
        batch["metadata"],
        batch["image_sizes"],
        max_nodes=max_nodes,
        device=torch.device(device),
    )
    scene_graph_batch = build_batched_scene_graphs(
        batch["scene_graphs"],
        slot_targets=gt_centers,
        slot_mask=slot_mask,
    )

    # These pooled object-label features are used twice: the GNN consumes them
    # internally, and the ControlNet condition map uses their first dimensions
    # as object-identity heatmap channels.
    label_features = pooled_label_embeddings(
        tokenizer=tokenizer,
        text_encoder=text_encoder,
        scene_graph_batch=scene_graph_batch,
        device=device,
        dtype=graph_encoder.node_proj.weight.dtype,
    )

    # The frozen GNN reads CLIP-pooled object labels plus scene-graph edges and
    # predicts one embedding, center, and sigma pair per object slot.
    conditioning = build_slot_conditioning(
        tokenizer=tokenizer,
        text_encoder=text_encoder,
        scene_graph_batch=scene_graph_batch,
        graph_encoder=graph_encoder,
        device=device,
    )
    if layout_source == "gt":
        # Debug/upper-bound mode: bypass the learned GNN layout and ask whether
        # ControlNet can use perfect dataset layout maps.
        layout_centers = gt_centers
        layout_log_sigmas = gt_log_sigmas
    else:
        # Real experiment mode: use the GNN-predicted spatial layout.
        layout_centers = conditioning.slot_positions
        layout_log_sigmas = conditioning.slot_log_sigmas

    # Convert centers/sigmas into the 3-channel raster condition consumed by
    # ControlNet's conditioning stem.
    layout = build_gaussian_layout_maps(
        slot_centers=layout_centers,
        slot_log_sigmas=layout_log_sigmas,
        slot_mask=slot_mask,
        image_size=image_size,
        channels=3 + semantic_channels,
        sigma_scale=sigma_scale,
        slot_features=label_features,
        semantic_channels=semantic_channels,
    )
    return layout, conditioning.slot_positions, conditioning.slot_log_sigmas, slot_mask


def _build_losses(
    *,
    batch: dict[str, Any],
    tokenizer: Any,
    text_encoder: Any,
    graph_encoder: GraphSlotEncoder,
    controlnet: Any,
    unet: Any,
    vae: Any,
    noise_scheduler: Any,
    device: str,
    weight_dtype: torch.dtype,
    mixed_precision: str,
    image_size: int,
    layout_source: str,
    layout_sigma_scale: float,
    layout_semantic_channels: int,
    conditioning_scale: float,
) -> dict[str, torch.Tensor]:
    """Run one forward pass and compute the ControlNet denoising objective.

    Only ControlNet receives gradients. The VAE, text encoder, base U-Net, and
    GNN are used as frozen feature/layout providers.
    """

    # Encode target images into SD latent space, exactly like normal diffusion
    # fine-tuning. This is frozen, so gradients are not needed here.
    pixel_values = batch["pixel_values"].to(device=device, dtype=weight_dtype)
    with torch.no_grad():
        latents = vae.encode(pixel_values).latent_dist.sample()
        latents = latents * vae.config.scaling_factor

    # Sample a diffusion timestep and corrupt the latent with Gaussian noise.
    noise = torch.randn_like(latents)
    timesteps = torch.randint(
        0,
        noise_scheduler.config.num_train_timesteps,
        (latents.shape[0],),
        device=latents.device,
        dtype=torch.int64,
    )
    noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

    # Text conditioning remains vanilla CLIP. We deliberately do not append slot
    # embeddings in this ControlNet experiment.
    text_inputs = tokenizer(
        batch["prompts"],
        padding="max_length",
        truncation=True,
        max_length=tokenizer.model_max_length,
        return_tensors="pt",
    )
    with torch.no_grad():
        text_hidden_states = text_encoder(
            text_inputs.input_ids.to(device),
            attention_mask=text_inputs.attention_mask.to(device),
        )[0]
        layout, _, _, _ = _layout_from_batch(
            batch=batch,
            tokenizer=tokenizer,
            text_encoder=text_encoder,
            graph_encoder=graph_encoder,
            device=device,
            image_size=image_size,
            layout_source=layout_source,
            sigma_scale=layout_sigma_scale,
            semantic_channels=layout_semantic_channels,
        )

    with build_autocast_context(device, mixed_precision):
        # ControlNet predicts additive residuals for the frozen U-Net blocks.
        # The base U-Net still performs the final denoising prediction.
        down_res, mid_res = controlnet(
            noisy_latents,
            timesteps,
            encoder_hidden_states=text_hidden_states,
            controlnet_cond=layout.to(device=device, dtype=weight_dtype),
            conditioning_scale=conditioning_scale,
            return_dict=False,
        )
        control_down_norm = torch.stack(
            [res.float().pow(2).mean().sqrt() for res in down_res]
        ).mean()
        control_mid_norm = mid_res.float().pow(2).mean().sqrt()
        model_pred = unet(
            noisy_latents,
            timesteps,
            encoder_hidden_states=text_hidden_states,
            down_block_additional_residuals=down_res,
            mid_block_additional_residual=mid_res,
        ).sample
        target = (
            noise
            if noise_scheduler.config.prediction_type == "epsilon"
            else noise_scheduler.get_velocity(latents, noise, timesteps)
        )

        # The loss is intentionally just denoising MSE. Layout pressure enters
        # through the ControlNet condition, not through extra hand-written terms.
        denoise_loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
    return {
        "loss": denoise_loss,
        "denoise_loss": denoise_loss,
        "control_down_norm": control_down_norm.detach(),
        "control_mid_norm": control_mid_norm.detach(),
    }


@torch.no_grad()
def _evaluate(
    *,
    dataloader: DataLoader,
    tokenizer: Any,
    text_encoder: Any,
    graph_encoder: GraphSlotEncoder,
    controlnet: Any,
    unet: Any,
    vae: Any,
    noise_scheduler: Any,
    device: str,
    weight_dtype: torch.dtype,
    mixed_precision: str,
    image_size: int,
    layout_source: str,
    layout_sigma_scale: float,
    layout_semantic_channels: int,
    conditioning_scale: float,
) -> dict[str, float]:
    """Measure validation denoising loss without updating ControlNet weights."""

    controlnet.eval()
    totals = {
        "loss": 0.0,
        "denoise_loss": 0.0,
        "control_down_norm": 0.0,
        "control_mid_norm": 0.0,
    }
    count = 0
    for batch in dataloader:
        metrics = _build_losses(
            batch=batch,
            tokenizer=tokenizer,
            text_encoder=text_encoder,
            graph_encoder=graph_encoder,
            controlnet=controlnet,
            unet=unet,
            vae=vae,
            noise_scheduler=noise_scheduler,
            device=device,
            weight_dtype=weight_dtype,
            mixed_precision=mixed_precision,
            image_size=image_size,
            layout_source=layout_source,
            layout_sigma_scale=layout_sigma_scale,
            layout_semantic_channels=layout_semantic_channels,
            conditioning_scale=conditioning_scale,
        )
        for key in totals:
            totals[key] += float(metrics[key].item())
        count += 1
    controlnet.train()
    if count == 0:
        return {key: 0.0 for key in totals}
    return {key: value / count for key, value in totals.items()}


@torch.no_grad()
def _run_eval_samples(
    *,
    args: argparse.Namespace,
    prompts: list[str],
    tokenizer: Any,
    text_encoder: Any,
    graph_encoder: GraphSlotEncoder,
    controlnet: Any,
    unet: Any,
    vae: Any,
    noise_scheduler: Any,
    device: str,
    weight_dtype: torch.dtype,
    step: int,
) -> None:
    """Generate qualitative validation images and matching GNN-layout overlays.

    These samples are not used for optimization. They are a quick visual check
    that the generated objects, ControlNet condition, and GNN-predicted ellipses
    are at least moving in the same direction during training.
    """

    if args.eval_sample_images <= 0 or not prompts:
        return
    from diffusers import StableDiffusionControlNetPipeline

    sample_dir = args.output_dir / "validation" / f"step-{step:06d}" / "samples"
    overlay_dir = args.output_dir / "validation" / f"step-{step:06d}" / "overlays"
    sample_dir.mkdir(parents=True, exist_ok=True)
    overlay_dir.mkdir(parents=True, exist_ok=True)
    (sample_dir.parent / "prompts.txt").write_text("\n".join(prompts) + "\n")

    was_training = controlnet.training
    controlnet.eval()

    # Training keeps ControlNet parameters in fp32 so GradScaler/autocast have
    # stable master weights. Diffusers' pipeline feeds fp16 tensors during fp16
    # inference, so temporarily match the inference dtype for sample generation.
    controlnet.to(device=device, dtype=weight_dtype)

    # Reuse the already-loaded frozen components and currently-trained
    # ControlNet, so snapshots reflect the exact in-memory checkpoint.
    pipeline = StableDiffusionControlNetPipeline.from_pretrained(
        args.model_id,
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        unet=unet,
        controlnet=controlnet,
        safety_checker=None,
        torch_dtype=weight_dtype,
    ).to(device)
    pipeline.set_progress_bar_config(disable=True)

    for prompt_index, prompt in enumerate(prompts):
        # Eval prompts come from dataset rows, but we still parse from text so
        # this path matches standalone prompt generation as closely as possible.
        scene_graph = parse_prompt_to_scene_graph(prompt)
        max_nodes = len(scene_graph["nodes"])

        # Prompt-only inference has no GT boxes, so dummy targets only provide
        # shapes for the graph batching utility. The GNN prediction is what
        # actually defines the ControlNet layout map below.
        dummy_targets = torch.zeros(1, max_nodes, 3, device=device)
        dummy_mask = torch.ones(1, max_nodes, dtype=torch.bool, device=device)
        scene_graph_batch = build_batched_scene_graphs(
            [scene_graph],
            slot_targets=dummy_targets,
            slot_mask=dummy_mask,
        )
        label_features = pooled_label_embeddings(
            tokenizer=tokenizer,
            text_encoder=text_encoder,
            scene_graph_batch=scene_graph_batch,
            device=device,
            dtype=graph_encoder.node_proj.weight.dtype,
        )
        conditioning = build_slot_conditioning(
            tokenizer=tokenizer,
            text_encoder=text_encoder,
            scene_graph_batch=scene_graph_batch,
            graph_encoder=graph_encoder,
            device=device,
        )
        layout = build_gaussian_layout_maps(
            slot_centers=conditioning.slot_positions,
            slot_log_sigmas=conditioning.slot_log_sigmas,
            slot_mask=dummy_mask,
            image_size=args.image_size,
            channels=3 + args.layout_semantic_channels,
            sigma_scale=args.layout_sigma_scale,
            slot_features=label_features,
            semantic_channels=args.layout_semantic_channels,
        )
        for image_index in range(args.eval_sample_images):
            # Use deterministic per-prompt/per-image seeds so snapshots are
            # comparable across checkpoints.
            generator = torch.Generator(device="cpu").manual_seed(
                args.seed + step * 1009 + prompt_index * args.eval_sample_images + image_index
            )
            image = pipeline(
                prompt=prompt,
                image=layout.to(device=device, dtype=weight_dtype),
                num_inference_steps=args.eval_sample_inference_steps,
                guidance_scale=args.eval_sample_guidance_scale,
                controlnet_conditioning_scale=args.controlnet_conditioning_scale,
                height=args.image_size,
                width=args.image_size,
                generator=generator,
            ).images[0]
            filename = f"prompt{prompt_index:02d}_{image_index:02d}.png"
            image.save(sample_dir / filename)

            # Overlay lets us inspect whether bad generation is due to layout
            # prediction or due to ControlNet/U-Net not using that layout well.
            save_gnn_overlay(
                image=image,
                prompt=prompt,
                slot_positions=conditioning.slot_positions,
                slot_log_sigmas=conditioning.slot_log_sigmas,
                output_path=overlay_dir / filename,
            )
    del pipeline

    # Restore fp32 parameters before returning to the optimizer-backed training
    # loop. Parameter objects stay the same, so optimizer state remains attached.
    controlnet.to(device=device, dtype=torch.float32)
    if was_training:
        controlnet.train()
    if device == "cuda":
        torch.cuda.empty_cache()


def main() -> int:
    """CLI entry point for frozen-GNN ControlNet training."""

    args = make_parser().parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    device = resolve_torch_device(args.device)
    weight_dtype = choose_weight_dtype(device, args.mixed_precision)
    set_seed(args.seed)
    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=device == "cuda" and args.mixed_precision == "fp16",
    )

    # Build stable train/eval/test splits and save the row manifest so a run can
    # be audited later.
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
    eval_dataloader = DataLoader(
        datasets["eval"],
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_training_items,
    )

    # Imports stay inside main so users who only inspect CLI help do not pay the
    # cost of importing diffusers/transformers.
    from diffusers import AutoencoderKL, ControlNetModel, DDPMScheduler, UNet2DConditionModel
    from transformers import CLIPTextModel, CLIPTokenizer

    tokenizer = CLIPTokenizer.from_pretrained(args.model_id, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(args.model_id, subfolder="text_encoder", torch_dtype=weight_dtype)
    vae = AutoencoderKL.from_pretrained(args.model_id, subfolder="vae", torch_dtype=weight_dtype)
    unet = UNet2DConditionModel.from_pretrained(args.model_id, subfolder="unet", torch_dtype=weight_dtype)
    noise_scheduler = DDPMScheduler.from_pretrained(args.model_id, subfolder="scheduler")
    controlnet = ControlNetModel.from_unet(
        unet,
        conditioning_channels=3 + args.layout_semantic_channels,
    )

    # Freeze the original SD pipeline. ControlNet is the only trainable model,
    # which isolates whether explicit layout residuals help.
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    unet.requires_grad_(False)
    controlnet.requires_grad_(True)
    vae.to(device)
    text_encoder.to(device)
    unet.to(device)
    controlnet.to(device)
    for parameter in controlnet.parameters():
        if parameter.requires_grad and parameter.dtype != torch.float32:
            parameter.data = parameter.data.to(torch.float32)

    # Load the pretrained relation-aware GNN and freeze it. This gives us a
    # fixed layout generator rather than letting layout drift during ControlNet
    # training.
    graph_encoder = GraphSlotEncoder(
        text_hidden_dim=text_encoder.config.hidden_size,
        slot_dim=args.slot_dim,
        num_layers=args.gnn_layers,
    ).to(device)
    graph_encoder.load_state_dict(torch.load(args.init_graph_encoder, map_location=device))
    graph_encoder.requires_grad_(False)
    graph_encoder.eval()

    # AdamW defaults are used intentionally here; the main hyperparameter we are
    # testing first is whether ControlNet can learn from the GNN condition map.
    optimizer = torch.optim.AdamW(controlnet.parameters(), lr=args.learning_rate)
    metrics_logger = MetricsLogger(
        args.output_dir,
        fieldnames=[
            "step",
            "split",
            "loss",
            "denoise_loss",
            "control_down_norm",
            "control_mid_norm",
        ],
    )
    _save_state(args.output_dir, step=0, args=args)

    progress = tqdm(total=args.max_train_steps, disable=_is_tqdm_disabled(args), desc="RelationControlNet")
    global_step = 0
    micro_step = 0
    running_totals = {
        "loss": 0.0,
        "denoise_loss": 0.0,
        "control_down_norm": 0.0,
        "control_mid_norm": 0.0,
    }
    running_updates = 0
    optimizer.zero_grad(set_to_none=True)
    while global_step < args.max_train_steps:
        for batch in train_dataloader:
            metrics = _build_losses(
                batch=batch,
                tokenizer=tokenizer,
                text_encoder=text_encoder,
                graph_encoder=graph_encoder,
                controlnet=controlnet,
                unet=unet,
                vae=vae,
                noise_scheduler=noise_scheduler,
                device=device,
                weight_dtype=weight_dtype,
                mixed_precision=args.mixed_precision,
                image_size=args.image_size,
                layout_source=args.layout_source,
                layout_sigma_scale=args.layout_sigma_scale,
                layout_semantic_channels=args.layout_semantic_channels,
                conditioning_scale=args.controlnet_conditioning_scale,
            )
            loss = metrics["loss"]
            if not torch.isfinite(loss):
                raise FloatingPointError("Encountered a non-finite ControlNet training loss.")

            # Gradient accumulation keeps the effective batch size configurable
            # without changing the per-GPU memory footprint.
            scaled_loss = loss / args.gradient_accumulation_steps
            if scaler.is_enabled():
                scaler.scale(scaled_loss).backward()
            else:
                scaled_loss.backward()
            micro_step += 1
            for key in running_totals:
                running_totals[key] += float(metrics[key].item())
            running_updates += 1
            if micro_step % args.gradient_accumulation_steps == 0:
                # Optimizer step updates ControlNet only; frozen modules have
                # requires_grad=False and never receive gradients.
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
                        **{
                            key: value / running_updates
                            for key, value in running_totals.items()
                        },
                    }
                    metrics_logger.log(train_log)
                    progress.set_postfix(
                        loss=f"{train_log['loss']:.4f}",
                        ctrl=f"{train_log['control_down_norm']:.4f}",
                    )
                    for key in running_totals:
                        running_totals[key] = 0.0
                    running_updates = 0
                if len(datasets["eval"]) > 0 and args.eval_every > 0 and global_step % args.eval_every == 0:
                    # Numeric eval catches training divergence, while image
                    # snapshots catch "low loss but ugly image" failure modes.
                    eval_log = {
                        "step": global_step,
                        "split": "eval",
                        **_evaluate(
                            dataloader=eval_dataloader,
                            tokenizer=tokenizer,
                            text_encoder=text_encoder,
                            graph_encoder=graph_encoder,
                            controlnet=controlnet,
                            unet=unet,
                            vae=vae,
                            noise_scheduler=noise_scheduler,
                            device=device,
                            weight_dtype=weight_dtype,
                            mixed_precision=args.mixed_precision,
                            image_size=args.image_size,
                            layout_source=args.layout_source,
                            layout_sigma_scale=args.layout_sigma_scale,
                            layout_semantic_channels=args.layout_semantic_channels,
                            conditioning_scale=args.controlnet_conditioning_scale,
                        ),
                    }
                    metrics_logger.log(eval_log)
                    print(f"Eval at step {global_step}: loss={eval_log['loss']:.4f}")
                    prompts = _sample_eval_prompts(
                        datasets["eval"],
                        prompt_count=args.eval_sample_prompts,
                        step=global_step,
                        seed=args.seed,
                    )
                    _run_eval_samples(
                        args=args,
                        prompts=prompts,
                        tokenizer=tokenizer,
                        text_encoder=text_encoder,
                        graph_encoder=graph_encoder,
                        controlnet=controlnet,
                        unet=unet,
                        vae=vae,
                        noise_scheduler=noise_scheduler,
                        device=device,
                        weight_dtype=weight_dtype,
                        step=global_step,
                    )
                if global_step % args.save_every == 0:
                    # Store periodic checkpoints so we can roll back to the
                    # best qualitative stage if later steps overfit or degrade.
                    checkpoint = _save_controlnet(
                        output_dir=args.output_dir,
                        step=global_step,
                        controlnet=controlnet,
                        graph_encoder=graph_encoder,
                        optimizer=optimizer,
                    )
                    _save_state(args.output_dir, step=global_step, args=args)
                    print(f"Saved ControlNet checkpoint to {checkpoint}")
                if global_step >= args.max_train_steps:
                    break
        else:
            continue
        break

    final_dir = args.output_dir / "final"
    final_dir.mkdir(parents=True, exist_ok=True)
    controlnet.save_pretrained(final_dir / "controlnet")
    torch.save(graph_encoder.state_dict(), final_dir / "graph_encoder.pt")
    _save_state(args.output_dir, step=global_step, args=args)
    print(f"Relation ControlNet training finished at step {global_step}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
