from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from .dataset import SCOPDepthTextToImageDataset, collate_training_items
from .graph_modules import (
    GraphSlotEncoder,
    build_slot_conditioning,
    relation_loss,
)
from .relation_attention import install_relation_aware_processors
from .scene_graph import build_batched_scene_graphs
from .train_sd15_lora import (
    DEFAULT_MODEL_ID,
    _attach_lora_adapters,
    _is_tqdm_disabled,
    _load_validation_prompts,
    build_autocast_context,
    choose_weight_dtype,
    resolve_torch_device,
    set_seed,
)


def _bbox_centers_after_crop(
    metadata_rows: list[dict[str, Any]],
    image_sizes: list[tuple[int, int]],
    max_nodes: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    targets = torch.zeros(len(metadata_rows), max_nodes, 3, device=device)
    mask = torch.zeros(len(metadata_rows), max_nodes, dtype=torch.bool, device=device)
    for batch_index, (row, (width, height)) in enumerate(zip(metadata_rows, image_sizes)):
        crop_size = min(width, height)
        left = (width - crop_size) / 2.0
        top = (height - crop_size) / 2.0
        depth = row.get("depth")
        depth_values = [
            float(depth["bbox1"]["median"]) if depth else 0.0,
            float(depth["bbox2"]["median"]) if depth else 0.0,
        ]
        for node_index, annot in enumerate(row["annots"][:max_nodes]):
            x, y, w, h = annot["bbox"]
            cx = ((x + w / 2.0) - left) / crop_size
            cy = ((y + h / 2.0) - top) / crop_size
            cx = max(0.0, min(1.0, cx))
            cy = max(0.0, min(1.0, cy))
            cz = depth_values[node_index] if node_index < len(depth_values) else 0.0
            targets[batch_index, node_index] = torch.tensor(
                [cx * 2 - 1, cy * 2 - 1, cz * 2 - 1],
                device=device,
            )
            mask[batch_index, node_index] = True
    return targets, mask


def _save_state(
    output_dir: Path,
    *,
    step: int,
    args: argparse.Namespace,
    validation_prompts: list[str],
) -> None:
    payload = {
        "step": step,
        "dataset_dir": str(args.dataset_dir),
        "model_id": args.model_id,
        "learning_rate": args.learning_rate,
        "graph_learning_rate": args.graph_learning_rate,
        "lora_rank": args.lora_rank,
        "slot_dim": args.slot_dim,
        "gnn_layers": args.gnn_layers,
        "aux_loss_weight": args.aux_loss_weight,
        "relation_loss_weight": args.relation_loss_weight,
        "validation_prompts": validation_prompts,
    }
    (output_dir / "training_state.json").write_text(json.dumps(payload, indent=2))


def _save_modules(
    output_dir: Path,
    *,
    step: int,
    graph_encoder: GraphSlotEncoder,
    unet: Any,
    optimizer: torch.optim.Optimizer,
    relation_attention_processors: dict[str, torch.nn.Module],
) -> Path:
    checkpoint_dir = output_dir / f"checkpoint-{step:06d}"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    if hasattr(unet, "save_lora_adapter"):
        unet.save_lora_adapter(checkpoint_dir / "lora", adapter_name="default")
    else:
        unet.save_attn_procs(checkpoint_dir / "lora")
    torch.save(graph_encoder.state_dict(), checkpoint_dir / "graph_encoder.pt")
    torch.save(
        {name: module.state_dict() for name, module in relation_attention_processors.items()},
        checkpoint_dir / "relation_attention.pt",
    )
    torch.save(optimizer.state_dict(), checkpoint_dir / "optimizer.pt")
    return checkpoint_dir


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train the relation-aware SCOP-Depth method with graph slots and modified cross-attention."
    )
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-id", type=str, default=DEFAULT_MODEL_ID)
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    parser.add_argument("--mixed-precision", choices=("no", "fp16", "bf16"), default="fp16")
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--graph-learning-rate", type=float, default=5e-4)
    parser.add_argument("--max-train-steps", type=int, default=1000)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--slot-dim", type=int, default=512)
    parser.add_argument("--gnn-layers", type=int, default=2)
    parser.add_argument("--save-every", type=int, default=250)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--limit-rows", type=int, default=None)
    parser.add_argument("--prompt-prefix", type=str, default="a photo of")
    parser.add_argument("--validation-prompts-file", type=Path, default=None)
    parser.add_argument("--num-validation-images", type=int, default=4)
    parser.add_argument("--validation-every", type=int, default=0)
    parser.add_argument("--aux-loss-weight", type=float, default=0.1)
    parser.add_argument("--relation-loss-weight", type=float, default=0.1)
    parser.add_argument("--disable-tqdm", action="store_true")
    return parser


def main() -> int:
    args = make_parser().parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    device = resolve_torch_device(args.device)
    weight_dtype = choose_weight_dtype(device, args.mixed_precision)
    set_seed(args.seed)
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=device == "cuda" and args.mixed_precision == "fp16",
    )

    dataset = SCOPDepthTextToImageDataset(
        args.dataset_dir,
        image_size=args.image_size,
        prompt_prefix=args.prompt_prefix,
        limit_rows=args.limit_rows,
        shuffle_rows=True,
        seed=args.seed,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_training_items,
    )

    from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
    from transformers import CLIPTextModel, CLIPTokenizer

    tokenizer = CLIPTokenizer.from_pretrained(args.model_id, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(
        args.model_id,
        subfolder="text_encoder",
        torch_dtype=weight_dtype,
    )
    vae = AutoencoderKL.from_pretrained(
        args.model_id,
        subfolder="vae",
        torch_dtype=weight_dtype,
    )
    unet = UNet2DConditionModel.from_pretrained(
        args.model_id,
        subfolder="unet",
        torch_dtype=weight_dtype,
    )
    noise_scheduler = DDPMScheduler.from_pretrained(args.model_id, subfolder="scheduler")

    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    unet.requires_grad_(False)
    vae.to(device)
    text_encoder.to(device)
    unet.to(device)

    relation_attention_processors = install_relation_aware_processors(unet)
    lora_optimizer = _attach_lora_adapters(unet, args.lora_rank, args.learning_rate)

    graph_encoder = GraphSlotEncoder(
        text_hidden_dim=text_encoder.config.hidden_size,
        slot_dim=args.slot_dim,
        num_layers=args.gnn_layers,
    ).to(device)

    relation_params = []
    for name, module in relation_attention_processors.items():
        if name.endswith("attn1.processor"):
            continue
        relation_params.extend(list(module.parameters()))

    optimizer = torch.optim.AdamW(
        [
            {"params": lora_optimizer.param_groups[0]["params"], "lr": args.learning_rate},
            {"params": graph_encoder.parameters(), "lr": args.graph_learning_rate},
            {"params": relation_params, "lr": args.graph_learning_rate},
        ]
    )

    validation_prompts = _load_validation_prompts(dataset, args.validation_prompts_file)
    _save_state(args.output_dir, step=0, args=args, validation_prompts=validation_prompts)

    progress_bar = tqdm(
        total=args.max_train_steps,
        disable=_is_tqdm_disabled(args),
        desc="RelationAwareTraining",
    )

    global_step = 0
    micro_step = 0
    optimizer.zero_grad(set_to_none=True)

    while global_step < args.max_train_steps:
        for batch in dataloader:
            pixel_values = batch["pixel_values"].to(device=device, dtype=weight_dtype)
            with torch.no_grad():
                latents = vae.encode(pixel_values).latent_dist.sample()
                latents = latents * vae.config.scaling_factor

            noise = torch.randn_like(latents)
            bsz = latents.shape[0]
            timesteps = torch.randint(
                0,
                noise_scheduler.config.num_train_timesteps,
                (bsz,),
                device=latents.device,
                dtype=torch.int64,
            )
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

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

            max_nodes = max(len(graph["nodes"]) for graph in batch["scene_graphs"])
            slot_targets, slot_mask = _bbox_centers_after_crop(
                batch["metadata"],
                batch["image_sizes"],
                max_nodes=max_nodes,
                device=torch.device(device),
            )
            scene_graph_batch = build_batched_scene_graphs(
                batch["scene_graphs"],
                slot_targets=slot_targets,
                slot_mask=slot_mask,
            )
            conditioning = build_slot_conditioning(
                tokenizer=tokenizer,
                text_encoder=text_encoder,
                scene_graph_batch=scene_graph_batch,
                graph_encoder=graph_encoder,
                device=device,
            )
            encoder_hidden_states = torch.cat(
                [text_hidden_states, conditioning.slot_embeddings.to(text_hidden_states.dtype)],
                dim=1,
            )

            with build_autocast_context(device, args.mixed_precision):
                model_pred = unet(
                    noisy_latents,
                    timesteps,
                    encoder_hidden_states=encoder_hidden_states,
                    cross_attention_kwargs={
                        "slot_positions": conditioning.slot_positions,
                        "slot_mask": conditioning.slot_mask,
                        "text_token_count": text_hidden_states.shape[1],
                    },
                ).sample
                target = (
                    noise
                    if noise_scheduler.config.prediction_type == "epsilon"
                    else noise_scheduler.get_velocity(latents, noise, timesteps)
                )
                denoise_loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
                geometry_loss = F.smooth_l1_loss(
                    conditioning.slot_positions[conditioning.slot_mask],
                    slot_targets[conditioning.slot_mask],
                )
                edge_loss = relation_loss(
                    conditioning.relation_logits,
                    conditioning.slot_positions,
                    scene_graph_batch,
                )
                loss = (
                    denoise_loss
                    + args.aux_loss_weight * geometry_loss
                    + args.relation_loss_weight * edge_loss
                )

            if not torch.isfinite(loss):
                raise FloatingPointError(
                    "Encountered a non-finite training loss in the relation-aware trainer."
                )
            scaled_loss = loss / args.gradient_accumulation_steps
            if scaler.is_enabled():
                scaler.scale(scaled_loss).backward()
            else:
                scaled_loss.backward()
            micro_step += 1

            if micro_step % args.gradient_accumulation_steps == 0:
                if scaler.is_enabled():
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
                progress_bar.update(1)

                if global_step % args.log_every == 0:
                    progress_bar.set_postfix(
                        denoise=f"{denoise_loss.item():.4f}",
                        geom=f"{geometry_loss.item():.4f}",
                        rel=f"{edge_loss.item():.4f}",
                    )

                if global_step % args.save_every == 0:
                    checkpoint_dir = _save_modules(
                        args.output_dir,
                        step=global_step,
                        graph_encoder=graph_encoder,
                        unet=unet,
                        optimizer=optimizer,
                        relation_attention_processors=relation_attention_processors,
                    )
                    _save_state(
                        args.output_dir,
                        step=global_step,
                        args=args,
                        validation_prompts=validation_prompts,
                    )
                    print(f"Saved relation-aware checkpoint to {checkpoint_dir}")

                if global_step >= args.max_train_steps:
                    break
        else:
            continue
        break

    final_dir = args.output_dir / "final"
    final_dir.mkdir(parents=True, exist_ok=True)
    if hasattr(unet, "save_lora_adapter"):
        unet.save_lora_adapter(final_dir / "lora", adapter_name="default")
    else:
        unet.save_attn_procs(final_dir / "lora")
    torch.save(graph_encoder.state_dict(), final_dir / "graph_encoder.pt")
    torch.save(
        {name: module.state_dict() for name, module in relation_attention_processors.items()},
        final_dir / "relation_attention.pt",
    )
    _save_state(args.output_dir, step=global_step, args=args, validation_prompts=validation_prompts)
    print(f"Relation-aware training finished at step {global_step}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
