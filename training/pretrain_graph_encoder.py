from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from .dataset import build_dataset_splits, collate_training_items
from .graph_modules import (
    GraphSlotEncoder,
    build_slot_conditioning,
    embedding_alignment_loss,
    inverse_relation_regularizer,
    log_size_3d_loss,
    log_sigma_loss,
    pooled_label_embeddings,
    relation_loss,
)
from .graph_targets import (
    bbox_centers_after_crop,
    bbox_log_sigmas_after_crop,
    bbox_log_sizes_3d_after_crop,
)
from .metrics import MetricsLogger, write_split_manifest
from .runtime import (
    DEFAULT_FLUX_MODEL_ID,
    choose_weight_dtype,
    is_tqdm_disabled,
    resolve_torch_device,
    set_seed,
)
from .scene_graph import build_batched_scene_graphs


def _save_state(output_dir: Path, *, step: int, args: argparse.Namespace) -> None:
    payload = {
        "step": step,
        "dataset_dir": str(args.dataset_dir),
        "model_id": args.model_id,
        "graph_learning_rate": args.graph_learning_rate,
        "slot_dim": args.slot_dim,
        "gnn_layers": args.gnn_layers,
        "position_loss_weight": args.position_loss_weight,
        "relation_loss_weight": args.relation_loss_weight,
        "embedding_loss_weight": args.embedding_loss_weight,
        "inverse_relation_loss_weight": args.inverse_relation_loss_weight,
        "box_loss_weight": args.box_loss_weight,
        "box3d_loss_weight": args.box3d_loss_weight,
    }
    (output_dir / "training_state.json").write_text(json.dumps(payload, indent=2))


def _save_graph_encoder(
    output_dir: Path,
    *,
    step: int,
    graph_encoder: GraphSlotEncoder,
    optimizer: torch.optim.Optimizer,
) -> Path:
    checkpoint_dir = output_dir / f"checkpoint-{step:06d}"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    torch.save(graph_encoder.state_dict(), checkpoint_dir / "graph_encoder.pt")
    torch.save(optimizer.state_dict(), checkpoint_dir / "optimizer.pt")
    return checkpoint_dir


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Pretrain the SCOP-Depth graph encoder before full relation-aware diffusion training."
    )
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-id", type=str, default=DEFAULT_FLUX_MODEL_ID)
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    parser.add_argument("--mixed-precision", choices=("no", "fp16", "bf16"), default="fp16")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--graph-learning-rate", type=float, default=5e-4)
    parser.add_argument("--max-train-steps", type=int, default=1000)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--slot-dim", type=int, default=512)
    parser.add_argument("--gnn-layers", type=int, default=2)
    parser.add_argument("--save-every", type=int, default=500)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--eval-every", type=int, default=250)
    parser.add_argument("--limit-rows", type=int, default=None)
    parser.add_argument("--prompt-prefix", type=str, default="a photo of")
    parser.add_argument("--eval-fraction", type=float, default=0.1)
    parser.add_argument("--test-fraction", type=float, default=0.1)
    parser.add_argument("--position-loss-weight", type=float, default=1.0)
    parser.add_argument("--relation-loss-weight", type=float, default=1.0)
    parser.add_argument("--embedding-loss-weight", type=float, default=0.25)
    parser.add_argument("--inverse-relation-loss-weight", type=float, default=0.0)
    parser.add_argument("--box-loss-weight", type=float, default=0.0)
    parser.add_argument("--box3d-loss-weight", type=float, default=0.0)
    parser.add_argument("--disable-tqdm", action="store_true")
    return parser


def _compute_graph_batch_losses(
    *,
    batch: dict[str, object],
    tokenizer: object,
    text_encoder: object,
    graph_encoder: GraphSlotEncoder,
    device: str,
    position_loss_weight: float,
    relation_loss_weight: float,
    embedding_loss_weight: float,
    inverse_relation_loss_weight: float,
    box_loss_weight: float,
    box3d_loss_weight: float = 0.0,
) -> dict[str, torch.Tensor]:
    max_nodes = max(len(graph["nodes"]) for graph in batch["scene_graphs"])  # type: ignore[index]
    slot_targets, slot_mask = bbox_centers_after_crop(
        batch["metadata"],  # type: ignore[arg-type]
        batch["image_sizes"],  # type: ignore[arg-type]
        max_nodes=max_nodes,
        device=torch.device(device),
    )
    log_sigma_targets, _ = bbox_log_sigmas_after_crop(
        batch["metadata"],  # type: ignore[arg-type]
        batch["image_sizes"],  # type: ignore[arg-type]
        max_nodes=max_nodes,
        device=torch.device(device),
    )
    log_size_3d_targets, _ = bbox_log_sizes_3d_after_crop(
        batch["metadata"],  # type: ignore[arg-type]
        batch["image_sizes"],  # type: ignore[arg-type]
        max_nodes=max_nodes,
        device=torch.device(device),
    )
    scene_graph_batch = build_batched_scene_graphs(
        batch["scene_graphs"],  # type: ignore[arg-type]
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
    pooled_embeddings = pooled_label_embeddings(
        tokenizer=tokenizer,
        text_encoder=text_encoder,
        scene_graph_batch=scene_graph_batch,
        device=device,
        dtype=conditioning.slot_embeddings.dtype,
    )

    position_loss = F.smooth_l1_loss(
        conditioning.slot_positions[conditioning.slot_mask],
        slot_targets[conditioning.slot_mask],
    )
    edge_loss = relation_loss(
        conditioning.relation_logits,
        conditioning.slot_positions,
        scene_graph_batch,
    )
    semantic_loss = embedding_alignment_loss(
        conditioning.slot_embeddings,
        pooled_embeddings,
        conditioning.slot_mask,
    )
    box_loss = log_sigma_loss(
        conditioning.slot_log_sigmas,
        log_sigma_targets,
        conditioning.slot_mask,
    )
    box3d_loss = log_size_3d_loss(
        conditioning.slot_log_sizes_3d,
        log_size_3d_targets,
        conditioning.slot_mask,
    )
    inverse_loss = inverse_relation_regularizer(graph_encoder)
    total_loss = (
        position_loss_weight * position_loss
        + relation_loss_weight * edge_loss
        + embedding_loss_weight * semantic_loss
        + inverse_relation_loss_weight * inverse_loss
        + box_loss_weight * box_loss
        + box3d_loss_weight * box3d_loss
    )
    return {
        "loss": total_loss,
        "position_loss": position_loss,
        "relation_loss": edge_loss,
        "embedding_loss": semantic_loss,
        "inverse_relation_loss": inverse_loss,
        "box_loss": box_loss,
        "box3d_loss": box3d_loss,
    }


@torch.no_grad()
def _evaluate_graph_encoder(
    *,
    dataloader: DataLoader,
    tokenizer: object,
    text_encoder: object,
    graph_encoder: GraphSlotEncoder,
    device: str,
    position_loss_weight: float,
    relation_loss_weight: float,
    embedding_loss_weight: float,
    inverse_relation_loss_weight: float,
    box_loss_weight: float,
    box3d_loss_weight: float,
) -> dict[str, float]:
    graph_encoder.eval()
    totals = {
        "loss": 0.0,
        "position_loss": 0.0,
        "relation_loss": 0.0,
        "embedding_loss": 0.0,
        "inverse_relation_loss": 0.0,
        "box_loss": 0.0,
        "box3d_loss": 0.0,
    }
    batch_count = 0
    for batch in dataloader:
        metrics = _compute_graph_batch_losses(
            batch=batch,
            tokenizer=tokenizer,
            text_encoder=text_encoder,
            graph_encoder=graph_encoder,
            device=device,
            position_loss_weight=position_loss_weight,
            relation_loss_weight=relation_loss_weight,
            embedding_loss_weight=embedding_loss_weight,
            inverse_relation_loss_weight=inverse_relation_loss_weight,
            box_loss_weight=box_loss_weight,
            box3d_loss_weight=box3d_loss_weight,
        )
        for key in totals:
            totals[key] += float(metrics[key].item())
        batch_count += 1
    graph_encoder.train()
    if batch_count == 0:
        return {key: 0.0 for key in totals}
    return {key: value / batch_count for key, value in totals.items()}


def main() -> int:
    args = make_parser().parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    device = resolve_torch_device(args.device)
    weight_dtype = choose_weight_dtype(device, args.mixed_precision)
    set_seed(args.seed)
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

    datasets = build_dataset_splits(
        args.dataset_dir,
        image_size=512,
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
    test_dataloader = DataLoader(
        datasets["test"],
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_training_items,
    )

    from transformers import CLIPTextModel, CLIPTokenizer

    tokenizer = CLIPTokenizer.from_pretrained(args.model_id, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(
        args.model_id,
        subfolder="text_encoder",
        torch_dtype=weight_dtype,
    )
    text_encoder.requires_grad_(False)
    text_encoder.to(device)

    graph_encoder = GraphSlotEncoder(
        text_hidden_dim=text_encoder.config.hidden_size,
        slot_dim=args.slot_dim,
        num_layers=args.gnn_layers,
    ).to(device)
    optimizer = torch.optim.AdamW(
        graph_encoder.parameters(),
        lr=args.graph_learning_rate,
    )

    _save_state(args.output_dir, step=0, args=args)
    metrics_logger = MetricsLogger(
        args.output_dir,
        fieldnames=[
            "step",
            "split",
            "loss",
            "position_loss",
            "relation_loss",
            "embedding_loss",
            "inverse_relation_loss",
            "box_loss",
            "box3d_loss",
        ],
    )
    progress_bar = tqdm(
        total=args.max_train_steps,
        disable=is_tqdm_disabled(args),
        desc="GraphPretraining",
    )

    global_step = 0
    running = {
        "loss": 0.0,
        "position_loss": 0.0,
        "relation_loss": 0.0,
        "embedding_loss": 0.0,
        "inverse_relation_loss": 0.0,
        "box_loss": 0.0,
        "box3d_loss": 0.0,
    }
    running_steps = 0
    while global_step < args.max_train_steps:
        for batch in train_dataloader:
            metrics = _compute_graph_batch_losses(
                batch=batch,
                tokenizer=tokenizer,
                text_encoder=text_encoder,
                graph_encoder=graph_encoder,
                device=device,
                position_loss_weight=args.position_loss_weight,
                relation_loss_weight=args.relation_loss_weight,
                embedding_loss_weight=args.embedding_loss_weight,
                inverse_relation_loss_weight=args.inverse_relation_loss_weight,
                box_loss_weight=args.box_loss_weight,
                box3d_loss_weight=args.box3d_loss_weight,
            )
            loss = metrics["loss"]

            if not torch.isfinite(loss):
                raise FloatingPointError(
                    "Encountered a non-finite loss in graph pretraining."
                )

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            global_step += 1
            progress_bar.update(1)
            running_steps += 1
            for key in running:
                running[key] += float(metrics[key].item())

            if global_step % args.log_every == 0:
                train_log = {
                    "step": global_step,
                    "split": "train",
                    "loss": running["loss"] / running_steps,
                    "position_loss": running["position_loss"] / running_steps,
                    "relation_loss": running["relation_loss"] / running_steps,
                    "embedding_loss": running["embedding_loss"] / running_steps,
                    "inverse_relation_loss": running["inverse_relation_loss"] / running_steps,
                    "box_loss": running["box_loss"] / running_steps,
                    "box3d_loss": running["box3d_loss"] / running_steps,
                }
                metrics_logger.log(train_log)
                progress_bar.set_postfix(
                    pos=f"{train_log['position_loss']:.4f}",
                    rel=f"{train_log['relation_loss']:.4f}",
                    sem=f"{train_log['embedding_loss']:.4f}",
                    inv=f"{train_log['inverse_relation_loss']:.4f}",
                    box=f"{train_log['box_loss']:.4f}",
                    box3d=f"{train_log['box3d_loss']:.4f}",
                )
                running = {key: 0.0 for key in running}
                running_steps = 0

            if len(datasets["eval"]) > 0 and args.eval_every > 0 and global_step % args.eval_every == 0:
                eval_log = {
                    "step": global_step,
                    "split": "eval",
                    **_evaluate_graph_encoder(
                        dataloader=eval_dataloader,
                        tokenizer=tokenizer,
                        text_encoder=text_encoder,
                        graph_encoder=graph_encoder,
                        device=device,
                        position_loss_weight=args.position_loss_weight,
                        relation_loss_weight=args.relation_loss_weight,
                        embedding_loss_weight=args.embedding_loss_weight,
                        inverse_relation_loss_weight=args.inverse_relation_loss_weight,
                        box_loss_weight=args.box_loss_weight,
                        box3d_loss_weight=args.box3d_loss_weight,
                    ),
                }
                metrics_logger.log(eval_log)
                print(
                    "Eval at step "
                    f"{global_step}: loss={eval_log['loss']:.4f}, "
                    f"pos={eval_log['position_loss']:.4f}, "
                    f"rel={eval_log['relation_loss']:.4f}, "
                    f"sem={eval_log['embedding_loss']:.4f}, "
                    f"inv={eval_log['inverse_relation_loss']:.4f}, "
                    f"box={eval_log['box_loss']:.4f}, "
                    f"box3d={eval_log['box3d_loss']:.4f}"
                )

            if global_step % args.save_every == 0:
                checkpoint_dir = _save_graph_encoder(
                    args.output_dir,
                    step=global_step,
                    graph_encoder=graph_encoder,
                    optimizer=optimizer,
                )
                _save_state(args.output_dir, step=global_step, args=args)
                print(f"Saved graph pretraining checkpoint to {checkpoint_dir}")

            if global_step >= args.max_train_steps:
                break
        else:
            continue
        break

    final_dir = args.output_dir / "final"
    final_dir.mkdir(parents=True, exist_ok=True)
    torch.save(graph_encoder.state_dict(), final_dir / "graph_encoder.pt")
    if len(datasets["test"]) > 0:
        test_log = {
            "step": global_step,
            "split": "test",
            **_evaluate_graph_encoder(
                dataloader=test_dataloader,
                tokenizer=tokenizer,
                text_encoder=text_encoder,
                graph_encoder=graph_encoder,
                device=device,
                position_loss_weight=args.position_loss_weight,
                relation_loss_weight=args.relation_loss_weight,
                embedding_loss_weight=args.embedding_loss_weight,
                inverse_relation_loss_weight=args.inverse_relation_loss_weight,
                box_loss_weight=args.box_loss_weight,
                box3d_loss_weight=args.box3d_loss_weight,
            ),
        }
        metrics_logger.log(test_log)
        print(
            "Final test loss: "
            f"{test_log['loss']:.4f} "
            f"(pos={test_log['position_loss']:.4f}, "
            f"rel={test_log['relation_loss']:.4f}, "
            f"sem={test_log['embedding_loss']:.4f}, "
            f"inv={test_log['inverse_relation_loss']:.4f}, "
            f"box={test_log['box_loss']:.4f}, "
            f"box3d={test_log['box3d_loss']:.4f})"
        )
    _save_state(args.output_dir, step=global_step, args=args)
    print(f"Graph pretraining finished at step {global_step}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
