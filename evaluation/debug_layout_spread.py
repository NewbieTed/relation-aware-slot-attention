from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch

from evaluation.prompt_parser import parse_prompt_to_scene_graph
from training.dataset import load_metadata_rows
from training.graph_targets import bbox_minmax_3d_after_crop
from training.graph_modules import build_slot_conditioning
from training.prompts import prompt_from_scop_depth_row, scene_graph_payload_from_row
from training.runtime import (
    DEFAULT_FLUX_MODEL_ID,
    infer_graph_encoder_config,
    infer_text_encoder_type,
    load_graph_encoder,
    load_graph_label_encoder,
    normalize_graph_encoder_state_dict,
    resolve_torch_device,
)
from training.scene_graph import build_batched_scene_graphs


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Summarize stochastic GNN layout spread for prompts.")
    parser.add_argument("--prompt", action="append", default=[], help="Prompt to inspect. Can be repeated.")
    parser.add_argument("--prompt-file", type=Path, default=None)
    parser.add_argument("--dataset-dir", type=Path, default=None, help="SCOP-style metadata folder for posterior-mode debugging.")
    parser.add_argument("--graph-encoder-path", type=Path, required=True)
    parser.add_argument("--model-id", type=str, default=DEFAULT_FLUX_MODEL_ID)
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    parser.add_argument("--layout-sample-mode", choices=("prior_mean", "prior_sample", "posterior"), default="prior_sample")
    parser.add_argument("--num-layout-samples", type=int, default=32)
    parser.add_argument("--layout-seed", type=int, default=42)
    parser.add_argument("--layout-z-scale", type=float, default=1.0)
    parser.add_argument("--posterior-rows", type=int, default=64, help="Maximum matching dataset rows to use in posterior mode.")
    parser.add_argument("--posterior-samples-per-row", type=int, default=1)
    parser.add_argument("--list-samples", action="store_true")
    return parser


def read_prompts(args: argparse.Namespace) -> list[str]:
    prompts = list(args.prompt)
    if args.prompt_file is not None:
        prompts.extend(line.strip() for line in args.prompt_file.read_text().splitlines() if line.strip())
    if not prompts:
        raise ValueError("Pass at least one --prompt or --prompt-file.")
    return prompts


def short_vec(values: torch.Tensor) -> str:
    return "[" + ", ".join(f"{float(value):+.4f}" for value in values.detach().cpu()) + "]"


def summarize_tensor(values: torch.Tensor) -> dict[str, torch.Tensor]:
    values = values.detach().cpu().to(torch.float32)
    return {
        "mean": values.mean(dim=0),
        "std": values.std(dim=0, unbiased=False),
        "min": values.min(dim=0).values,
        "max": values.max(dim=0).values,
        "range": values.max(dim=0).values - values.min(dim=0).values,
    }


def normalize_prompt(prompt: str) -> str:
    return " ".join(prompt.split()).lower()


def prompt_match_candidates(prompt: str) -> set[str]:
    normalized = normalize_prompt(prompt)
    candidates = {normalized}
    if not normalized.startswith("a photo of "):
        candidates.add(normalize_prompt(f"a photo of {prompt}"))
    return candidates


def rows_for_prompt(dataset_dir: Path, prompt: str, *, limit: int | None) -> list[dict[str, Any]]:
    expected = prompt_match_candidates(prompt)
    rows = [
        row
        for row in load_metadata_rows(dataset_dir)
        if normalize_prompt(prompt_from_scop_depth_row(row)) in expected
    ]
    if limit is not None:
        rows = rows[:limit]
    if not rows:
        raise ValueError(f"No rows in {dataset_dir} matched prompt: {prompt}")
    return rows


@torch.no_grad()
def predict_once(
    *,
    prompt: str,
    tokenizer: object,
    text_encoder: object,
    graph_encoder: torch.nn.Module,
    device: str,
    layout_sample_mode: str,
    layout_z_scale: float,
) -> tuple[list[str], torch.Tensor, torch.Tensor]:
    scene_graph = parse_prompt_to_scene_graph(prompt)
    node_count = len(scene_graph["nodes"])
    slot_targets = torch.zeros(1, node_count, 3, device=torch.device(device))
    slot_mask = torch.ones(1, node_count, device=torch.device(device), dtype=torch.bool)
    batched_graph = build_batched_scene_graphs(
        [scene_graph],
        slot_targets=slot_targets,
        slot_mask=slot_mask,
    )
    conditioning = build_slot_conditioning(
        tokenizer=tokenizer,
        text_encoder=text_encoder,
        scene_graph_batch=batched_graph,
        graph_encoder=graph_encoder,
        device=device,
        layout_sample_mode=layout_sample_mode,
        layout_z_scale=layout_z_scale,
    )
    labels = [str(node["label"]) for node in scene_graph["nodes"]]
    centers = conditioning.slot_positions[0, :node_count].detach().cpu().to(torch.float32)
    sizes = conditioning.slot_log_sizes_3d[0, :node_count].detach().cpu().to(torch.float32).exp()
    return labels, centers, sizes


@torch.no_grad()
def predict_posterior_row(
    *,
    row: dict[str, Any],
    tokenizer: object,
    text_encoder: object,
    graph_encoder: torch.nn.Module,
    device: str,
    layout_z_scale: float,
) -> tuple[list[str], torch.Tensor, torch.Tensor]:
    scene_graph = scene_graph_payload_from_row(row)
    node_count = len(scene_graph["nodes"])
    image_size = tuple(row.get("image_size") or (512, 512))
    box_targets, slot_mask = bbox_minmax_3d_after_crop(
        [row],
        [image_size],
        max_nodes=node_count,
        device=torch.device(device),
    )
    centers_01 = (box_targets[..., :3] + box_targets[..., 3:]) * 0.5
    sizes_01 = (box_targets[..., 3:] - box_targets[..., :3]).clamp_min(0.03)
    slot_targets = centers_01.mul(2.0).sub(1.0)
    log_size_targets = sizes_01.log()
    batched_graph = build_batched_scene_graphs(
        [scene_graph],
        slot_targets=slot_targets,
        slot_mask=slot_mask,
        log_size_targets=log_size_targets,
        box_targets=box_targets,
    )
    conditioning = build_slot_conditioning(
        tokenizer=tokenizer,
        text_encoder=text_encoder,
        scene_graph_batch=batched_graph,
        graph_encoder=graph_encoder,
        device=device,
        layout_sample_mode="posterior",
        layout_z_scale=layout_z_scale,
    )
    labels = [str(node["label"]) for node in scene_graph["nodes"]]
    centers = conditioning.slot_positions[0, :node_count].detach().cpu().to(torch.float32)
    sizes = conditioning.slot_log_sizes_3d[0, :node_count].detach().cpu().to(torch.float32).exp()
    return labels, centers, sizes


def print_object_summary(label: str, centers: torch.Tensor, sizes: torch.Tensor, *, list_samples: bool) -> None:
    center_stats = summarize_tensor(centers)
    size_stats = summarize_tensor(sizes)
    print(f"  {label}")
    print(f"    center mean:  {short_vec(center_stats['mean'])}")
    print(f"    center std:   {short_vec(center_stats['std'])}")
    print(f"    center min:   {short_vec(center_stats['min'])}")
    print(f"    center max:   {short_vec(center_stats['max'])}")
    print(f"    center range: {short_vec(center_stats['range'])}")
    print(f"    size mean:    {short_vec(size_stats['mean'])}")
    print(f"    size std:     {short_vec(size_stats['std'])}")
    print(f"    size min:     {short_vec(size_stats['min'])}")
    print(f"    size max:     {short_vec(size_stats['max'])}")
    print(f"    size range:   {short_vec(size_stats['range'])}")
    if list_samples:
        print("    samples:")
        for index, (center, size) in enumerate(zip(centers, sizes)):
            print(f"      {index:02d}: center={short_vec(center)} size={short_vec(size)}")


def main() -> int:
    args = make_parser().parse_args()
    prompts = read_prompts(args)
    device = resolve_torch_device(args.device)

    state_dict = normalize_graph_encoder_state_dict(torch.load(args.graph_encoder_path, map_location="cpu"))
    (
        _slot_dim,
        text_hidden_dim,
        _gnn_layers,
        _layout_mode,
        _latent_dim,
        _decoder_mode,
        _decoder_box_residual,
        _decoder_film_scale,
        _use_scene_latent,
    ) = infer_graph_encoder_config(state_dict)
    text_encoder_type = infer_text_encoder_type(text_hidden_dim)
    tokenizer, text_encoder, encoder_hidden_dim = load_graph_label_encoder(
        model_id=args.model_id,
        text_encoder_type=text_encoder_type,
        torch_dtype=torch.float32,
        device=device,
    )
    graph_encoder = load_graph_encoder(
        path=args.graph_encoder_path,
        text_hidden_dim=encoder_hidden_dim,
        device=device,
        dtype=text_encoder.dtype,
    )
    graph_encoder.eval()

    for prompt_index, prompt in enumerate(prompts):
        all_centers: list[torch.Tensor] = []
        all_sizes: list[torch.Tensor] = []
        labels: list[str] | None = None
        if args.layout_sample_mode == "posterior":
            if args.dataset_dir is None:
                raise ValueError("--dataset-dir is required for --layout-sample-mode posterior")
            rows = rows_for_prompt(args.dataset_dir, prompt, limit=args.posterior_rows)
            for row_index, row in enumerate(rows):
                for sample_index in range(max(1, args.posterior_samples_per_row)):
                    sample_seed = int(args.layout_seed) + prompt_index * 100000 + row_index * 1000 + sample_index
                    torch.manual_seed(sample_seed)
                    if torch.cuda.is_available():
                        torch.cuda.manual_seed_all(sample_seed)
                    labels, centers, sizes = predict_posterior_row(
                        row=row,
                        tokenizer=tokenizer,
                        text_encoder=text_encoder,
                        graph_encoder=graph_encoder,
                        device=device,
                        layout_z_scale=args.layout_z_scale,
                    )
                    all_centers.append(centers)
                    all_sizes.append(sizes)
        else:
            for sample_index in range(max(1, args.num_layout_samples)):
                sample_seed = int(args.layout_seed) + prompt_index * 1000 + sample_index
                if args.layout_sample_mode == "prior_sample":
                    torch.manual_seed(sample_seed)
                    if torch.cuda.is_available():
                        torch.cuda.manual_seed_all(sample_seed)
                labels, centers, sizes = predict_once(
                    prompt=prompt,
                    tokenizer=tokenizer,
                    text_encoder=text_encoder,
                    graph_encoder=graph_encoder,
                    device=device,
                    layout_sample_mode=args.layout_sample_mode,
                    layout_z_scale=args.layout_z_scale,
                )
                all_centers.append(centers)
                all_sizes.append(sizes)

        if labels is None:
            continue
        centers_tensor = torch.stack(all_centers, dim=0)
        sizes_tensor = torch.stack(all_sizes, dim=0)
        print("")
        print(f"Prompt {prompt_index}: {prompt}")
        print(
            f"  mode={args.layout_sample_mode}, samples={centers_tensor.shape[0]}, "
            f"seed={args.layout_seed}, z_scale={args.layout_z_scale}"
        )
        if args.layout_sample_mode == "posterior":
            print(
                f"  posterior_rows={min(args.posterior_rows, centers_tensor.shape[0])}, "
                f"samples_per_row={args.posterior_samples_per_row}"
            )
        for object_index, label in enumerate(labels):
            print_object_summary(
                label,
                centers_tensor[:, object_index, :],
                sizes_tensor[:, object_index, :],
                list_samples=args.list_samples,
            )
        if len(labels) >= 2:
            delta = centers_tensor[:, 1, :] - centers_tensor[:, 0, :]
            delta_stats = summarize_tensor(delta)
            print("  object1 - object0 center delta")
            print(f"    mean:  {short_vec(delta_stats['mean'])}")
            print(f"    std:   {short_vec(delta_stats['std'])}")
            print(f"    min:   {short_vec(delta_stats['min'])}")
            print(f"    max:   {short_vec(delta_stats['max'])}")
            print(f"    range: {short_vec(delta_stats['range'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
