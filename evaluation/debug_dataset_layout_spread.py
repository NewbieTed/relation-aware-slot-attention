from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from training.graph_targets import bbox_minmax_3d_after_crop
from training.prompts import prompt_from_scop_depth_row


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Summarize target bbox spread in a SCOP-style metadata folder.")
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--prompt-filter", type=str, default=None)
    parser.add_argument("--prompt", type=str, default=None, help="Alias for --prompt-filter.")
    parser.add_argument("--limit-rows", type=int, default=None)
    parser.add_argument(
        "--coordinate-space",
        choices=("model", "unit"),
        default="model",
        help="Report centers in model space [-1,1] or raw unit cube [0,1]. Sizes stay in normalized box units.",
    )
    parser.add_argument("--list-samples", action="store_true")
    return parser


def load_rows(dataset_dir: Path) -> list[dict[str, Any]]:
    rows = []
    for line in (dataset_dir / "metadata.jsonl").read_text().splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


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


def labels_from_row(row: dict[str, Any]) -> list[str]:
    labels = []
    for index, annot in enumerate(row.get("annots", [])):
        labels.append(str(annot.get("category_name") or annot.get("category_id") or f"obj{index}"))
    return labels


def normalize_prompt(prompt: str) -> str:
    normalized = " ".join(prompt.split()).lower()
    for prefix in ("a photo of ", "an image of ", "a picture of "):
        if normalized.startswith(prefix):
            normalized = normalized[len(prefix) :]
            break
    return normalized


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
            print(f"      {index:04d}: center={short_vec(center)} size={short_vec(size)}")


def main() -> int:
    args = make_parser().parse_args()
    rows = load_rows(args.dataset_dir)
    prompt_filter = args.prompt_filter or args.prompt
    if prompt_filter:
        expected = normalize_prompt(prompt_filter)
        rows = [
            row
            for row in rows
            if normalize_prompt(prompt_from_scop_depth_row(row)) == expected
        ]
    if args.limit_rows is not None:
        rows = rows[: args.limit_rows]
    if not rows:
        raise ValueError("No rows matched the requested dataset/filter.")

    image_sizes = [tuple(row.get("image_size") or (512, 512)) for row in rows]
    boxes, mask = bbox_minmax_3d_after_crop(rows, image_sizes, max_nodes=2, device=torch.device("cpu"))
    centers = (boxes[..., :3] + boxes[..., 3:]) * 0.5
    sizes = boxes[..., 3:] - boxes[..., :3]
    if args.coordinate_space == "model":
        centers = centers.mul(2.0).sub(1.0)
    labels = labels_from_row(rows[0])

    print(f"Dataset: {args.dataset_dir}")
    print(f"Rows: {len(rows)}")
    if args.coordinate_space == "model":
        print("Coordinate space: model centers [-1, 1], sizes in normalized box units")
    else:
        print("Coordinate space: unit cube centers [0, 1], sizes in normalized box units")
    if prompt_filter:
        print(f"Prompt filter: {prompt_filter}")
    for object_index, label in enumerate(labels[:2]):
        valid = mask[:, object_index]
        print_object_summary(
            label,
            centers[valid, object_index, :],
            sizes[valid, object_index, :],
            list_samples=args.list_samples,
        )
    if len(labels) >= 2:
        valid = mask[:, 0] & mask[:, 1]
        delta = centers[valid, 1, :] - centers[valid, 0, :]
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
