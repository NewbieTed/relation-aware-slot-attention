from __future__ import annotations

import argparse
from collections import defaultdict
import copy
import json
import math
import os
import random
import shutil
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont

from scop_depth.coco_categories import COCO_CATEGORY_ID_TO_NAME
from scop_depth.prompt_graph import scene_graph_from_scop_depth_row


RELATION_ALIASES = {
    "left": "left_of",
    "to the left of": "left_of",
    "left_of": "left_of",
    "right": "right_of",
    "to the right of": "right_of",
    "right_of": "right_of",
    "above": "above",
    "on the top of": "above",
    "below": "below",
    "on the bottom of": "below",
    "in front of": "in_front_of",
    "in_front_of": "in_front_of",
    "behind": "behind",
    "hidden by": "behind",
}
SUPPORTED_RELATIONS = {"left_of", "right_of", "above", "below", "in_front_of", "behind"}


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Create a SCOP-Depth-compatible derived dataset with symlinked "
            "images and relation-preserving augmented 3D box targets."
        )
    )
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--variants-per-row", type=int, default=4)
    parser.add_argument("--limit-rows", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-gap", type=float, default=0.18)
    parser.add_argument("--max-gap", type=float, default=0.70)
    parser.add_argument("--center-low", type=float, default=0.12)
    parser.add_argument("--center-high", type=float, default=0.88)
    parser.add_argument("--size-jitter", type=float, default=0.20)
    parser.add_argument("--min-size", type=float, default=0.08)
    parser.add_argument("--max-size", type=float, default=0.70)
    parser.add_argument("--empirical-weight", type=float, default=0.50)
    parser.add_argument("--original-jitter-weight", type=float, default=0.30)
    parser.add_argument("--synthetic-weight", type=float, default=0.20)
    parser.add_argument("--empirical-jitter", type=float, default=0.06)
    parser.add_argument("--original-center-jitter", type=float, default=0.08)
    parser.add_argument("--copy-images", action="store_true")
    parser.add_argument("--num-samples", type=int, default=24)
    parser.add_argument("--progress-every", type=int, default=1000)
    return parser


def load_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def dump_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text("\n".join(json.dumps(row, separators=(",", ":")) for row in rows) + "\n")


def relation_from_row(row: dict[str, Any]) -> tuple[int, int, str] | None:
    try:
        graph = scene_graph_from_scop_depth_row(row)
    except (KeyError, ValueError, TypeError):
        return None
    node_id_to_index = {node.id: index for index, node in enumerate(graph.nodes)}
    for edge in graph.edges:
        relation = "behind" if edge.relation == "hidden_by" else edge.relation
        if relation not in SUPPORTED_RELATIONS:
            continue
        source_index = node_id_to_index[edge.source_id]
        target_index = node_id_to_index[edge.target_id]
        if source_index != target_index:
            return source_index, target_index, relation
    return None


def _same_bbox(left: list[float] | tuple[float, ...], right: list[float] | tuple[float, ...]) -> bool:
    return len(left) == len(right) and all(abs(float(a) - float(b)) < 1e-4 for a, b in zip(left, right))


def canonicalize_row(row: dict[str, Any]) -> dict[str, Any] | None:
    try:
        graph = scene_graph_from_scop_depth_row(row)
    except (KeyError, ValueError, TypeError):
        return None

    ordered_indices: list[int] = []
    used_indices: set[int] = set()
    for node in graph.nodes:
        match_index = None
        for index, annot in enumerate(row.get("annots", [])):
            if index in used_indices:
                continue
            if int(annot.get("category_id", -1)) != int(node.category_id):
                continue
            if node.bbox is not None and not _same_bbox(annot.get("bbox", []), node.bbox):
                continue
            match_index = index
            break
        if match_index is None:
            return None
        used_indices.add(match_index)
        ordered_indices.append(match_index)

    if len(ordered_indices) != 2:
        return None

    new_row = copy.deepcopy(row)
    new_row["annots"] = [copy.deepcopy(row["annots"][index]) for index in ordered_indices]
    if row.get("depth"):
        new_depth = copy.deepcopy(row["depth"])
        for new_index, old_index in enumerate(ordered_indices):
            old_key = f"bbox{old_index + 1}"
            new_key = f"bbox{new_index + 1}"
            if old_key in row["depth"]:
                new_depth[new_key] = copy.deepcopy(row["depth"][old_key])
        new_row["depth"] = new_depth
    return new_row


def original_box_01(row: dict[str, Any], node_index: int) -> tuple[float, float, float, float, float, float]:
    width, height = row.get("image_size") or [1, 1]
    crop_size = float(min(width, height))
    x, y, w, h = [float(v) for v in row["annots"][node_index]["bbox"]]
    depth = row.get("depth") or {}
    stats = depth.get(f"bbox{node_index + 1}", {})
    z_center = float(stats.get("median", 0.5))
    z_size = float(stats.get("iqr", stats.get("std", max((w + h) / crop_size * 0.25, 0.08))))
    cx = max(0.0, min(1.0, (x + w * 0.5) / crop_size))
    cy = max(0.0, min(1.0, (y + h * 0.5) / crop_size))
    sx = max(w / crop_size, 0.08)
    sy = max(h / crop_size, 0.08)
    sz = max(z_size, 0.08)
    return cx, cy, z_center, sx, sy, sz


def category_key(row: dict[str, Any], node_index: int) -> str:
    annot = row["annots"][node_index]
    return str(annot.get("category_id", annot.get("category_name", f"node{node_index}")))


def category_label(row: dict[str, Any], node_index: int) -> str:
    annot = row["annots"][node_index]
    if annot.get("category_name"):
        return str(annot["category_name"])
    category_id = annot.get("category_id")
    return COCO_CATEGORY_ID_TO_NAME.get(int(category_id), f"obj{node_index}") if category_id is not None else f"obj{node_index}"


def collect_empirical_stats(rows: list[dict[str, Any]]) -> dict[str, Any]:
    category_stats: dict[str, dict[str, list[list[float]]]] = defaultdict(lambda: {"centers": [], "sizes": []})
    relation_stats: dict[str, dict[str, list[list[float]]]] = defaultdict(lambda: {"deltas": []})
    relation_counts: dict[str, int] = defaultdict(int)
    category_counts: dict[str, int] = defaultdict(int)

    for row in rows:
        relation_info = relation_from_row(row)
        if relation_info is None:
            continue
        source_index, target_index, relation = relation_info
        boxes = [original_box_01(row, 0), original_box_01(row, 1)]
        for node_index, box in enumerate(boxes):
            key = category_key(row, node_index)
            category_stats[key]["centers"].append(list(box[:3]))
            category_stats[key]["sizes"].append(list(box[3:]))
            category_counts[key] += 1
        source_center = boxes[source_index][:3]
        target_center = boxes[target_index][:3]
        relation_stats[relation]["deltas"].append([target_center[i] - source_center[i] for i in range(3)])
        relation_counts[relation] += 1

    return {
        "category_stats": category_stats,
        "relation_stats": relation_stats,
        "category_counts": dict(category_counts),
        "relation_counts": dict(relation_counts),
    }


def jitter_size(base: float, rng: random.Random, *, jitter: float, min_size: float, max_size: float) -> float:
    factor = math.exp(rng.gauss(0.0, jitter))
    return max(min_size, min(max_size, base * factor))


def clamp_center(value: float, *, low: float, high: float) -> float:
    return max(low, min(high, value))


def shifted_into_bounds(source: list[float], target: list[float], *, low: float, high: float) -> tuple[list[float], list[float]]:
    source = list(source)
    target = list(target)
    for axis in range(3):
        min_value = min(source[axis], target[axis])
        max_value = max(source[axis], target[axis])
        if min_value < low:
            shift = low - min_value
            source[axis] += shift
            target[axis] += shift
        if max_value > high:
            shift = high - max_value
            source[axis] += shift
            target[axis] += shift
        source[axis] = clamp_center(source[axis], low=low, high=high)
        target[axis] = clamp_center(target[axis], low=low, high=high)
    return source, target


def relation_axis_and_sign(relation: str) -> tuple[int, float]:
    if relation in {"left_of", "above", "behind"}:
        return {"left_of": 0, "above": 1, "behind": 2}[relation], 1.0
    return {"right_of": 0, "below": 1, "in_front_of": 2}[relation], -1.0


def enforce_relation_gap(
    source: list[float],
    target: list[float],
    relation: str,
    rng: random.Random,
    args: argparse.Namespace,
    *,
    preferred_gap: float | None = None,
) -> tuple[list[float], list[float]]:
    axis, sign = relation_axis_and_sign(relation)
    gap = preferred_gap if preferred_gap is not None else rng.uniform(args.min_gap, args.max_gap)
    gap = max(args.min_gap, min(args.max_gap, abs(gap)))
    midpoint = (source[axis] + target[axis]) * 0.5
    midpoint = clamp_center(midpoint, low=args.center_low + gap * 0.5, high=args.center_high - gap * 0.5)
    source[axis] = midpoint - sign * gap * 0.5
    target[axis] = midpoint + sign * gap * 0.5
    return shifted_into_bounds(source, target, low=args.center_low, high=args.center_high)


def sample_pair_centers(
    relation: str,
    rng: random.Random,
    *,
    min_gap: float,
    max_gap: float,
    center_low: float,
    center_high: float,
) -> tuple[list[float], list[float]]:
    a = [rng.uniform(center_low, center_high) for _ in range(3)]
    b = [rng.uniform(center_low, center_high) for _ in range(3)]
    axis = {"left_of": 0, "right_of": 0, "above": 1, "below": 1, "in_front_of": 2, "behind": 2}[relation]
    gap = rng.uniform(min_gap, max_gap)
    midpoint = rng.uniform(center_low + gap * 0.5, center_high - gap * 0.5)
    low_value = midpoint - gap * 0.5
    high_value = midpoint + gap * 0.5
    if relation in {"left_of", "above", "behind"}:
        a[axis], b[axis] = low_value, high_value
    else:
        a[axis], b[axis] = high_value, low_value
    return a, b


def sample_vector(values: list[list[float]], fallback: list[float], rng: random.Random) -> list[float]:
    if not values:
        return list(fallback)
    return list(rng.choice(values))


def sample_sizes(row: dict[str, Any], stats: dict[str, Any], rng: random.Random, args: argparse.Namespace) -> list[list[float]]:
    category_stats = stats["category_stats"]
    sizes: list[list[float]] = []
    for node_index in range(2):
        fallback = list(original_box_01(row, node_index)[3:])
        key = category_key(row, node_index)
        base = sample_vector(category_stats.get(key, {}).get("sizes", []), fallback, rng)
        sizes.append(
            [
                jitter_size(base[axis], rng, jitter=args.size_jitter, min_size=args.min_size, max_size=args.max_size)
                for axis in range(3)
            ]
        )
    return sizes


def sample_synthetic_layout(
    row: dict[str, Any],
    relation: str,
    rng: random.Random,
    args: argparse.Namespace,
    stats: dict[str, Any],
) -> tuple[list[list[float]], list[list[float]], str]:
    source_center, target_center = sample_pair_centers(
        relation,
        rng,
        min_gap=args.min_gap,
        max_gap=args.max_gap,
        center_low=args.center_low,
        center_high=args.center_high,
    )
    return [source_center, target_center], sample_sizes(row, stats, rng, args), "synthetic"


def sample_original_jitter_layout(
    row: dict[str, Any],
    source_index: int,
    target_index: int,
    relation: str,
    rng: random.Random,
    args: argparse.Namespace,
    stats: dict[str, Any],
) -> tuple[list[list[float]], list[list[float]], str]:
    original = [original_box_01(row, 0), original_box_01(row, 1)]
    centers = [
        [
            clamp_center(original[node_index][axis] + rng.gauss(0.0, args.original_center_jitter), low=args.center_low, high=args.center_high)
            for axis in range(3)
        ]
        for node_index in range(2)
    ]
    axis, _ = relation_axis_and_sign(relation)
    original_gap = abs(original[target_index][axis] - original[source_index][axis])
    preferred_gap = original_gap * math.exp(rng.gauss(0.0, args.empirical_jitter))
    centers[source_index], centers[target_index] = enforce_relation_gap(
        centers[source_index],
        centers[target_index],
        relation,
        rng,
        args,
        preferred_gap=preferred_gap,
    )
    return centers, sample_sizes(row, stats, rng, args), "original_jitter"


def sample_empirical_layout(
    row: dict[str, Any],
    source_index: int,
    target_index: int,
    relation: str,
    rng: random.Random,
    args: argparse.Namespace,
    stats: dict[str, Any],
) -> tuple[list[list[float]], list[list[float]], str]:
    category_stats = stats["category_stats"]
    relation_deltas = stats["relation_stats"].get(relation, {}).get("deltas", [])
    original = [original_box_01(row, 0), original_box_01(row, 1)]
    centers = [
        sample_vector(category_stats.get(category_key(row, node_index), {}).get("centers", []), list(original[node_index][:3]), rng)
        for node_index in range(2)
    ]
    delta = sample_vector(
        relation_deltas,
        [original[target_index][axis] - original[source_index][axis] for axis in range(3)],
        rng,
    )
    delta = [value + rng.gauss(0.0, args.empirical_jitter) for value in delta]
    source_center = centers[source_index]
    target_center = [source_center[axis] + delta[axis] for axis in range(3)]
    source_center, target_center = shifted_into_bounds(
        source_center,
        target_center,
        low=args.center_low,
        high=args.center_high,
    )
    axis, sign = relation_axis_and_sign(relation)
    if sign * (target_center[axis] - source_center[axis]) < args.min_gap:
        source_center, target_center = enforce_relation_gap(source_center, target_center, relation, rng, args)
    centers[source_index] = source_center
    centers[target_index] = target_center
    return centers, sample_sizes(row, stats, rng, args), "empirical"


def choose_layout(
    row: dict[str, Any],
    source_index: int,
    target_index: int,
    relation: str,
    rng: random.Random,
    args: argparse.Namespace,
    stats: dict[str, Any],
) -> tuple[list[list[float]], list[list[float]], str]:
    weights = [
        ("empirical", max(0.0, args.empirical_weight)),
        ("original_jitter", max(0.0, args.original_jitter_weight)),
        ("synthetic", max(0.0, args.synthetic_weight)),
    ]
    total = sum(weight for _, weight in weights)
    draw = rng.uniform(0.0, total if total > 0 else 1.0)
    running = 0.0
    mode = "synthetic"
    for candidate, weight in weights:
        running += weight
        if draw <= running:
            mode = candidate
            break

    if mode == "empirical":
        return sample_empirical_layout(row, source_index, target_index, relation, rng, args, stats)
    if mode == "original_jitter":
        return sample_original_jitter_layout(row, source_index, target_index, relation, rng, args, stats)
    centers, sizes, name = sample_synthetic_layout(row, relation, rng, args, stats)
    ordered_centers = [None, None]
    ordered_centers[source_index] = centers[0]
    ordered_centers[target_index] = centers[1]
    return ordered_centers, sizes, name  # type: ignore[return-value]


def center_size_to_bbox(center: list[float], size: list[float], image_size: list[int]) -> tuple[list[float], dict[str, float]]:
    crop_size = float(min(image_size))
    sx = min(size[0], 2.0 * min(center[0], 1.0 - center[0]))
    sy = min(size[1], 2.0 * min(center[1], 1.0 - center[1]))
    sz = min(size[2], 2.0 * min(center[2], 1.0 - center[2]))
    sx = max(sx, 0.03)
    sy = max(sy, 0.03)
    sz = max(sz, 0.03)
    x = (center[0] - sx * 0.5) * crop_size
    y = (center[1] - sy * 0.5) * crop_size
    w = sx * crop_size
    h = sy * crop_size
    depth = {
        "median": center[2],
        "iqr": sz,
        "std": sz * 0.5,
        "mad": sz * 0.25,
        "range": sz,
    }
    return [x, y, w, h], depth


def satisfies_relation(row: dict[str, Any], source_index: int, target_index: int, relation: str, margin: float) -> bool:
    source = original_box_01(row, source_index)
    target = original_box_01(row, target_index)
    delta = [target[i] - source[i] for i in range(3)]
    if relation == "left_of":
        return delta[0] >= margin
    if relation == "right_of":
        return delta[0] <= -margin
    if relation == "above":
        return delta[1] >= margin
    if relation == "below":
        return delta[1] <= -margin
    if relation == "in_front_of":
        return delta[2] <= -margin
    if relation == "behind":
        return delta[2] >= margin
    return False


def augment_row(
    row: dict[str, Any],
    rng: random.Random,
    args: argparse.Namespace,
    variant_index: int,
    stats: dict[str, Any],
) -> dict[str, Any] | None:
    relation_info = relation_from_row(row)
    if relation_info is None:
        return None
    source_index, target_index, relation = relation_info
    image_size = row.get("image_size")
    if image_size is None:
        return None

    centers, sizes, sample_mode = choose_layout(row, source_index, target_index, relation, rng, args, stats)

    new_row = copy.deepcopy(row)
    new_row["augmented_layout"] = {
        "source_dataset": str(args.input_dir),
        "source_seq": row.get("seq"),
        "variant_index": variant_index,
        "relation": relation,
        "source_index": source_index,
        "target_index": target_index,
        "sample_mode": sample_mode,
        "source_category": category_key(row, source_index),
        "target_category": category_key(row, target_index),
    }
    new_depth = copy.deepcopy(row.get("depth") or {})
    for node_index in range(2):
        bbox, depth_stats = center_size_to_bbox(centers[node_index], sizes[node_index], image_size)
        new_row["annots"][node_index]["bbox"] = bbox
        new_depth[f"bbox{node_index + 1}"] = {
            **new_depth.get(f"bbox{node_index + 1}", {}),
            **depth_stats,
        }
    new_row["depth"] = new_depth
    if not satisfies_relation(new_row, source_index, target_index, relation, args.min_gap * 0.8):
        return None
    return new_row


def is_augmentable_row(row: dict[str, Any]) -> bool:
    canonical_row = canonicalize_row(row)
    return (
        canonical_row is not None
        and relation_from_row(canonical_row) is not None
        and canonical_row.get("image_size") is not None
        and len(canonical_row.get("annots", [])) >= 2
    )


def maybe_print_progress(message: str, current: int, total: int | None, every: int) -> None:
    if every <= 0:
        return
    if current != 1 and current % every != 0 and (total is None or current != total):
        return
    suffix = f"/{total}" if total is not None else ""
    print(f"{message}: {current}{suffix}", flush=True)


def link_or_copy_image(input_dir: Path, output_dir: Path, file_name: str, *, copy_images: bool) -> None:
    source = input_dir / file_name
    target = output_dir / file_name
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() or target.is_symlink():
        return
    if copy_images:
        shutil.copy2(source, target)
    else:
        os.symlink(source, target)


def draw_sample(row: dict[str, Any], dataset_dir: Path, output_path: Path) -> None:
    image = Image.open(dataset_dir / row["file_name"]).convert("RGB").resize((512, 512), Image.Resampling.BICUBIC)
    draw = ImageDraw.Draw(image, "RGBA")
    font = ImageFont.load_default()
    width, height = row.get("image_size") or [512, 512]
    scale_x = 512 / float(width)
    scale_y = 512 / float(height)
    colors = [(0, 180, 255, 220), (255, 60, 120, 220)]
    for index, annot in enumerate(row["annots"]):
        x, y, w, h = annot["bbox"]
        rect = [x * scale_x, y * scale_y, (x + w) * scale_x, (y + h) * scale_y]
        draw.rectangle(rect, outline=colors[index], width=4)
        label = category_label(row, index)
        depth = row.get("depth", {}).get(f"bbox{index + 1}", {}).get("median", 0.5)
        draw.text((rect[0] + 4, rect[1] + 4), f"{label} z={depth:.2f}", fill=(255, 255, 255, 255), font=font)
    lines = [" / ".join(" ".join(map(str, oro)) for oro in row.get("oros", []))]
    aug = row.get("augmented_layout", {})
    if aug:
        lines.append(f"aug rel={aug.get('relation')} variant={aug.get('variant_index')}")
    text = "\n".join(lines)
    box = draw.multiline_textbbox((0, 0), text, font=font)
    draw.rectangle([0, 0, box[2] + 12, box[3] + 12], fill=(0, 0, 0, 180))
    draw.multiline_text((6, 6), text, fill=(255, 255, 255, 255), font=font)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)


def main() -> int:
    args = make_parser().parse_args()
    rng = random.Random(args.seed)
    all_rows = load_rows(args.input_dir / "metadata.jsonl")
    rows: list[dict[str, Any]] = []
    for raw_index, row in enumerate(all_rows, start=1):
        canonical_row = canonicalize_row(row)
        if canonical_row is None or not is_augmentable_row(canonical_row):
            maybe_print_progress("Selecting usable rows", raw_index, len(all_rows), args.progress_every)
            continue
        rows.append(canonical_row)
        maybe_print_progress("Selecting usable rows", raw_index, len(all_rows), args.progress_every)
        if args.limit_rows is not None and len(rows) >= args.limit_rows:
            break
    stats_rows = []
    for raw_index, row in enumerate(all_rows, start=1):
        canonical_row = canonicalize_row(row)
        if canonical_row is not None and is_augmentable_row(canonical_row):
            stats_rows.append(canonical_row)
        maybe_print_progress("Collecting stats rows", raw_index, len(all_rows), args.progress_every)
    stats = collect_empirical_stats(stats_rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    augmented_rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for row_index, row in enumerate(rows):
        link_or_copy_image(args.input_dir, args.output_dir, row["file_name"], copy_images=args.copy_images)
        for variant_index in range(args.variants_per_row):
            new_row = augment_row(row, rng, args, variant_index, stats)
            if new_row is None:
                failures.append({"row_index": row_index, "seq": row.get("seq")})
                continue
            augmented_rows.append(new_row)
        maybe_print_progress("Augmenting usable rows", row_index + 1, len(rows), args.progress_every)

    if not augmented_rows:
        raise RuntimeError("No augmented rows were created")
    dump_rows(args.output_dir / "metadata.jsonl", augmented_rows)

    checked = 0
    failed_checks = []
    for row_index, row in enumerate(augmented_rows):
        relation_info = relation_from_row(row)
        if relation_info is None:
            failed_checks.append({"row_index": row_index, "reason": "missing_relation"})
            continue
        source_index, target_index, relation = relation_info
        if not satisfies_relation(row, source_index, target_index, relation, args.min_gap * 0.8):
            failed_checks.append({"row_index": row_index, "reason": "relation_violation"})
        checked += 1
        maybe_print_progress("Checking augmented rows", row_index + 1, len(augmented_rows), args.progress_every)

    sample_dir = args.output_dir / "samples"
    for sample_index, row in enumerate(rng.sample(augmented_rows, min(args.num_samples, len(augmented_rows)))):
        draw_sample(row, args.output_dir, sample_dir / f"sample_{sample_index:03d}.jpg")

    report = {
        "input_dir": str(args.input_dir),
        "output_dir": str(args.output_dir),
        "source_rows_total": len(all_rows),
        "usable_input_rows": len(rows),
        "stats_rows": len(stats_rows),
        "variants_per_row": args.variants_per_row,
        "augmented_rows": len(augmented_rows),
        "sampling_weights": {
            "empirical": args.empirical_weight,
            "original_jitter": args.original_jitter_weight,
            "synthetic": args.synthetic_weight,
        },
        "relation_counts": stats["relation_counts"],
        "category_count": len(stats["category_counts"]),
        "failed_generation_attempts": len(failures),
        "checked_rows": checked,
        "failed_checks": failed_checks[:100],
        "failed_check_count": len(failed_checks),
        "copy_images": args.copy_images,
    }
    (args.output_dir / "augmentation_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    return 0 if not failed_checks else 1


if __name__ == "__main__":
    raise SystemExit(main())
