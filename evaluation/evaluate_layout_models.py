from __future__ import annotations

import argparse
import csv
import json
import math
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from training.config import _load_raw_config
from training.dataset import build_dataset_splits
from training.graph_modules import build_slot_conditioning
from training.graph_targets import bbox_minmax_3d_after_crop
from training.prompts import prompt_from_scop_depth_row, scene_graph_payload_from_row
from training.runtime import (
    DEFAULT_FLUX_MODEL_ID,
    choose_weight_dtype,
    infer_graph_encoder_config,
    infer_text_encoder_type,
    load_graph_encoder,
    load_graph_label_encoder,
    normalize_graph_encoder_state_dict,
    resolve_torch_device,
)
from training.scene_graph import build_batched_scene_graphs


RELATION_NAMES = (
    "left_of",
    "right_of",
    "above",
    "below",
    "on",
    "in_front_of",
    "behind",
    "hidden_by",
    "next_to",
)

OCCLUSION_RELATIONS = {"in_front_of", "behind", "hidden_by"}
RELATION_2D = {"left_of", "right_of", "above", "below", "on", "next_to"}
RELATION_3D = OCCLUSION_RELATIONS
DEFAULT_OCCLUSION_OVERLAP_THRESHOLD = 0.2


@dataclass(frozen=True)
class LayoutPrediction:
    labels: list[str]
    centers: torch.Tensor
    sizes: torch.Tensor
    boxes: torch.Tensor


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate scene-graph layout predictors on relation validity, box accuracy, and diversity."
    )
    parser.add_argument("--config", type=Path, required=True)
    return parser


def normalize_prompt(prompt: str) -> str:
    normalized = " ".join(prompt.split()).lower()
    for prefix in ("a photo of ", "an image of ", "a picture of "):
        if normalized.startswith(prefix):
            normalized = normalized[len(prefix) :]
            break
    return normalized


def row_image_size(row: dict[str, Any]) -> tuple[int, int]:
    raw = row.get("image_size")
    if isinstance(raw, (list, tuple)) and len(raw) >= 2:
        return int(raw[0]), int(raw[1])
    width = row.get("width")
    height = row.get("height")
    if width is not None and height is not None:
        return int(width), int(height)
    return 512, 512


def row_target_layout(row: dict[str, Any]) -> LayoutPrediction:
    scene_graph = scene_graph_payload_from_row(row)
    node_count = len(scene_graph["nodes"])
    boxes, _mask = bbox_minmax_3d_after_crop(
        [row],
        [row_image_size(row)],
        max_nodes=node_count,
        device=torch.device("cpu"),
    )
    boxes = boxes[0, :node_count].to(torch.float32)
    centers = boxes_to_centers(boxes)
    sizes = boxes_to_sizes(boxes)
    labels = [str(node["label"]) for node in scene_graph["nodes"]]
    return LayoutPrediction(labels=labels, centers=centers, sizes=sizes, boxes=boxes)


def boxes_to_centers(boxes: torch.Tensor) -> torch.Tensor:
    centers_01 = (boxes[..., :3] + boxes[..., 3:]) * 0.5
    return centers_01.mul(2.0).sub(1.0)


def boxes_to_sizes(boxes: torch.Tensor) -> torch.Tensor:
    return (boxes[..., 3:] - boxes[..., :3]).clamp_min(1e-6)


def centers_sizes_to_boxes(centers: torch.Tensor, sizes: torch.Tensor) -> torch.Tensor:
    centers_01 = centers.add(1.0).mul(0.5)
    half_sizes = sizes.clamp_min(1e-6) * 0.5
    return torch.cat([centers_01 - half_sizes, centers_01 + half_sizes], dim=-1)


def clip_centers_to_unit_box(centers: torch.Tensor, sizes: torch.Tensor) -> torch.Tensor:
    lower = sizes.clamp_min(1e-6).mul(0.5).mul(2.0).sub(1.0)
    upper = 1.0 - sizes.clamp_min(1e-6).mul(0.5).mul(2.0)
    return torch.maximum(torch.minimum(centers, upper), lower)


def box_iou_nd(pred_boxes: torch.Tensor, target_boxes: torch.Tensor, dims: int = 3) -> torch.Tensor:
    pred_min = pred_boxes[..., :dims]
    pred_max = pred_boxes[..., 3 : 3 + dims]
    target_min = target_boxes[..., :dims]
    target_max = target_boxes[..., 3 : 3 + dims]
    inter_min = torch.maximum(pred_min, target_min)
    inter_max = torch.minimum(pred_max, target_max)
    inter = (inter_max - inter_min).clamp_min(0.0).prod(dim=-1)
    pred_volume = (pred_max - pred_min).clamp_min(0.0).prod(dim=-1)
    target_volume = (target_max - target_min).clamp_min(0.0).prod(dim=-1)
    union = pred_volume + target_volume - inter
    return torch.where(union > 0, inter / union.clamp_min(1e-8), torch.zeros_like(union))


def box_iou_2d(pred_boxes: torch.Tensor, target_boxes: torch.Tensor) -> torch.Tensor:
    pred_xy = torch.stack(
        [pred_boxes[..., 0], pred_boxes[..., 1], pred_boxes[..., 3], pred_boxes[..., 4]],
        dim=-1,
    )
    target_xy = torch.stack(
        [target_boxes[..., 0], target_boxes[..., 1], target_boxes[..., 3], target_boxes[..., 4]],
        dim=-1,
    )
    pred_min = pred_xy[..., :2]
    pred_max = pred_xy[..., 2:]
    target_min = target_xy[..., :2]
    target_max = target_xy[..., 2:]
    inter_min = torch.maximum(pred_min, target_min)
    inter_max = torch.minimum(pred_max, target_max)
    inter = (inter_max - inter_min).clamp_min(0.0).prod(dim=-1)
    pred_area = (pred_max - pred_min).clamp_min(0.0).prod(dim=-1)
    target_area = (target_max - target_min).clamp_min(0.0).prod(dim=-1)
    union = pred_area + target_area - inter
    return torch.where(union > 0, inter / union.clamp_min(1e-8), torch.zeros_like(union))


def pair_overlap_2d_iomin(box_a: torch.Tensor, box_b: torch.Tensor) -> torch.Tensor:
    """Projected 2D overlap normalized by the smaller box area.

    IoU can be harsh when one object is much smaller than the other. For
    occlusion-like relations, intersection-over-min-area better captures whether
    the smaller object is actually projected onto the larger one.
    """

    a_min = box_a[..., [0, 1]]
    a_max = box_a[..., [3, 4]]
    b_min = box_b[..., [0, 1]]
    b_max = box_b[..., [3, 4]]
    inter_min = torch.maximum(a_min, b_min)
    inter_max = torch.minimum(a_max, b_max)
    inter = (inter_max - inter_min).clamp_min(0.0).prod(dim=-1)
    area_a = (a_max - a_min).clamp_min(0.0).prod(dim=-1)
    area_b = (b_max - b_min).clamp_min(0.0).prod(dim=-1)
    denom = torch.minimum(area_a, area_b).clamp_min(1e-8)
    return inter / denom


def relation_triplets(row: dict[str, Any]) -> list[tuple[int, int, str]]:
    scene_graph = scene_graph_payload_from_row(row)
    node_id_to_index = {node["id"]: index for index, node in enumerate(scene_graph["nodes"])}
    triplets = []
    for edge in scene_graph["edges"]:
        relation = str(edge["relation"])
        if relation in RELATION_NAMES:
            triplets.append(
                (
                    node_id_to_index[edge["source_id"]],
                    node_id_to_index[edge["target_id"]],
                    relation,
                )
            )
    return triplets


def relation_margin_violation(delta: torch.Tensor, relation: str, *, margin: bool) -> torch.Tensor:
    xy_margin = 0.1 if margin else 0.0
    z_margin = 0.05 if margin else 0.0
    if relation == "left_of":
        return torch.relu(torch.tensor(xy_margin, dtype=delta.dtype) - delta[0])
    if relation == "right_of":
        return torch.relu(torch.tensor(xy_margin, dtype=delta.dtype) + delta[0])
    if relation == "above":
        return torch.relu(torch.tensor(xy_margin, dtype=delta.dtype) - delta[1])
    if relation == "below":
        return torch.relu(torch.tensor(xy_margin, dtype=delta.dtype) + delta[1])
    if relation == "on":
        return torch.relu(torch.tensor(xy_margin, dtype=delta.dtype) - delta[1])
    if relation == "in_front_of":
        return torch.relu(torch.tensor(z_margin, dtype=delta.dtype) + delta[2])
    if relation == "behind":
        return torch.relu(torch.tensor(z_margin, dtype=delta.dtype) - delta[2])
    if relation == "hidden_by":
        return torch.relu(torch.tensor(z_margin, dtype=delta.dtype) - delta[2])
    if relation == "next_to":
        x_ok = torch.relu(delta[0].abs() - (0.35 if margin else 0.5))
        y_ok = torch.relu(delta[1].abs() - (0.2 if margin else 0.35))
        z_ok = torch.relu(delta[2].abs() - (0.2 if margin else 0.35))
        return (x_ok + y_ok + z_ok) / 3.0
    return torch.tensor(0.0, dtype=delta.dtype)


def relation_metrics(
    centers: torch.Tensor,
    triplets: list[tuple[int, int, str]],
    *,
    boxes: torch.Tensor | None = None,
    occlusion_overlap_threshold: float = DEFAULT_OCCLUSION_OVERLAP_THRESHOLD,
) -> dict[str, float]:
    if not triplets:
        return {
            "rel_acc": math.nan,
            "rel_2d_acc": math.nan,
            "rel_3d_acc": math.nan,
            "rel_margin_acc": math.nan,
            "rel_order_acc": math.nan,
            "rel_2d_order_acc": math.nan,
            "rel_3d_order_acc": math.nan,
            "rel_order_margin_acc": math.nan,
            "rel_violation": 0.0,
            "rel_margin_violation": 0.0,
            "occlusion_overlap_acc": math.nan,
            "occlusion_overlap_iomin": math.nan,
        }
    sign_ok = []
    sign_2d_ok = []
    sign_3d_ok = []
    margin_ok = []
    order_ok = []
    order_2d_ok = []
    order_3d_ok = []
    order_margin_ok = []
    violations = []
    margin_violations = []
    occlusion_overlap_ok = []
    occlusion_overlaps = []
    for src, dst, relation in triplets:
        delta = centers[dst] - centers[src]
        violation = relation_margin_violation(delta, relation, margin=False)
        margin_violation = relation_margin_violation(delta, relation, margin=True)
        pair_order_ok = float(violation <= 1e-6)
        pair_order_margin_ok = float(margin_violation <= 1e-6)
        pair_overlap_ok = 1.0
        if relation in OCCLUSION_RELATIONS and boxes is not None:
            overlap = float(pair_overlap_2d_iomin(boxes[src], boxes[dst]))
            pair_overlap_ok = float(overlap >= occlusion_overlap_threshold)
            occlusion_overlap_ok.append(pair_overlap_ok)
            occlusion_overlaps.append(overlap)
        order_ok.append(pair_order_ok)
        order_margin_ok.append(pair_order_margin_ok)
        pair_sign_ok = float(pair_order_ok and pair_overlap_ok)
        pair_margin_ok = float(pair_order_margin_ok and pair_overlap_ok)
        sign_ok.append(pair_sign_ok)
        margin_ok.append(pair_margin_ok)
        if relation in RELATION_2D:
            sign_2d_ok.append(pair_sign_ok)
            order_2d_ok.append(pair_order_ok)
        elif relation in RELATION_3D:
            sign_3d_ok.append(pair_sign_ok)
            order_3d_ok.append(pair_order_ok)
        violations.append(float(violation))
        margin_violations.append(float(margin_violation))
    return {
        "rel_acc": sum(sign_ok) / len(sign_ok),
        "rel_2d_acc": aggregate(sign_2d_ok),
        "rel_3d_acc": aggregate(sign_3d_ok),
        "rel_margin_acc": sum(margin_ok) / len(margin_ok),
        "rel_order_acc": sum(order_ok) / len(order_ok),
        "rel_2d_order_acc": aggregate(order_2d_ok),
        "rel_3d_order_acc": aggregate(order_3d_ok),
        "rel_order_margin_acc": sum(order_margin_ok) / len(order_margin_ok),
        "rel_violation": sum(violations) / len(violations),
        "rel_margin_violation": sum(margin_violations) / len(margin_violations),
        "occlusion_overlap_acc": aggregate(occlusion_overlap_ok),
        "occlusion_overlap_iomin": aggregate(occlusion_overlaps),
    }


def has_relation_type(triplets: list[tuple[int, int, str]], relation_names: set[str]) -> bool:
    return any(relation in relation_names for _src, _dst, relation in triplets)


def box_quality_metrics(pred: LayoutPrediction, target: LayoutPrediction) -> dict[str, float]:
    pred_boxes = pred.boxes.to(torch.float32)
    target_boxes = target.boxes.to(torch.float32)
    return {
        "box_l1": float(torch.mean(torch.abs(pred_boxes - target_boxes))),
        "center_l1": float(torch.mean(torch.abs(pred.centers - target.centers))),
        "size_l1": float(torch.mean(torch.abs(pred.sizes - target.sizes))),
        "iou_3d": float(box_iou_nd(pred_boxes, target_boxes, dims=3).mean()),
        "iou_2d": float(box_iou_2d(pred_boxes, target_boxes).mean()),
    }


def oob_rate(boxes: torch.Tensor) -> float:
    bad = (boxes[..., :3] < 0.0) | (boxes[..., 3:] > 1.0) | (boxes[..., 3:] <= boxes[..., :3])
    return float(bad.any(dim=-1).to(torch.float32).mean())


def mean_pairwise_iou(boxes: torch.Tensor, *, dims: int) -> float:
    count = boxes.shape[0]
    if count < 2:
        return 0.0
    values = []
    for i in range(count):
        for j in range(i + 1, count):
            values.append(box_iou_nd(boxes[i], boxes[j], dims=dims).mean())
    return float(torch.stack(values).mean()) if values else 0.0


def mean_pairwise_flat_l2(values: torch.Tensor) -> float:
    count = values.shape[0]
    if count < 2:
        return 0.0
    flat = values.reshape(count, -1)
    distances = torch.pdist(flat, p=2)
    if distances.numel() == 0:
        return 0.0
    return float(distances.mean())


def aggregate(values: list[float]) -> float:
    finite = [value for value in values if not math.isnan(value)]
    if not finite:
        return math.nan
    return sum(finite) / len(finite)


def finite_min(values: list[float]) -> float:
    finite = [value for value in values if not math.isnan(value)]
    return min(finite) if finite else math.nan


def finite_max(values: list[float]) -> float:
    finite = [value for value in values if not math.isnan(value)]
    return max(finite) if finite else math.nan


def build_prompt_references(rows: list[dict[str, Any]]) -> dict[str, list[LayoutPrediction]]:
    references: dict[str, list[LayoutPrediction]] = defaultdict(list)
    for row in rows:
        references[normalize_prompt(prompt_from_scop_depth_row(row))].append(row_target_layout(row))
    return dict(references)


class PriorStats:
    def __init__(self, train_rows: list[dict[str, Any]]) -> None:
        self.category_centers: dict[str, torch.Tensor] = {}
        self.category_sizes: dict[str, torch.Tensor] = {}
        self.global_center = torch.zeros(3)
        self.global_size = torch.tensor([0.35, 0.35, 0.18])
        self.relation_offsets: dict[tuple[str, str, str], torch.Tensor] = {}
        self.relation_fallback_offsets: dict[str, torch.Tensor] = {}
        self._build(train_rows)

    def _build(self, train_rows: list[dict[str, Any]]) -> None:
        category_centers: dict[str, list[torch.Tensor]] = defaultdict(list)
        category_sizes: dict[str, list[torch.Tensor]] = defaultdict(list)
        relation_offsets: dict[tuple[str, str, str], list[torch.Tensor]] = defaultdict(list)
        relation_fallback_offsets: dict[str, list[torch.Tensor]] = defaultdict(list)
        all_centers: list[torch.Tensor] = []
        all_sizes: list[torch.Tensor] = []

        for row in train_rows:
            layout = row_target_layout(row)
            for label, center, size in zip(layout.labels, layout.centers, layout.sizes):
                category_centers[label].append(center)
                category_sizes[label].append(size)
                all_centers.append(center)
                all_sizes.append(size)
            for src, dst, relation in relation_triplets(row):
                src_label = layout.labels[src]
                dst_label = layout.labels[dst]
                delta = layout.centers[dst] - layout.centers[src]
                relation_offsets[(src_label, relation, dst_label)].append(delta)
                relation_fallback_offsets[relation].append(delta)

        if all_centers:
            self.global_center = torch.stack(all_centers).mean(dim=0)
        if all_sizes:
            self.global_size = torch.stack(all_sizes).mean(dim=0)
        self.category_centers = {
            label: torch.stack(values).mean(dim=0) for label, values in category_centers.items()
        }
        self.category_sizes = {
            label: torch.stack(values).mean(dim=0).clamp_min(0.03)
            for label, values in category_sizes.items()
        }
        self.relation_offsets = {
            key: torch.stack(values).mean(dim=0) for key, values in relation_offsets.items()
        }
        self.relation_fallback_offsets = {
            key: torch.stack(values).mean(dim=0) for key, values in relation_fallback_offsets.items()
        }

    def center_for(self, label: str) -> torch.Tensor:
        return self.category_centers.get(label, self.global_center).clone()

    def size_for(self, label: str) -> torch.Tensor:
        return self.category_sizes.get(label, self.global_size).clone().clamp_min(0.03)

    def relation_offset_for(self, src_label: str, relation: str, dst_label: str) -> torch.Tensor:
        return self.relation_offsets.get(
            (src_label, relation, dst_label),
            self.relation_fallback_offsets.get(relation, torch.zeros(3)),
        ).clone()


def baseline_class_prior(row: dict[str, Any], stats: PriorStats) -> LayoutPrediction:
    scene_graph = scene_graph_payload_from_row(row)
    labels = [str(node["label"]) for node in scene_graph["nodes"]]
    sizes = torch.stack([stats.size_for(label) for label in labels])
    centers = torch.stack([stats.center_for(label) for label in labels])
    boxes = centers_sizes_to_boxes(centers, sizes)
    return LayoutPrediction(labels=labels, centers=centers, sizes=sizes, boxes=boxes)


def baseline_relation_prior(row: dict[str, Any], stats: PriorStats) -> LayoutPrediction:
    prediction = baseline_class_prior(row, stats)
    centers = prediction.centers.clone()
    for src, dst, relation in relation_triplets(row):
        offset = stats.relation_offset_for(prediction.labels[src], relation, prediction.labels[dst])
        pair_mid = (centers[src] + centers[dst]) * 0.5
        centers[src] = pair_mid - offset * 0.5
        centers[dst] = pair_mid + offset * 0.5
    centers = clip_centers_to_unit_box(centers, prediction.sizes)
    boxes = centers_sizes_to_boxes(centers, prediction.sizes)
    return LayoutPrediction(labels=prediction.labels, centers=centers, sizes=prediction.sizes, boxes=boxes)


def heuristic_delta(relation: str, src_size: torch.Tensor, dst_size: torch.Tensor) -> torch.Tensor:
    delta = torch.zeros(3)
    margin_xy = 0.12
    margin_z = 0.08
    if relation == "left_of":
        delta[0] = src_size[0] + dst_size[0] + margin_xy
    elif relation == "right_of":
        delta[0] = -(src_size[0] + dst_size[0] + margin_xy)
    elif relation in {"above", "on"}:
        delta[1] = src_size[1] + dst_size[1] + margin_xy
    elif relation == "below":
        delta[1] = -(src_size[1] + dst_size[1] + margin_xy)
    elif relation == "in_front_of":
        delta[2] = -(src_size[2] + dst_size[2] + margin_z)
    elif relation in {"behind", "hidden_by"}:
        delta[2] = src_size[2] + dst_size[2] + margin_z
    elif relation == "next_to":
        delta[0] = src_size[0] + dst_size[0] + margin_xy
    return delta


def baseline_relation_heuristic(row: dict[str, Any], stats: PriorStats) -> LayoutPrediction:
    prediction = baseline_class_prior(row, stats)
    centers = prediction.centers.clone()
    if centers.numel():
        pair_mid = centers.mean(dim=0)
    else:
        pair_mid = torch.zeros(3)
    for src, dst, relation in relation_triplets(row):
        delta = heuristic_delta(relation, prediction.sizes[src], prediction.sizes[dst])
        centers[src] = pair_mid - delta * 0.5
        centers[dst] = pair_mid + delta * 0.5
    centers = clip_centers_to_unit_box(centers, prediction.sizes)
    boxes = centers_sizes_to_boxes(centers, prediction.sizes)
    return LayoutPrediction(labels=prediction.labels, centers=centers, sizes=prediction.sizes, boxes=boxes)


def baseline_random_jitter(
    row: dict[str, Any],
    stats: PriorStats,
    *,
    generator: torch.Generator,
    center_std: float,
    size_std: float,
) -> LayoutPrediction:
    base = baseline_relation_heuristic(row, stats)
    centers = base.centers + torch.randn(base.centers.shape, generator=generator) * center_std
    log_sizes = base.sizes.clamp_min(0.03).log() + torch.randn(base.sizes.shape, generator=generator) * size_std
    sizes = log_sizes.exp().clamp(0.03, 0.95)
    boxes = centers_sizes_to_boxes(centers, sizes)
    return LayoutPrediction(labels=base.labels, centers=centers, sizes=sizes, boxes=boxes)


class GraphModelPredictor:
    def __init__(
        self,
        *,
        checkpoint: Path,
        model_id: str,
        device: str,
        mixed_precision: str,
        text_runtime_cache: dict[tuple[str, str, torch.dtype], tuple[object, object, int]],
    ) -> None:
        self.checkpoint = checkpoint
        self.device = device
        state_dict = normalize_graph_encoder_state_dict(torch.load(checkpoint, map_location="cpu"))
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
        dtype = choose_weight_dtype(device, mixed_precision)
        cache_key = (model_id, text_encoder_type, dtype)
        if cache_key not in text_runtime_cache:
            text_runtime_cache[cache_key] = load_graph_label_encoder(
                model_id=model_id,
                text_encoder_type=text_encoder_type,
                torch_dtype=dtype,
                device=device,
            )
        self.tokenizer, self.text_encoder, encoder_hidden_dim = text_runtime_cache[cache_key]
        self.graph_encoder = load_graph_encoder(
            path=checkpoint,
            text_hidden_dim=encoder_hidden_dim,
            device=device,
            dtype=self.text_encoder.dtype,
        )
        self.graph_encoder.eval()
        self.label_embedding_cache: dict[str, torch.Tensor] = {}

    @torch.no_grad()
    def predict(self, row: dict[str, Any], *, layout_sample_mode: str, layout_z_scale: float) -> LayoutPrediction:
        scene_graph = scene_graph_payload_from_row(row)
        node_count = len(scene_graph["nodes"])
        slot_targets = torch.zeros(1, node_count, 3, device=torch.device(self.device))
        slot_mask = torch.ones(1, node_count, dtype=torch.bool, device=torch.device(self.device))
        batched_graph = build_batched_scene_graphs(
            [scene_graph],
            slot_targets=slot_targets,
            slot_mask=slot_mask,
        )
        output = build_slot_conditioning(
            tokenizer=self.tokenizer,
            text_encoder=self.text_encoder,
            scene_graph_batch=batched_graph,
            graph_encoder=self.graph_encoder,
            device=self.device,
            layout_sample_mode=layout_sample_mode,
            label_embedding_cache=self.label_embedding_cache,
            layout_z_scale=layout_z_scale,
        )
        labels = [str(node["label"]) for node in scene_graph["nodes"]]
        centers = output.slot_positions[0, :node_count].detach().cpu().to(torch.float32)
        sizes = output.slot_log_sizes_3d[0, :node_count].detach().cpu().to(torch.float32).exp().clamp_min(1e-6)
        if output.slot_boxes_3d is not None:
            boxes = output.slot_boxes_3d[0, :node_count].detach().cpu().to(torch.float32)
        else:
            boxes = centers_sizes_to_boxes(centers, sizes)
        return LayoutPrediction(labels=labels, centers=centers, sizes=sizes, boxes=boxes)


def prompt_reference_metrics(
    prediction: LayoutPrediction,
    prompt_key: str,
    prompt_references: dict[str, list[LayoutPrediction]],
) -> dict[str, float]:
    refs = prompt_references.get(prompt_key, [])
    if not refs:
        return {
            "nearest_prompt_box_l1": math.nan,
            "nearest_prompt_iou_3d": math.nan,
            "nearest_prompt_iou_2d": math.nan,
        }
    metrics = [box_quality_metrics(prediction, ref) for ref in refs if len(ref.labels) == len(prediction.labels)]
    if not metrics:
        return {
            "nearest_prompt_box_l1": math.nan,
            "nearest_prompt_iou_3d": math.nan,
            "nearest_prompt_iou_2d": math.nan,
        }
    return {
        "nearest_prompt_box_l1": min(item["box_l1"] for item in metrics),
        "nearest_prompt_iou_3d": max(item["iou_3d"] for item in metrics),
        "nearest_prompt_iou_2d": max(item["iou_2d"] for item in metrics),
    }


def evaluate_prediction(
    *,
    row: dict[str, Any],
    prediction: LayoutPrediction,
    target: LayoutPrediction,
    prompt_references: dict[str, list[LayoutPrediction]],
) -> dict[str, float]:
    triplets = relation_triplets(row)
    prompt_key = normalize_prompt(prompt_from_scop_depth_row(row))
    metrics = {}
    metrics.update(relation_metrics(prediction.centers, triplets, boxes=prediction.boxes))
    metrics.update(box_quality_metrics(prediction, target))
    metrics.update(prompt_reference_metrics(prediction, prompt_key, prompt_references))
    metrics["oob_rate"] = oob_rate(prediction.boxes)
    metrics["pair_iou_3d"] = mean_pairwise_iou(prediction.boxes, dims=3)
    metrics["pair_iou_2d"] = mean_pairwise_iou(prediction.boxes, dims=2)
    return metrics


def evaluate_method(
    *,
    method: dict[str, Any],
    rows: list[dict[str, Any]],
    stats: PriorStats,
    prompt_references: dict[str, list[LayoutPrediction]],
    graph_predictors: dict[str, GraphModelPredictor],
    base_seed: int,
) -> tuple[dict[str, float], list[dict[str, Any]], list[dict[str, Any]]]:
    name = str(method["name"])
    method_type = str(method.get("type", name))
    num_samples = int(method.get("num_samples", 1))
    layout_sample_mode = str(method.get("layout_sample_mode", "prior_sample"))
    layout_z_scale = float(method.get("layout_z_scale", 1.0))
    random_center_std = float(method.get("center_std", 0.25))
    random_size_std = float(method.get("size_std", 0.35))

    sample_rows: list[dict[str, Any]] = []
    per_relation_rows: list[dict[str, Any]] = []
    row_summaries: list[dict[str, float]] = []

    for row_index, row in enumerate(rows):
        target = row_target_layout(row)
        prompt = prompt_from_scop_depth_row(row)
        prompt_key = normalize_prompt(prompt)
        predictions: list[LayoutPrediction] = []
        for sample_index in range(num_samples):
            seed = base_seed + row_index * 1009 + sample_index
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
            if method_type == "class_prior":
                pred = baseline_class_prior(row, stats)
            elif method_type == "relation_prior":
                pred = baseline_relation_prior(row, stats)
            elif method_type == "relation_heuristic":
                pred = baseline_relation_heuristic(row, stats)
            elif method_type == "random_jitter":
                generator = torch.Generator().manual_seed(seed)
                pred = baseline_random_jitter(
                    row,
                    stats,
                    generator=generator,
                    center_std=random_center_std,
                    size_std=random_size_std,
                )
            elif method_type == "graph_checkpoint":
                predictor_key = str(method["checkpoint"])
                pred = graph_predictors[predictor_key].predict(
                    row,
                    layout_sample_mode=layout_sample_mode,
                    layout_z_scale=layout_z_scale,
                )
            else:
                raise ValueError(f"Unsupported layout eval method type: {method_type}")
            predictions.append(pred)

            metrics = evaluate_prediction(
                row=row,
                prediction=pred,
                target=target,
                prompt_references=prompt_references,
            )
            sample_row = {
                "method": name,
                "row_index": row_index,
                "sample_index": sample_index,
                "prompt": prompt,
                **metrics,
            }
            sample_rows.append(sample_row)

        triplets = relation_triplets(row)
        for src, dst, relation in triplets:
            relation_accs = []
            relation_2d_accs = []
            relation_3d_accs = []
            relation_margin_accs = []
            relation_order_accs = []
            relation_2d_order_accs = []
            relation_3d_order_accs = []
            occlusion_overlap_accs = []
            occlusion_overlap_iomins = []
            for pred in predictions:
                single_metrics = relation_metrics(pred.centers, [(src, dst, relation)], boxes=pred.boxes)
                relation_accs.append(single_metrics["rel_acc"])
                relation_2d_accs.append(single_metrics["rel_2d_acc"])
                relation_3d_accs.append(single_metrics["rel_3d_acc"])
                relation_margin_accs.append(single_metrics["rel_margin_acc"])
                relation_order_accs.append(single_metrics["rel_order_acc"])
                relation_2d_order_accs.append(single_metrics["rel_2d_order_acc"])
                relation_3d_order_accs.append(single_metrics["rel_3d_order_acc"])
                occlusion_overlap_accs.append(single_metrics["occlusion_overlap_acc"])
                occlusion_overlap_iomins.append(single_metrics["occlusion_overlap_iomin"])
            per_relation_rows.append(
                {
                    "method": name,
                    "relation": relation,
                    "row_index": row_index,
                    "rel_acc": aggregate(relation_accs),
                    "rel_2d_acc": aggregate(relation_2d_accs),
                    "rel_3d_acc": aggregate(relation_3d_accs),
                    "rel_margin_acc": aggregate(relation_margin_accs),
                    "rel_order_acc": aggregate(relation_order_accs),
                    "rel_2d_order_acc": aggregate(relation_2d_order_accs),
                    "rel_3d_order_acc": aggregate(relation_3d_order_accs),
                    "occlusion_overlap_acc": aggregate(occlusion_overlap_accs),
                    "occlusion_overlap_iomin": aggregate(occlusion_overlap_iomins),
                }
            )

        centers = torch.stack([pred.centers for pred in predictions], dim=0)
        sizes = torch.stack([pred.sizes for pred in predictions], dim=0)
        boxes = torch.stack([pred.boxes for pred in predictions], dim=0)
        sample_metrics = [
            evaluate_prediction(
                row=row,
                prediction=pred,
                target=target,
                prompt_references=prompt_references,
            )
            for pred in predictions
        ]
        valid_mask = torch.tensor(
            [
                float(item["rel_acc"] == 1.0 and item["oob_rate"] == 0.0)
                for item in sample_metrics
            ],
            dtype=torch.bool,
        )
        triplets = relation_triplets(row)
        has_2d_relations = has_relation_type(triplets, RELATION_2D)
        has_3d_relations = has_relation_type(triplets, RELATION_3D)
        valid_2d_sample_rate = (
            aggregate(
                [
                    float(item["rel_2d_acc"] == 1.0 and item["oob_rate"] == 0.0)
                    for item in sample_metrics
                ]
            )
            if has_2d_relations
            else math.nan
        )
        valid_3d_sample_rate = (
            aggregate(
                [
                    float(item["rel_3d_acc"] == 1.0 and item["oob_rate"] == 0.0)
                    for item in sample_metrics
                ]
            )
            if has_3d_relations
            else math.nan
        )
        valid_centers = centers[valid_mask]
        valid_sizes = sizes[valid_mask]
        row_summaries.append(
            {
                "sample_rel_acc": aggregate([item["rel_acc"] for item in sample_metrics]),
                "sample_rel_2d_acc": aggregate([item["rel_2d_acc"] for item in sample_metrics]),
                "sample_rel_3d_acc": aggregate([item["rel_3d_acc"] for item in sample_metrics]),
                "sample_rel_margin_acc": aggregate([item["rel_margin_acc"] for item in sample_metrics]),
                "sample_rel_order_acc": aggregate([item["rel_order_acc"] for item in sample_metrics]),
                "sample_rel_2d_order_acc": aggregate(
                    [item["rel_2d_order_acc"] for item in sample_metrics]
                ),
                "sample_rel_3d_order_acc": aggregate(
                    [item["rel_3d_order_acc"] for item in sample_metrics]
                ),
                "sample_rel_order_margin_acc": aggregate(
                    [item["rel_order_margin_acc"] for item in sample_metrics]
                ),
                "sample_occlusion_overlap_acc": aggregate(
                    [item["occlusion_overlap_acc"] for item in sample_metrics]
                ),
                "sample_occlusion_overlap_iomin": aggregate(
                    [item["occlusion_overlap_iomin"] for item in sample_metrics]
                ),
                "sample_box_l1": aggregate([item["box_l1"] for item in sample_metrics]),
                "sample_center_l1": aggregate([item["center_l1"] for item in sample_metrics]),
                "sample_size_l1": aggregate([item["size_l1"] for item in sample_metrics]),
                "sample_iou_3d": aggregate([item["iou_3d"] for item in sample_metrics]),
                "sample_iou_2d": aggregate([item["iou_2d"] for item in sample_metrics]),
                "sample_nearest_prompt_box_l1": aggregate(
                    [item["nearest_prompt_box_l1"] for item in sample_metrics]
                ),
                "sample_nearest_prompt_iou_3d": aggregate(
                    [item["nearest_prompt_iou_3d"] for item in sample_metrics]
                ),
                "best_box_l1": finite_min([item["box_l1"] for item in sample_metrics]),
                "best_nearest_prompt_box_l1": finite_min(
                    [item["nearest_prompt_box_l1"] for item in sample_metrics]
                ),
                "best_iou_3d": finite_max([item["iou_3d"] for item in sample_metrics]),
                "best_nearest_prompt_iou_3d": finite_max(
                    [item["nearest_prompt_iou_3d"] for item in sample_metrics]
                ),
                "valid_sample_rate": float(valid_mask.to(torch.float32).mean()),
                "valid_2d_sample_rate": valid_2d_sample_rate,
                "valid_3d_sample_rate": valid_3d_sample_rate,
                "center_std": float(centers.std(dim=0, unbiased=False).mean()),
                "size_std": float(sizes.std(dim=0, unbiased=False).mean()),
                "all_diversity_center_l2": mean_pairwise_flat_l2(centers),
                "all_diversity_size_l2": mean_pairwise_flat_l2(sizes),
                "valid_diversity_center_l2": mean_pairwise_flat_l2(valid_centers)
                if valid_centers.shape[0] >= 2
                else 0.0,
                "valid_diversity_size_l2": mean_pairwise_flat_l2(valid_sizes)
                if valid_sizes.shape[0] >= 2
                else 0.0,
                "sample_oob_rate": aggregate([item["oob_rate"] for item in sample_metrics]),
                "sample_pair_iou_3d": float(
                    torch.tensor([mean_pairwise_iou(boxes[sample_idx], dims=3) for sample_idx in range(boxes.shape[0])]).mean()
                ),
            }
        )

    summary = {"method": name, "rows": float(len(rows)), "samples_per_row": float(num_samples)}
    for key in sorted(row_summaries[0]):
        summary[key] = aggregate([row_summary[key] for row_summary in row_summaries])
    return summary, sample_rows, per_relation_rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("")
        return
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                fieldnames.append(key)
                seen.add(key)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_paper_table_rows(summaries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return a compact table with the metrics we are likely to report."""

    table_rows: list[dict[str, Any]] = []
    for row in summaries:
        table_rows.append(
            {
                "Method": row.get("method"),
                "Rel Acc ↑": row.get("sample_rel_acc"),
                "2D Rel Acc ↑": row.get("sample_rel_2d_acc"),
                "3D Rel Acc ↑": row.get("sample_rel_3d_acc"),
                "Order Acc ↑": row.get("sample_rel_order_acc"),
                "3D Order Acc ↑": row.get("sample_rel_3d_order_acc"),
                "Occ. Overlap ↑": row.get("sample_occlusion_overlap_iomin"),
                "Box L1 ↓": row.get("sample_box_l1"),
                "Nearest L1 ↓": row.get("sample_nearest_prompt_box_l1"),
                "Best-of-K L1 ↓": row.get("best_box_l1"),
                "2D IoU ↑": row.get("sample_iou_2d"),
                "3D IoU ↑": row.get("sample_iou_3d"),
                "Best-of-K 3D IoU ↑": row.get("best_iou_3d"),
                "Center STD ↑": row.get("center_std"),
                "Size STD ↑": row.get("size_std"),
                "Valid Rate ↑": row.get("valid_sample_rate"),
                "2D Valid Rate ↑": row.get("valid_2d_sample_rate"),
                "3D Valid Rate ↑": row.get("valid_3d_sample_rate"),
                "Valid Diversity ↑": row.get("valid_diversity_center_l2"),
                "OOB ↓": row.get("sample_oob_rate"),
                "Overlap ↓": row.get("sample_pair_iou_3d"),
            }
        )
    return table_rows


def format_float(value: Any) -> str:
    if isinstance(value, str):
        return value
    try:
        number = float(value)
    except (TypeError, ValueError):
        return ""
    if math.isnan(number):
        return ""
    return f"{number:.4f}"


def write_markdown_table(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("")
        return
    headers = list(rows[0].keys())
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(format_float(row.get(header)) for header in headers) + " |")
    path.write_text("\n".join(lines) + "\n")


def prepare_methods(
    methods: list[dict[str, Any]],
    *,
    model_id: str,
    device: str,
    mixed_precision: str,
) -> dict[str, GraphModelPredictor]:
    text_runtime_cache: dict[tuple[str, str, torch.dtype], tuple[object, object, int]] = {}
    predictors: dict[str, GraphModelPredictor] = {}
    for method in methods:
        if str(method.get("type", method.get("name"))) != "graph_checkpoint":
            continue
        checkpoint = Path(method["checkpoint"])
        key = str(checkpoint)
        if key not in predictors:
            predictors[key] = GraphModelPredictor(
                checkpoint=checkpoint,
                model_id=str(method.get("model_id", model_id)),
                device=device,
                mixed_precision=mixed_precision,
                text_runtime_cache=text_runtime_cache,
            )
    return predictors


def main() -> int:
    args = make_parser().parse_args()
    raw_config = _load_raw_config(args.config)
    dataset_cfg = raw_config.get("dataset", {})
    runtime_cfg = raw_config.get("runtime", {})
    output_cfg = raw_config.get("output", {})
    methods = raw_config.get("methods", [])
    if not methods:
        raise ValueError("Layout eval config must include at least one method.")

    dataset_dir = Path(dataset_cfg["dataset_dir"])
    output_dir = Path(output_cfg.get("output_dir", "outputs/eval/layout_eval")).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    seed = int(runtime_cfg.get("seed", 42))
    device = resolve_torch_device(str(runtime_cfg.get("device", "auto")))
    mixed_precision = str(runtime_cfg.get("mixed_precision", "no"))
    model_id = str(runtime_cfg.get("model_id", DEFAULT_FLUX_MODEL_ID))

    splits = build_dataset_splits(
        dataset_dir,
        image_size=int(dataset_cfg.get("image_size", 512)),
        prompt_prefix=str(dataset_cfg.get("prompt_prefix", "a photo of")),
        limit_rows=dataset_cfg.get("limit_rows"),
        seed=int(dataset_cfg.get("split_seed", seed)),
        eval_fraction=float(dataset_cfg.get("eval_fraction", 0.05)),
        test_fraction=float(dataset_cfg.get("test_fraction", 0.05)),
        load_images=False,
        prompt_filter=dataset_cfg.get("prompt_filter"),
    )
    train_rows = splits[str(dataset_cfg.get("prior_split", "train"))].rows
    eval_rows = splits[str(dataset_cfg.get("eval_split", "test"))].rows
    max_eval_rows = dataset_cfg.get("max_eval_rows")
    if max_eval_rows is not None:
        rng = random.Random(seed)
        eval_rows = list(eval_rows)
        rng.shuffle(eval_rows)
        eval_rows = eval_rows[: int(max_eval_rows)]
    if not eval_rows:
        raise ValueError("No rows selected for layout evaluation.")

    stats = PriorStats(train_rows)
    prompt_references = build_prompt_references(eval_rows)
    graph_predictors = prepare_methods(
        methods,
        model_id=model_id,
        device=device,
        mixed_precision=mixed_precision,
    )

    all_summaries: list[dict[str, Any]] = []
    all_samples: list[dict[str, Any]] = []
    all_relations: list[dict[str, Any]] = []
    for method in methods:
        summary, sample_rows, relation_rows = evaluate_method(
            method=method,
            rows=eval_rows,
            stats=stats,
            prompt_references=prompt_references,
            graph_predictors=graph_predictors,
            base_seed=seed,
        )
        all_summaries.append(summary)
        all_samples.extend(sample_rows)
        all_relations.extend(relation_rows)
        print(
            f"{summary['method']}: rel_acc={summary['sample_rel_acc']:.4f}, "
            f"box_l1={summary['sample_box_l1']:.4f}, "
            f"valid_div={summary['valid_diversity_center_l2']:.4f}, "
            f"oob={summary['sample_oob_rate']:.4f}"
        )

    relation_summary: list[dict[str, Any]] = []
    relation_groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in all_relations:
        relation_groups[(str(row["method"]), str(row["relation"]))].append(row)
    for (method, relation), rows in sorted(relation_groups.items()):
        relation_summary.append(
            {
                "method": method,
                "relation": relation,
                "count": len(rows),
                "rel_acc": aggregate([float(row["rel_acc"]) for row in rows]),
                "rel_2d_acc": aggregate([float(row["rel_2d_acc"]) for row in rows]),
                "rel_3d_acc": aggregate([float(row["rel_3d_acc"]) for row in rows]),
                "rel_margin_acc": aggregate([float(row["rel_margin_acc"]) for row in rows]),
                "rel_order_acc": aggregate([float(row["rel_order_acc"]) for row in rows]),
                "rel_2d_order_acc": aggregate([float(row["rel_2d_order_acc"]) for row in rows]),
                "rel_3d_order_acc": aggregate([float(row["rel_3d_order_acc"]) for row in rows]),
                "occlusion_overlap_acc": aggregate(
                    [float(row["occlusion_overlap_acc"]) for row in rows]
                ),
                "occlusion_overlap_iomin": aggregate(
                    [float(row["occlusion_overlap_iomin"]) for row in rows]
                ),
            }
        )

    paper_table_rows = build_paper_table_rows(all_summaries)
    write_csv(output_dir / "metrics_summary.csv", all_summaries)
    write_csv(output_dir / "paper_table.csv", paper_table_rows)
    write_markdown_table(output_dir / "paper_table.md", paper_table_rows)
    write_csv(output_dir / "per_sample_metrics.csv", all_samples)
    write_csv(output_dir / "per_relation_metrics.csv", relation_summary)
    (output_dir / "layout_eval_config.resolved.json").write_text(
        json.dumps(
            {
                "config": raw_config,
                "dataset_dir": str(dataset_dir),
                "output_dir": str(output_dir),
                "train_rows": len(train_rows),
                "eval_rows": len(eval_rows),
            },
            indent=2,
        )
    )
    print(f"Wrote layout eval metrics to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
