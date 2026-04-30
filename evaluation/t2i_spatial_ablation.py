from __future__ import annotations

import argparse
import json
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from tqdm import tqdm

from evaluation.prompt_parser import parse_prompt_to_scene_graph


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_T2I_ROOT = REPO_ROOT / "external" / "T2I-CompBench"
VALID_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
PERSON_ALIASES = {"girl", "boy", "man", "woman"}
TRAILING_DIGITS_RE = re.compile(r"\d+$")



@dataclass
class Detection:
    label: str
    normalized_label: str
    score: float
    box: dict[str, float]


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run a diagnostic 2D spatial ablation that separates object-detection success "
            "from relation accuracy on an existing generated samples directory."
        )
    )
    parser.add_argument(
        "--generated-dir",
        type=Path,
        required=True,
        help="Our generated output directory containing samples/ and optionally run_config.json.",
    )
    parser.add_argument(
        "--prompt-file",
        type=Path,
        default=None,
        help="Optional plain-text prompt file used to verify prompt coverage and ordering.",
    )
    parser.add_argument(
        "--t2i-compbench-root",
        type=Path,
        default=DEFAULT_T2I_ROOT,
        help="Path to a T2I-CompBench checkout. Defaults to external/T2I-CompBench.",
    )
    parser.add_argument(
        "--relation-threshold",
        type=float,
        default=0.5,
        help="Threshold used to binarize relation correctness once both objects are detected.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Detector batch size. Keep at 1 to match the benchmark default.",
    )
    parser.add_argument(
        "--limit-images",
        type=int,
        default=None,
        help="Optional cap on the number of images to score.",
    )
    parser.add_argument(
        "--output-file",
        type=Path,
        default=None,
        help="Optional explicit summary JSON path.",
    )
    parser.add_argument(
        "--records-file",
        type=Path,
        default=None,
        help="Optional explicit per-image JSON path.",
    )
    return parser


def resolve_t2i_root(t2i_compbench_root: Path) -> Path:
    root = t2i_compbench_root.resolve()
    if (root / "UniDet_eval").exists():
        return root
    raise FileNotFoundError(f"Could not locate a T2I-CompBench checkout under {root}")


def load_prompt_lines(path: Path) -> list[str]:
    prompts: list[str] = []
    for raw_line in path.read_text().splitlines():
        line = raw_line.strip()
        if line:
            prompts.append(line)
    return prompts


def list_sample_images(samples_dir: Path) -> list[Path]:
    image_paths = [
        path
        for path in samples_dir.iterdir()
        if path.is_file() and path.suffix.lower() in VALID_IMAGE_SUFFIXES
    ]
    image_paths.sort(key=lambda path: int(path.stem.rsplit("_", 1)[1]))
    return image_paths


def image_index_from_path(path: Path) -> int:
    return int(path.stem.rsplit("_", 1)[1])


def prompt_from_image_path(path: Path) -> str:
    prompt_stem, _ = path.stem.rsplit("_", 1)
    return prompt_stem


def normalize_label(text: str) -> str:
    cleaned = "".join(char.lower() if char.isalnum() or char.isspace() else " " for char in text)
    tokens = [token for token in cleaned.split() if token]
    if not tokens:
        return ""
    label = TRAILING_DIGITS_RE.sub("", tokens[-1])
    if label in PERSON_ALIASES:
        return "person"
    if label.endswith("ies") and len(label) > 3:
        return label[:-3] + "y"
    if label.endswith("ses") and len(label) > 3:
        return label[:-2]
    if label.endswith("s") and not label.endswith("ss") and len(label) > 3:
        return label[:-1]
    return label


def label_aliases(text: str) -> set[str]:
    cleaned = "".join(char.lower() if char.isalnum() or char.isspace() else " " for char in text)
    tokens = [token for token in cleaned.split() if token]
    aliases = {normalize_label(text)}
    if tokens:
        aliases.add(tokens[-1].lower())
        aliases.add(normalize_label(tokens[-1]))
    if len(tokens) >= 2:
        aliases.add(" ".join(tokens[-2:]))
    aliases.discard("")
    return aliases


def parse_prompt(prompt: str) -> tuple[str, str, str]:
    scene_graph = parse_prompt_to_scene_graph(prompt)
    nodes = scene_graph["nodes"]
    edges = scene_graph["edges"]
    if len(nodes) != 2 or len(edges) != 1:
        raise ValueError(f"Expected a single binary relation prompt, but got: {prompt}")
    return str(nodes[0]["label"]), str(edges[0]["relation"]), str(nodes[1]["label"])


def box_center(box: dict[str, float]) -> tuple[float, float]:
    return ((box["x_min"] + box["x_max"]) / 2.0, (box["y_min"] + box["y_max"]) / 2.0)


def box_iou(box1: dict[str, float], box2: dict[str, float]) -> float:
    x_overlap = max(0.0, min(box1["x_max"], box2["x_max"]) - max(box1["x_min"], box2["x_min"]))
    y_overlap = max(0.0, min(box1["y_max"], box2["y_max"]) - max(box1["y_min"], box2["y_min"]))
    intersection = x_overlap * y_overlap
    area1 = max(0.0, box1["x_max"] - box1["x_min"]) * max(0.0, box1["y_max"] - box1["y_min"])
    area2 = max(0.0, box2["x_max"] - box2["x_min"]) * max(0.0, box2["y_max"] - box2["y_min"])
    union = area1 + area2 - intersection
    if union <= 0:
        return 0.0
    return intersection / union


def relation_score(
    relation: str,
    subject_box: dict[str, float],
    object_box: dict[str, float],
    *,
    iou_threshold: float = 0.1,
    distance_threshold: float = 150.0,
) -> float:
    subject_center = box_center(subject_box)
    object_center = box_center(object_box)
    x_distance = object_center[0] - subject_center[0]
    y_distance = object_center[1] - subject_center[1]
    iou = box_iou(subject_box, object_box)

    if relation == "next_to":
        max_distance = max(abs(x_distance), abs(y_distance))
        if max_distance < distance_threshold:
            return 1.0
        return distance_threshold / max_distance if max_distance > 0 else 1.0

    if relation == "left_of":
        if x_distance <= 0:
            return 0.0
        if abs(x_distance) <= abs(y_distance):
            return 0.0
        if iou < iou_threshold:
            return 1.0
        return iou_threshold / iou

    if relation == "right_of":
        if x_distance >= 0:
            return 0.0
        if abs(x_distance) <= abs(y_distance):
            return 0.0
        if iou < iou_threshold:
            return 1.0
        return iou_threshold / iou

    if relation == "above":
        if y_distance <= 0:
            return 0.0
        if abs(y_distance) <= abs(x_distance):
            return 0.0
        if iou < iou_threshold:
            return 1.0
        return iou_threshold / iou

    if relation == "below":
        if y_distance >= 0:
            return 0.0
        if abs(y_distance) <= abs(x_distance):
            return 0.0
        if iou < iou_threshold:
            return 1.0
        return iou_threshold / iou

    if relation == "on":
        x_overlap = max(0.0, min(subject_box["x_max"], object_box["x_max"]) - max(subject_box["x_min"], object_box["x_min"]))
        subject_width = max(1.0, subject_box["x_max"] - subject_box["x_min"])
        object_width = max(1.0, object_box["x_max"] - object_box["x_min"])
        horizontal_overlap = x_overlap / min(subject_width, object_width)
        vertical_gap = max(0.0, object_box["y_min"] - subject_box["y_max"])
        if object_center[1] <= subject_center[1]:
            return 0.0
        overlap_score = min(1.0, horizontal_overlap)
        gap_score = 1.0 if vertical_gap <= distance_threshold * 0.2 else max(0.0, 1.0 - vertical_gap / distance_threshold)
        return max(0.0, min(1.0, 0.7 * overlap_score + 0.3 * gap_score))

    raise ValueError(f"Unsupported 2D spatial relation for ablation scoring: {relation}")


def load_detector_assets(t2i_root: Path) -> tuple[Any, Any, list[str], Any, Any]:
    unidet_root = t2i_root / "UniDet_eval"
    sys.path.insert(0, str(unidet_root))
    os.chdir(unidet_root)
    from experts.model_bank import load_expert_model  # type: ignore
    from experts.obj_detection.generate_dataset import Dataset, collate_fn  # type: ignore

    obj_label_map = torch.load("dataset/detection_features.pt", map_location="cpu")["labels"]
    original_argv = sys.argv[:]
    try:
        sys.argv = [str(unidet_root / "2D_spatial_eval.py")]
        model, transform = load_expert_model(task="obj_detection", ckpt="RS200")
    finally:
        sys.argv = original_argv
    return model, transform, obj_label_map, Dataset, collate_fn


def detection_from_prediction(
    prediction: dict[str, Any],
    *,
    obj_label_map: list[str],
) -> list[Detection]:
    fields = prediction["instances"].get_fields()
    boxes = fields["pred_boxes"].tensor
    labels = fields["pred_classes"]
    scores = fields["scores"]

    detections: list[Detection] = []
    for index in range(len(boxes)):
        label_index = int(labels[index].item())
        box_tensor = boxes[index]
        detections.append(
            Detection(
                label=str(obj_label_map[label_index]).lower(),
                normalized_label=normalize_label(str(obj_label_map[label_index])),
                score=float(scores[index].item()),
                box={
                    "x_min": float(box_tensor[0].item()),
                    "y_min": float(box_tensor[1].item()),
                    "x_max": float(box_tensor[2].item()),
                    "y_max": float(box_tensor[3].item()),
                },
            )
        )
    return detections


def match_detections(detections: list[Detection], target_label: str) -> list[Detection]:
    aliases = label_aliases(target_label)
    return [
        detection
        for detection in detections
        if detection.label in aliases or detection.normalized_label in aliases
    ]


def summarize_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(records)
    parsed = [record for record in records if record["parse_success"]]
    parse_failures = total - len(parsed)
    detection_successes = sum(1 for record in parsed if record["detection_success"])
    relation_evaluable = [record for record in parsed if record["detection_success"]]
    relation_correct = sum(1 for record in relation_evaluable if record["relation_correct"])
    end_to_end_success = sum(1 for record in parsed if record["end_to_end_success"])

    detection_rate = detection_successes / len(parsed) if parsed else None
    relation_accuracy = relation_correct / len(relation_evaluable) if relation_evaluable else None
    soft_relation_score = (
        sum(float(record["best_relation_score"]) for record in relation_evaluable) / len(relation_evaluable)
        if relation_evaluable
        else None
    )
    end_to_end_rate = end_to_end_success / len(parsed) if parsed else None

    return {
        "num_images": total,
        "num_parse_success": len(parsed),
        "num_parse_failures": parse_failures,
        "num_detection_success": detection_successes,
        "num_relation_evaluable": len(relation_evaluable),
        "num_relation_correct": relation_correct,
        "num_end_to_end_success": end_to_end_success,
        "detection_success_rate": detection_rate,
        "relation_accuracy_given_detection": relation_accuracy,
        "mean_relation_score_given_detection": soft_relation_score,
        "end_to_end_success_rate": end_to_end_rate,
    }


def main() -> int:
    args = make_parser().parse_args()
    generated_dir = args.generated_dir.resolve()
    samples_dir = generated_dir / "samples"
    if not samples_dir.exists():
        raise FileNotFoundError(f"Missing generated samples dir: {samples_dir}")

    t2i_root = resolve_t2i_root(args.t2i_compbench_root)
    prompt_file = args.prompt_file.resolve() if args.prompt_file is not None else None
    prompt_set = set(load_prompt_lines(prompt_file)) if prompt_file is not None else None

    image_paths = list_sample_images(samples_dir)
    if args.limit_images is not None:
        image_paths = image_paths[: args.limit_images]
    if not image_paths:
        raise RuntimeError(f"No sample images found in {samples_dir}")

    original_cwd = Path.cwd()
    model, transform, obj_label_map, Dataset, collate_fn = load_detector_assets(t2i_root)
    try:
        dataset = Dataset(str(generated_dir), transform)
        if args.limit_images is not None and len(dataset.data_list) > args.limit_images:
            dataset.data_list = dataset.data_list[: args.limit_images]

        data_loader = torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=True,
            collate_fn=collate_fn,
        )

        records: list[dict[str, Any]] = []
        with torch.no_grad():
            for batch in tqdm(data_loader, desc="Spatial ablation"):
                predictions = model(batch)
                for sample, prediction in zip(batch, predictions):
                    image_path = Path(str(sample["image_path"]))
                    prompt = prompt_from_image_path(image_path)
                    record: dict[str, Any] = {
                        "image": image_path.name,
                        "image_index": image_index_from_path(image_path),
                        "prompt": prompt,
                        "prompt_in_prompt_file": prompt in prompt_set if prompt_set is not None else None,
                    }

                    try:
                        subject, relation, obj = parse_prompt(prompt)
                    except ValueError as exc:
                        record.update(
                            {
                                "parse_success": False,
                                "parse_error": str(exc),
                                "subject": None,
                                "relation": None,
                                "object": None,
                                "subject_matches": 0,
                                "object_matches": 0,
                                "detection_success": False,
                                "best_relation_score": None,
                                "relation_correct": False,
                                "end_to_end_success": False,
                                "detections": [],
                            }
                        )
                        records.append(record)
                        continue

                    detections = detection_from_prediction(prediction, obj_label_map=obj_label_map)
                    subject_matches = match_detections(detections, subject)
                    object_matches = match_detections(detections, obj)
                    detection_success = bool(subject_matches) and bool(object_matches)

                    best_pair: dict[str, Any] | None = None
                    best_relation_score: float | None = None
                    if detection_success:
                        for subject_detection in subject_matches:
                            for object_detection in object_matches:
                                score = relation_score(relation, subject_detection.box, object_detection.box)
                                if best_relation_score is None or score > best_relation_score:
                                    best_relation_score = score
                                    best_pair = {
                                        "subject_label": subject_detection.label,
                                        "subject_normalized_label": subject_detection.normalized_label,
                                        "subject_score": subject_detection.score,
                                        "subject_box": subject_detection.box,
                                        "object_label": object_detection.label,
                                        "object_normalized_label": object_detection.normalized_label,
                                        "object_score": object_detection.score,
                                        "object_box": object_detection.box,
                                    }

                    relation_correct = bool(
                        detection_success and best_relation_score is not None and best_relation_score >= args.relation_threshold
                    )
                    record.update(
                        {
                            "parse_success": True,
                            "parse_error": None,
                            "subject": subject,
                            "relation": relation,
                            "object": obj,
                            "subject_matches": len(subject_matches),
                            "object_matches": len(object_matches),
                            "detection_success": detection_success,
                            "best_relation_score": best_relation_score,
                            "relation_correct": relation_correct,
                            "end_to_end_success": bool(detection_success and relation_correct),
                            "best_pair": best_pair,
                            "detections": [
                                {
                                    "label": detection.label,
                                    "normalized_label": detection.normalized_label,
                                    "score": detection.score,
                                    "box": detection.box,
                                }
                                for detection in detections
                            ],
                        }
                    )
                    records.append(record)
    finally:
        os.chdir(original_cwd)

    summary = summarize_records(records)
    summary_payload = {
        "generated_dir": str(generated_dir),
        "samples_dir": str(samples_dir),
        "prompt_file": str(prompt_file) if prompt_file is not None else None,
        "t2i_compbench_root": str(t2i_root),
        "relation_threshold": args.relation_threshold,
        **summary,
    }

    summary_path = args.output_file.resolve() if args.output_file is not None else generated_dir / "t2i_spatial_ablation_summary.json"
    records_path = args.records_file.resolve() if args.records_file is not None else generated_dir / "t2i_spatial_ablation_records.json"
    summary_path.write_text(json.dumps(summary_payload, indent=2))
    records_path.write_text(json.dumps(records, indent=2))

    print(
        "Spatial ablation: "
        f"detection={summary['detection_success_rate']:.6f} "
        f"relation|det={summary['relation_accuracy_given_detection']:.6f} "
        f"e2e={summary['end_to_end_success_rate']:.6f}"
        if summary["detection_success_rate"] is not None
        else "Spatial ablation completed, but no prompts were parseable."
    )
    print(f"Summary saved to {summary_path}")
    print(f"Per-image records saved to {records_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
