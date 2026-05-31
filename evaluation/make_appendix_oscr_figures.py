"""Generate appendix-ready qualitative OSCR layout figures.

This script renders the actual SeeThrough3D-style OSCR cuboids from ground-truth
SCOP-Depth boxes and graph-layout model predictions. It does not run FLUX image
generation; the goal is to make compact layout grids for the paper appendix.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from PIL import Image, ImageDraw, ImageFont

from training.config import _load_raw_config
from training.dataset import build_dataset_splits, load_metadata_rows
from training.prompts import prompt_from_scop_depth_row
from training.runtime import DEFAULT_FLUX_MODEL_ID, resolve_torch_device
from training.seethrough_condition import render_seethrough_oscr_and_masks

from evaluation.evaluate_layout_models import (
    DEFAULT_OCCLUSION_OVERLAP_THRESHOLD,
    GraphModelPredictor,
    LayoutPrediction,
    evaluate_prediction,
    normalize_prompt,
    relation_metrics,
    relation_triplets,
    row_target_layout,
)


@dataclass(frozen=True)
class RenderedPanel:
    title: str
    image: Image.Image
    metadata: dict[str, Any]


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Render qualitative OSCR grids for appendix figures.")
    parser.add_argument("--config", type=Path, required=True)
    return parser


def tensor_oscr_to_pil(oscr: torch.Tensor) -> Image.Image:
    array = (
        oscr.detach()
        .cpu()
        .clamp(-1.0, 1.0)
        .add(1.0)
        .mul(127.5)
        .round()
        .to(torch.uint8)
        .permute(1, 2, 0)
        .numpy()
    )
    return Image.fromarray(array, mode="RGB")


def render_prediction_oscr(
    prediction: LayoutPrediction,
    *,
    image_size: int,
    face_alpha: float,
    azimuth_degrees: float,
) -> Image.Image:
    centers = prediction.centers.unsqueeze(0)
    log_sizes = prediction.sizes.clamp_min(1e-6).log().unsqueeze(0)
    slot_mask = torch.ones(1, prediction.centers.shape[0], dtype=torch.bool)
    oscr, _masks = render_seethrough_oscr_and_masks(
        centers=centers,
        log_sizes=log_sizes,
        slot_mask=slot_mask,
        image_size=image_size,
        face_alpha=face_alpha,
        azimuth_degrees=azimuth_degrees,
    )
    return tensor_oscr_to_pil(oscr[0])


def default_font(size: int) -> ImageFont.ImageFont:
    for font_path in (
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/System/Library/Fonts/Supplemental/Helvetica.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ):
        path = Path(font_path)
        if path.exists():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


def add_title(image: Image.Image, title: str, *, title_height: int = 42) -> Image.Image:
    output = Image.new("RGB", (image.width, image.height + title_height), "white")
    output.paste(image, (0, title_height))
    draw = ImageDraw.Draw(output)
    font = default_font(18)
    text = title.strip()
    bbox = draw.textbbox((0, 0), text, font=font)
    text_width = bbox[2] - bbox[0]
    draw.text(((image.width - text_width) // 2, 12), text, fill=(25, 25, 25), font=font)
    return output


def compose_grid(
    panels: list[RenderedPanel],
    *,
    columns: int,
    output_path: Path,
    title: str | None = None,
    gap: int = 16,
    margin: int = 20,
) -> None:
    titled = [add_title(panel.image, panel.title) for panel in panels]
    if not titled:
        raise ValueError("Cannot compose an empty figure grid")

    columns = max(1, min(columns, len(titled)))
    rows = math.ceil(len(titled) / columns)
    cell_w = max(image.width for image in titled)
    cell_h = max(image.height for image in titled)
    header_h = 56 if title else 0
    width = margin * 2 + columns * cell_w + (columns - 1) * gap
    height = margin * 2 + header_h + rows * cell_h + (rows - 1) * gap
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    if title:
        font = default_font(24)
        bbox = draw.textbbox((0, 0), title, font=font)
        draw.text(((width - (bbox[2] - bbox[0])) // 2, margin), title, fill=(20, 20, 20), font=font)

    for index, image in enumerate(titled):
        row = index // columns
        col = index % columns
        x = margin + col * (cell_w + gap) + (cell_w - image.width) // 2
        y = margin + header_h + row * (cell_h + gap) + (cell_h - image.height) // 2
        canvas.paste(image, (x, y))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)


def write_individual_panels(panels: list[RenderedPanel], output_dir: Path, figure_name: str) -> None:
    panel_dir = output_dir / "individual" / figure_name
    panel_dir.mkdir(parents=True, exist_ok=True)
    manifest = []
    for index, panel in enumerate(panels):
        safe_title = "".join(ch if ch.isalnum() else "_" for ch in panel.title.lower()).strip("_")[:64]
        path = panel_dir / f"{index:02d}_{safe_title}.png"
        panel.image.save(path)
        manifest.append({"index": index, "title": panel.title, "path": str(path), **panel.metadata})
    (panel_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))


def load_rows(raw_config: dict[str, Any]) -> list[dict[str, Any]]:
    dataset_config = raw_config.get("dataset", {})
    dataset_dir = Path(dataset_config["dataset_dir"])
    prompt_prefix = str(dataset_config.get("prompt_prefix", "a photo of"))
    image_size = int(dataset_config.get("image_size", 512))
    split_seed = int(dataset_config.get("split_seed", 42))
    eval_fraction = float(dataset_config.get("eval_fraction", 0.05))
    test_fraction = float(dataset_config.get("test_fraction", 0.05))
    split = str(dataset_config.get("split", "test"))

    if split == "all":
        return load_metadata_rows(dataset_dir)

    splits = build_dataset_splits(
        dataset_dir,
        image_size=image_size,
        prompt_prefix=prompt_prefix,
        seed=split_seed,
        eval_fraction=eval_fraction,
        test_fraction=test_fraction,
        load_images=False,
    )
    if split not in splits:
        raise ValueError(f"Unknown split {split!r}; expected one of train/eval/test/all")
    return list(splits[split].rows)


def find_row(rows: list[dict[str, Any]], prompt: str, *, prompt_prefix: str) -> dict[str, Any]:
    target = normalize_prompt(prompt)
    for row in rows:
        if normalize_prompt(prompt_from_scop_depth_row(row, prefix=prompt_prefix)) == target:
            return row
    raise ValueError(f"No row matched prompt in selected split: {prompt}")


def load_predictors(raw_config: dict[str, Any]) -> dict[str, GraphModelPredictor]:
    runtime = raw_config.get("runtime", {})
    model_id = str(runtime.get("model_id", DEFAULT_FLUX_MODEL_ID))
    device = str(runtime.get("device", "auto"))
    mixed_precision = str(runtime.get("mixed_precision", "bf16"))
    resolved_device = str(resolve_torch_device(device))
    text_runtime_cache: dict[tuple[str, str, torch.dtype], tuple[object, object, int]] = {}

    predictors: dict[str, GraphModelPredictor] = {}
    for name, checkpoint in raw_config.get("checkpoints", {}).items():
        predictors[str(name)] = GraphModelPredictor(
            checkpoint=Path(checkpoint),
            model_id=model_id,
            device=resolved_device,
            mixed_precision=mixed_precision,
            text_runtime_cache=text_runtime_cache,
        )
    return predictors


def predict_samples(
    predictor: GraphModelPredictor,
    row: dict[str, Any],
    *,
    num_samples: int,
    layout_sample_mode: str,
    layout_z_scale: float,
    seed: int,
) -> list[LayoutPrediction]:
    predictions = []
    for sample_index in range(num_samples):
        sample_seed = seed + sample_index
        torch.manual_seed(sample_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(sample_seed)
        predictions.append(
            predictor.predict(
                row,
                layout_sample_mode=layout_sample_mode,
                layout_z_scale=layout_z_scale,
            )
        )
    return predictions


def make_panel(
    title: str,
    prediction: LayoutPrediction,
    *,
    image_size: int,
    face_alpha: float,
    azimuth_degrees: float,
    metadata: dict[str, Any] | None = None,
) -> RenderedPanel:
    return RenderedPanel(
        title=title,
        image=render_prediction_oscr(
            prediction,
            image_size=image_size,
            face_alpha=face_alpha,
            azimuth_degrees=azimuth_degrees,
        ),
        metadata=metadata or {},
    )


def figure_deterministic_vs_relay(
    *,
    name: str,
    figure_config: dict[str, Any],
    rows: list[dict[str, Any]],
    predictors: dict[str, GraphModelPredictor],
    prompt_prefix: str,
    render_config: dict[str, Any],
) -> list[RenderedPanel]:
    row = find_row(rows, str(figure_config["prompt"]), prompt_prefix=prompt_prefix)
    image_size = int(render_config.get("image_size", 512))
    face_alpha = float(render_config.get("face_alpha", 0.10))
    azimuth_degrees = float(render_config.get("azimuth_degrees", 0.0))
    seed = int(figure_config.get("seed", 42))
    relay_samples = int(figure_config.get("relay_samples", 6))
    deterministic_key = str(figure_config.get("deterministic_checkpoint", "deterministic"))
    relay_key = str(figure_config.get("relay_checkpoint", "relay"))
    relay_z_scale = float(figure_config.get("layout_z_scale", 1.0))

    panels = [
        make_panel(
            "GT pseudo-3D",
            row_target_layout(row),
            image_size=image_size,
            face_alpha=face_alpha,
            azimuth_degrees=azimuth_degrees,
            metadata={"kind": "ground_truth"},
        )
    ]
    deterministic = predict_samples(
        predictors[deterministic_key],
        row,
        num_samples=1,
        layout_sample_mode="prior_mean",
        layout_z_scale=1.0,
        seed=seed,
    )[0]
    panels.append(
        make_panel(
            "Deterministic GNN",
            deterministic,
            image_size=image_size,
            face_alpha=face_alpha,
            azimuth_degrees=azimuth_degrees,
            metadata={"kind": "deterministic", "checkpoint": deterministic_key},
        )
    )
    relay_predictions = predict_samples(
        predictors[relay_key],
        row,
        num_samples=relay_samples,
        layout_sample_mode="prior_sample",
        layout_z_scale=relay_z_scale,
        seed=seed,
    )
    for sample_index, prediction in enumerate(relay_predictions):
        panels.append(
            make_panel(
                f"RELAY-3D sample {sample_index + 1}",
                prediction,
                image_size=image_size,
                face_alpha=face_alpha,
                azimuth_degrees=azimuth_degrees,
                metadata={"kind": "relay_sample", "sample_index": sample_index, "checkpoint": relay_key},
            )
        )
    return panels


def figure_augmentation_comparison(
    *,
    name: str,
    figure_config: dict[str, Any],
    rows: list[dict[str, Any]],
    predictors: dict[str, GraphModelPredictor],
    prompt_prefix: str,
    render_config: dict[str, Any],
) -> list[RenderedPanel]:
    row = find_row(rows, str(figure_config["prompt"]), prompt_prefix=prompt_prefix)
    image_size = int(render_config.get("image_size", 512))
    face_alpha = float(render_config.get("face_alpha", 0.10))
    azimuth_degrees = float(render_config.get("azimuth_degrees", 0.0))
    seed = int(figure_config.get("seed", 42))
    original_key = str(figure_config.get("original_checkpoint", "deterministic"))
    augmented_key = str(figure_config.get("augmented_checkpoint", "deterministic_aug"))
    panels = [
        make_panel(
            "GT pseudo-3D",
            row_target_layout(row),
            image_size=image_size,
            face_alpha=face_alpha,
            azimuth_degrees=azimuth_degrees,
            metadata={"kind": "ground_truth"},
        )
    ]
    for title, key in (("Original-data GNN", original_key), ("Prompt-balanced GNN", augmented_key)):
        prediction = predict_samples(
            predictors[key],
            row,
            num_samples=1,
            layout_sample_mode="prior_mean",
            layout_z_scale=1.0,
            seed=seed,
        )[0]
        panels.append(
            make_panel(
                title,
                prediction,
                image_size=image_size,
                face_alpha=face_alpha,
                azimuth_degrees=azimuth_degrees,
                metadata={"kind": "deterministic", "checkpoint": key},
            )
        )
    return panels


def sample_eval_metrics(row: dict[str, Any], prediction: LayoutPrediction) -> dict[str, float]:
    target = row_target_layout(row)
    metrics = evaluate_prediction(row=row, prediction=prediction, target=target, prompt_references={})
    metrics.update(
        relation_metrics(
            prediction.centers,
            relation_triplets(row),
            boxes=prediction.boxes,
            occlusion_overlap_threshold=DEFAULT_OCCLUSION_OVERLAP_THRESHOLD,
        )
    )
    return metrics


def is_success(metrics: dict[str, float]) -> bool:
    return (
        metrics.get("rel_acc") == 1.0
        and metrics.get("oob_rate") == 0.0
        and (math.isnan(metrics.get("rel_3d_acc", math.nan)) or metrics.get("rel_3d_acc") == 1.0)
    )


def is_failure(metrics: dict[str, float]) -> bool:
    rel_acc = metrics.get("rel_acc", math.nan)
    oob_rate = metrics.get("oob_rate", math.nan)
    box_l1 = metrics.get("box_l1", 0.0)
    iou_3d = metrics.get("iou_3d", 1.0)
    return (
        (not math.isnan(rel_acc) and rel_acc < 1.0)
        or (not math.isnan(oob_rate) and oob_rate > 0.0)
        or box_l1 > 0.30
        or iou_3d < 0.05
    )


def figure_success_failure(
    *,
    name: str,
    figure_config: dict[str, Any],
    rows: list[dict[str, Any]],
    predictors: dict[str, GraphModelPredictor],
    prompt_prefix: str,
    render_config: dict[str, Any],
) -> list[RenderedPanel]:
    image_size = int(render_config.get("image_size", 512))
    face_alpha = float(render_config.get("face_alpha", 0.10))
    azimuth_degrees = float(render_config.get("azimuth_degrees", 0.0))
    checkpoint_key = str(figure_config.get("relay_checkpoint", "relay"))
    seed = int(figure_config.get("seed", 123))
    samples_per_row = int(figure_config.get("samples_per_row", 8))
    success_count = int(figure_config.get("success_count", 2))
    failure_count = int(figure_config.get("failure_count", 2))
    max_scan_rows = int(figure_config.get("max_scan_rows", 256))

    explicit_cases = figure_config.get("cases") or []
    panels: list[RenderedPanel] = []
    if explicit_cases:
        for case_index, case in enumerate(explicit_cases):
            row = find_row(rows, str(case["prompt"]), prompt_prefix=prompt_prefix)
            mode = str(case.get("layout_sample_mode", "prior_sample"))
            sample_index = int(case.get("sample_index", 0))
            predictions = predict_samples(
                predictors[checkpoint_key],
                row,
                num_samples=sample_index + 1,
                layout_sample_mode=mode,
                layout_z_scale=float(case.get("layout_z_scale", 1.0)),
                seed=seed + case_index * 1000,
            )
            prediction = predictions[sample_index]
            metrics = sample_eval_metrics(row, prediction)
            panels.append(
                make_panel(
                    str(case.get("title", f"{case.get('tag', 'case')} {case_index + 1}")),
                    prediction,
                    image_size=image_size,
                    face_alpha=face_alpha,
                    azimuth_degrees=azimuth_degrees,
                    metadata={"kind": "explicit_case", "metrics": metrics},
                )
            )
        return panels

    successes: list[RenderedPanel] = []
    failures: list[RenderedPanel] = []
    for row_index, row in enumerate(rows[:max_scan_rows]):
        predictions = predict_samples(
            predictors[checkpoint_key],
            row,
            num_samples=samples_per_row,
            layout_sample_mode="prior_sample",
            layout_z_scale=float(figure_config.get("layout_z_scale", 1.0)),
            seed=seed + row_index * 1009,
        )
        prompt = normalize_prompt(prompt_from_scop_depth_row(row, prefix=prompt_prefix))
        for sample_index, prediction in enumerate(predictions):
            metrics = sample_eval_metrics(row, prediction)
            if len(successes) < success_count and is_success(metrics):
                successes.append(
                    make_panel(
                        f"Success: {prompt}",
                        prediction,
                        image_size=image_size,
                        face_alpha=face_alpha,
                        azimuth_degrees=azimuth_degrees,
                        metadata={"kind": "success", "sample_index": sample_index, "metrics": metrics},
                    )
                )
            if len(failures) < failure_count and is_failure(metrics):
                failures.append(
                    make_panel(
                        f"Failure: {prompt}",
                        prediction,
                        image_size=image_size,
                        face_alpha=face_alpha,
                        azimuth_degrees=azimuth_degrees,
                        metadata={"kind": "failure", "sample_index": sample_index, "metrics": metrics},
                    )
                )
            if len(successes) >= success_count and len(failures) >= failure_count:
                return successes + failures
    return successes + failures


def main() -> int:
    args = make_parser().parse_args()
    raw_config = _load_raw_config(args.config)
    rows = load_rows(raw_config)
    dataset_config = raw_config.get("dataset", {})
    prompt_prefix = str(dataset_config.get("prompt_prefix", "a photo of"))
    output_dir = Path(raw_config.get("output", {}).get("output_dir", "outputs/debug/appendix_oscr_figures"))
    figure_dir = output_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    predictors = load_predictors(raw_config)
    render_config = raw_config.get("rendering", {})
    summary: list[dict[str, Any]] = []

    figure_builders = {
        "deterministic_vs_relay": figure_deterministic_vs_relay,
        "augmentation_comparison": figure_augmentation_comparison,
        "success_failure": figure_success_failure,
    }

    for figure_config in raw_config.get("figures", []):
        name = str(figure_config["name"])
        figure_type = str(figure_config["type"])
        if figure_type not in figure_builders:
            raise ValueError(f"Unknown figure type {figure_type!r}; expected one of {sorted(figure_builders)}")
        panels = figure_builders[figure_type](
            name=name,
            figure_config=figure_config,
            rows=rows,
            predictors=predictors,
            prompt_prefix=prompt_prefix,
            render_config=render_config,
        )
        if not panels:
            raise RuntimeError(f"Figure {name!r} produced no panels")
        write_individual_panels(panels, output_dir, name)
        output_path = figure_dir / f"{name}.png"
        compose_grid(
            panels,
            columns=int(figure_config.get("columns", min(4, len(panels)))),
            output_path=output_path,
            title=figure_config.get("title"),
        )
        summary.append({"name": name, "type": figure_type, "path": str(output_path), "panels": len(panels)})
        print(f"Wrote {output_path}")

    (output_dir / "appendix_oscr_manifest.json").write_text(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
