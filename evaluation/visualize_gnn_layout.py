from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import torch
from PIL import Image, ImageDraw, ImageFont
from transformers import CLIPTextModel, CLIPTokenizer

from evaluation.generate import MODEL_REGISTRY, load_graph_encoder, resolve_torch_device
from evaluation.prompt_parser import parse_prompt_to_scene_graph
from training.dataset import load_metadata_rows
from training.graph_modules import build_slot_conditioning
from training.graph_targets import bbox_centers_after_crop, bbox_log_sigmas_after_crop
from training.prompts import prompt_from_scop_depth_row, scene_graph_payload_from_row
from training.scene_graph import INVERSE_RELATION, build_batched_scene_graphs


CANVAS_SIZE = 512


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Visualize GNN-predicted slot layout against SCOP-Depth bbox center/sigma targets."
    )
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=None,
        help="Optional SCOP-Depth export for GT comparison. If omitted or no prompt match is found, saves prediction-only visuals.",
    )
    parser.add_argument("--graph-encoder-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--row-index", type=int, default=None)
    parser.add_argument("--model", choices=sorted(MODEL_REGISTRY.keys()), default="sd15")
    parser.add_argument("--model-id", type=str, default=None)
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    return parser


def _load_font(size: int = 22) -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size)
    except IOError:
        return ImageFont.load_default()


def _wrap_text(text: str, *, max_chars: int) -> list[str]:
    words = text.split()
    lines: list[str] = []
    current: list[str] = []
    for word in words:
        candidate = " ".join(current + [word])
        if len(candidate) > max_chars and current:
            lines.append(" ".join(current))
            current = [word]
        else:
            current.append(word)
    if current:
        lines.append(" ".join(current))
    return lines


def _draw_text_block(
    draw: ImageDraw.ImageDraw,
    lines: list[str],
    xy: tuple[int, int],
    font: ImageFont.ImageFont,
    *,
    fill: str = "white",
    background: str = "black",
) -> None:
    if not lines:
        return
    x, y = xy
    padding = 8
    line_gap = 4
    boxes = [draw.textbbox((0, 0), line, font=font) for line in lines]
    widths = [right - left for left, top, right, bottom in boxes]
    heights = [bottom - top for left, top, right, bottom in boxes]
    block_w = max(widths) + 2 * padding
    block_h = sum(heights) + line_gap * (len(lines) - 1) + 2 * padding
    draw.rectangle([x, y, x + block_w, y + block_h], fill=background)
    cursor_y = y + padding
    for line, height in zip(lines, heights):
        draw.text((x + padding, cursor_y), line, font=font, fill=fill)
        cursor_y += height + line_gap


def _normalize_prompt(text: str) -> str:
    return " ".join(text.lower().strip().split())


def _find_row(
    *,
    prompt: str,
    rows: list[dict[str, Any]],
    row_index: int | None,
) -> tuple[int, dict[str, Any]]:
    if row_index is not None:
        if row_index < 0 or row_index >= len(rows):
            raise IndexError(f"row-index {row_index} is outside metadata range 0..{len(rows)-1}")
        return row_index, rows[row_index]

    target = _normalize_prompt(prompt)
    for index, row in enumerate(rows):
        if _normalize_prompt(prompt_from_scop_depth_row(row)) == target:
            return index, row

    raise ValueError(
        "Could not find an exact prompt match in metadata.jsonl. "
        "Pass --row-index to visualize a specific metadata row."
    )


def _edges_with_inverses(scene_graph: dict[str, Any]) -> list[dict[str, Any]]:
    """Return raw graph edges plus the inverse edges used for message passing."""

    edges: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    for edge in scene_graph["edges"]:
        forward = (edge["source_id"], edge["target_id"], edge["relation"])
        inverse = (
            edge["target_id"],
            edge["source_id"],
            INVERSE_RELATION[edge["relation"]],
        )
        for source_id, target_id, relation in (forward, inverse):
            key = (source_id, target_id, relation)
            if key in seen:
                continue
            seen.add(key)
            edges.append(
                {
                    "source_id": source_id,
                    "target_id": target_id,
                    "relation": relation,
                    "is_inverse": key == inverse,
                }
            )
    return edges


def _to_float_list(tensor: torch.Tensor) -> list[float]:
    return [float(value) for value in tensor.detach().cpu().to(torch.float32).tolist()]


def _coord_to_px(x: float, y: float, size: int = CANVAS_SIZE) -> tuple[float, float]:
    return (x + 1.0) * 0.5 * size, (y + 1.0) * 0.5 * size


def _sigma_to_px(sigma_x: float, sigma_y: float, size: int = CANVAS_SIZE) -> tuple[float, float]:
    return sigma_x * size * 0.5, sigma_y * size * 0.5


def _draw_layout(
    draw: ImageDraw.ImageDraw,
    *,
    centers: list[list[float]],
    sigmas: list[list[float]],
    labels: list[str],
    colors: list[str],
    offset_x: int,
    offset_y: int,
    title: str,
    font: ImageFont.ImageFont,
    solid: bool,
) -> None:
    draw.rectangle(
        [offset_x, offset_y, offset_x + CANVAS_SIZE, offset_y + CANVAS_SIZE],
        fill="#f7f3e8",
        outline="#1f2933",
        width=2,
    )
    draw.line(
        [offset_x, offset_y + CANVAS_SIZE // 2, offset_x + CANVAS_SIZE, offset_y + CANVAS_SIZE // 2],
        fill="#c8bfae",
        width=1,
    )
    draw.line(
        [offset_x + CANVAS_SIZE // 2, offset_y, offset_x + CANVAS_SIZE // 2, offset_y + CANVAS_SIZE],
        fill="#c8bfae",
        width=1,
    )
    draw.text((offset_x, offset_y - 30), title, font=font, fill="#111827")

    for index, (center, sigma, label) in enumerate(zip(centers, sigmas, labels)):
        x, y = center[:2]
        sigma_x, sigma_y = sigma[:2]
        px, py = _coord_to_px(x, y)
        rx, ry = _sigma_to_px(sigma_x, sigma_y)
        color = colors[index % len(colors)]
        box = [
            offset_x + px - rx,
            offset_y + py - ry,
            offset_x + px + rx,
            offset_y + py + ry,
        ]
        if solid:
            draw.ellipse(box, outline=color, width=4)
        else:
            for expand in (0, 4, 8):
                draw.ellipse(
                    [box[0] - expand, box[1] - expand, box[2] + expand, box[3] + expand],
                    outline=color,
                    width=2,
                )
        draw.ellipse(
            [offset_x + px - 6, offset_y + py - 6, offset_x + px + 6, offset_y + py + 6],
            fill=color,
            outline="black",
            width=1,
        )
        draw.text((offset_x + px + 8, offset_y + py + 8), label, font=font, fill=color)


def _render_scene_graph(
    *,
    prompt: str,
    scene_graph: dict[str, Any],
    output_path: Path,
) -> None:
    font = _load_font(22)
    small = _load_font(18)
    image = Image.new("RGB", (900, 520), "#f7f3e8")
    draw = ImageDraw.Draw(image)
    _draw_text_block(draw, ["Prompt:"] + _wrap_text(prompt, max_chars=70), (20, 20), small)

    nodes = scene_graph["nodes"]
    positions = [(250, 285), (650, 285)]
    colors = ["#0077b6", "#d62828"]
    for node, pos, color in zip(nodes, positions, colors):
        x, y = pos
        draw.ellipse([x - 90, y - 55, x + 90, y + 55], fill="white", outline=color, width=4)
        draw.text((x - 55, y - 12), str(node["label"]), font=font, fill=color)
        draw.text((x - 25, y + 18), str(node["id"]), font=small, fill="#334e68")

    for edge_index, edge in enumerate(_edges_with_inverses(scene_graph)):
        src_idx = 0 if edge["source_id"] == nodes[0]["id"] else 1
        dst_idx = 0 if edge["target_id"] == nodes[0]["id"] else 1
        sx, sy = positions[src_idx]
        dx, dy = positions[dst_idx]
        curve_offset = -36 if edge_index % 2 == 0 else 36
        start = (sx + (80 if dx > sx else -80), sy + curve_offset)
        end = (dx - (80 if dx > sx else -80), dy + curve_offset)
        line_color = "#111827" if not edge.get("is_inverse") else "#6b7280"
        draw.line([start, end], fill=line_color, width=4)
        angle = math.atan2(end[1] - start[1], end[0] - start[0])
        arrow_len = 18
        for sign in (-1, 1):
            theta = angle + sign * 2.55
            arrow = (end[0] + arrow_len * math.cos(theta), end[1] + arrow_len * math.sin(theta))
            draw.line([end, arrow], fill=line_color, width=4)
        mid = ((start[0] + end[0]) // 2 - 86, (start[1] + end[1]) // 2 - 28)
        _draw_text_block(draw, [str(edge["relation"])], mid, small, background="#111827")

    image.save(output_path)


def _render_layout_comparison(
    *,
    labels: list[str],
    gt_centers: list[list[float]],
    gt_sigmas: list[list[float]],
    pred_centers: list[list[float]],
    pred_sigmas: list[list[float]],
    output_path: Path,
) -> None:
    font = _load_font(20)
    image = Image.new("RGB", (1100, 640), "white")
    draw = ImageDraw.Draw(image)
    colors = ["#0077b6", "#d62828", "#2a9d8f", "#f77f00"]
    draw.text((20, 18), "GNN Layout Comparison (normalized image/latent coordinates: x,y in [-1,1])", font=font, fill="#111827")
    _draw_layout(
        draw,
        centers=gt_centers,
        sigmas=gt_sigmas,
        labels=labels,
        colors=colors,
        offset_x=20,
        offset_y=90,
        title="Ground truth center / sigma",
        font=font,
        solid=True,
    )
    _draw_layout(
        draw,
        centers=pred_centers,
        sigmas=pred_sigmas,
        labels=labels,
        colors=colors,
        offset_x=560,
        offset_y=90,
        title="GNN predicted center / sigma",
        font=font,
        solid=False,
    )
    image.save(output_path)


def _render_prediction_layout(
    *,
    labels: list[str],
    pred_centers: list[list[float]],
    pred_sigmas: list[list[float]],
    output_path: Path,
) -> None:
    font = _load_font(20)
    image = Image.new("RGB", (620, 640), "white")
    draw = ImageDraw.Draw(image)
    colors = ["#0077b6", "#d62828", "#2a9d8f", "#f77f00"]
    draw.text(
        (20, 18),
        "GNN Predicted Layout (x,y in [-1,1])",
        font=font,
        fill="#111827",
    )
    _draw_layout(
        draw,
        centers=pred_centers,
        sigmas=pred_sigmas,
        labels=labels,
        colors=colors,
        offset_x=54,
        offset_y=90,
        title="GNN predicted center / sigma",
        font=font,
        solid=False,
    )
    image.save(output_path)


def _draw_overlay_on_crop(
    *,
    image_path: Path,
    labels: list[str],
    gt_centers: list[list[float]],
    gt_sigmas: list[list[float]],
    pred_centers: list[list[float]],
    pred_sigmas: list[list[float]],
    output_path: Path,
) -> None:
    base = Image.open(image_path).convert("RGB").resize((CANVAS_SIZE, CANVAS_SIZE), Image.Resampling.BICUBIC)
    panels = []
    font = _load_font(19)
    colors = ["#00d5ff", "#ff3366", "#2a9d8f", "#f77f00"]
    for title, centers, sigmas, solid in [
        ("GT on crop", gt_centers, gt_sigmas, True),
        ("Prediction on crop", pred_centers, pred_sigmas, False),
    ]:
        panel = base.copy()
        draw = ImageDraw.Draw(panel)
        draw.rectangle([0, 0, CANVAS_SIZE, 34], fill="black")
        draw.text((10, 7), title, font=font, fill="white")
        for index, (center, sigma, label) in enumerate(zip(centers, sigmas, labels)):
            px, py = _coord_to_px(center[0], center[1])
            rx, ry = _sigma_to_px(sigma[0], sigma[1])
            color = colors[index % len(colors)]
            box = [px - rx, py - ry, px + rx, py + ry]
            draw.ellipse(box, outline=color, width=4 if solid else 2)
            draw.ellipse([px - 5, py - 5, px + 5, py + 5], fill=color, outline="black")
            draw.text((px + 8, py + 8), label, font=font, fill=color)
        panels.append(panel)
    combined = Image.new("RGB", (CANVAS_SIZE * 2, CANVAS_SIZE), "black")
    combined.paste(panels[0], (0, 0))
    combined.paste(panels[1], (CANVAS_SIZE, 0))
    combined.save(output_path)


def _render_summary_image(
    *,
    prompt: str,
    report_lines: list[str],
    output_path: Path,
) -> None:
    font = _load_font(20)
    small = _load_font(17)
    image = Image.new("RGB", (1200, 760), "#f7f3e8")
    draw = ImageDraw.Draw(image)
    _draw_text_block(draw, ["Prompt:"] + _wrap_text(prompt, max_chars=90), (20, 20), small)
    y = 120
    draw.text((20, y), "Ground Truth vs GNN Prediction", font=font, fill="#111827")
    y += 42
    for line in report_lines[:28]:
        draw.text((28, y), line, font=small, fill="#111827")
        y += 24
    image.save(output_path)


def _build_report(
    *,
    row_index: int,
    row: dict[str, Any],
    prompt: str,
    scene_graph: dict[str, Any],
    labels: list[str],
    gt_centers: list[list[float]],
    gt_sigmas: list[list[float]],
    pred_centers: list[list[float]],
    pred_sigmas: list[list[float]],
) -> tuple[str, list[str], dict[str, Any]]:
    lines: list[str] = []
    lines.append(f"Prompt: {prompt}")
    lines.append(f"Matched metadata row index: {row_index}")
    lines.append(f"Image: {row['file_name']}")
    if "crop_box" in row:
        lines.append(f"Crop box in source image: {row['crop_box']}")
    lines.append("")
    lines.append("Scene graph:")
    for node in scene_graph["nodes"]:
        lines.append(f"  Node {node['id']}: {node['label']}")
    for edge in scene_graph["edges"]:
        lines.append(f"  Dataset edge {edge['source_id']} -> {edge['target_id']}: {edge['relation']}")
    lines.append("Effective message-passing edges:")
    for edge in _edges_with_inverses(scene_graph):
        kind = "inverse" if edge["is_inverse"] else "forward"
        lines.append(
            f"  {kind} edge {edge['source_id']} -> {edge['target_id']}: {edge['relation']}"
        )
    lines.append("")
    lines.append("Centers and sigmas use normalized image/latent coordinates.")
    lines.append("x,y are in [-1,1]. sigma values are normalized ellipse radii.")
    lines.append("")
    table_lines: list[str] = []
    for label, gt_center, gt_sigma, pred_center, pred_sigma in zip(
        labels,
        gt_centers,
        gt_sigmas,
        pred_centers,
        pred_sigmas,
    ):
        dx = pred_center[0] - gt_center[0]
        dy = pred_center[1] - gt_center[1]
        dz = pred_center[2] - gt_center[2]
        dsx = pred_sigma[0] - gt_sigma[0]
        dsy = pred_sigma[1] - gt_sigma[1]
        block = [
            f"{label}:",
            f"  GT center    = ({gt_center[0]:+.4f}, {gt_center[1]:+.4f}, {gt_center[2]:+.4f})",
            f"  Pred center  = ({pred_center[0]:+.4f}, {pred_center[1]:+.4f}, {pred_center[2]:+.4f})",
            f"  Center error = ({dx:+.4f}, {dy:+.4f}, {dz:+.4f})",
            f"  GT sigma     = ({gt_sigma[0]:.4f}, {gt_sigma[1]:.4f})",
            f"  Pred sigma   = ({pred_sigma[0]:.4f}, {pred_sigma[1]:.4f})",
            f"  Sigma error  = ({dsx:+.4f}, {dsy:+.4f})",
        ]
        lines.extend(block)
        lines.append("")
        table_lines.extend(block)
        table_lines.append("")

    payload = {
        "prompt": prompt,
        "matched_row_index": row_index,
        "file_name": row["file_name"],
        "crop_box": row.get("crop_box"),
        "scene_graph": scene_graph,
        "labels": labels,
        "ground_truth": [
            {"label": label, "center_xyz": center, "sigma_xy": sigma}
            for label, center, sigma in zip(labels, gt_centers, gt_sigmas)
        ],
        "prediction": [
            {"label": label, "center_xyz": center, "sigma_xy": sigma}
            for label, center, sigma in zip(labels, pred_centers, pred_sigmas)
        ],
    }
    return "\n".join(lines), table_lines, payload


def _build_prediction_only_report(
    *,
    prompt: str,
    scene_graph: dict[str, Any],
    labels: list[str],
    pred_centers: list[list[float]],
    pred_sigmas: list[list[float]],
) -> tuple[str, list[str], dict[str, Any]]:
    lines: list[str] = []
    lines.append(f"Prompt: {prompt}")
    lines.append("Mode: prediction-only prompt visualization")
    lines.append("")
    lines.append("Scene graph:")
    for node in scene_graph["nodes"]:
        lines.append(f"  Node {node['id']}: {node['label']}")
    for edge in scene_graph["edges"]:
        lines.append(f"  Prompt edge {edge['source_id']} -> {edge['target_id']}: {edge['relation']}")
    lines.append("Effective message-passing edges:")
    for edge in _edges_with_inverses(scene_graph):
        kind = "inverse" if edge["is_inverse"] else "forward"
        lines.append(
            f"  {kind} edge {edge['source_id']} -> {edge['target_id']}: {edge['relation']}"
        )
    lines.append("")
    lines.append("No ground-truth row was used. Values below are GNN predictions only.")
    lines.append("x,y are in [-1,1]. sigma values are normalized ellipse radii.")
    lines.append("")

    table_lines: list[str] = []
    for label, pred_center, pred_sigma in zip(labels, pred_centers, pred_sigmas):
        block = [
            f"{label}:",
            f"  Pred center = ({pred_center[0]:+.4f}, {pred_center[1]:+.4f}, {pred_center[2]:+.4f})",
            f"  Pred sigma  = ({pred_sigma[0]:.4f}, {pred_sigma[1]:.4f})",
        ]
        lines.extend(block)
        lines.append("")
        table_lines.extend(block)
        table_lines.append("")

    payload = {
        "prompt": prompt,
        "matched_row_index": None,
        "mode": "prediction_only",
        "scene_graph": scene_graph,
        "labels": labels,
        "prediction": [
            {"label": label, "center_xyz": center, "sigma_xy": sigma}
            for label, center, sigma in zip(labels, pred_centers, pred_sigmas)
        ],
    }
    return "\n".join(lines), table_lines, payload


def main() -> int:
    args = make_parser().parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    row_index: int | None = None
    row: dict[str, Any] | None = None
    image_size: tuple[int, int] | None = None
    prompt = args.prompt
    if args.dataset_dir is not None:
        rows = load_metadata_rows(args.dataset_dir)
        try:
            row_index, row = _find_row(prompt=args.prompt, rows=rows, row_index=args.row_index)
        except ValueError:
            if args.row_index is not None:
                raise
            print(
                "No exact metadata prompt match found. Falling back to prompt-only "
                "prediction visualization without ground truth."
            )
        if row is not None:
            prompt = prompt_from_scop_depth_row(row) if args.row_index is not None else args.prompt
            image_path = args.dataset_dir / row["file_name"]
            if not image_path.exists():
                raise FileNotFoundError(f"Could not find matched row image: {image_path}")
            image_size = Image.open(image_path).size
            scene_graph = scene_graph_payload_from_row(row)
        else:
            scene_graph = parse_prompt_to_scene_graph(prompt)
    else:
        if args.row_index is not None:
            raise ValueError("--row-index requires --dataset-dir")
        scene_graph = parse_prompt_to_scene_graph(prompt)

    prompt_relation: str | None = None
    try:
        parsed_prompt_graph = parse_prompt_to_scene_graph(prompt)
        prompt_relation = parsed_prompt_graph["edges"][0]["relation"]
    except ValueError:
        prompt_relation = None
    if prompt_relation is not None and prompt_relation not in {
        edge["relation"] for edge in scene_graph["edges"]
    }:
        print(
            "Warning: parsed prompt relation does not appear in matched metadata row. "
            "Using metadata scene graph for GT alignment."
        )

    device = resolve_torch_device(args.device)
    model_id = args.model_id or MODEL_REGISTRY[args.model]
    tokenizer = CLIPTokenizer.from_pretrained(model_id, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(model_id, subfolder="text_encoder").to(device)
    text_encoder.eval()
    graph_encoder = load_graph_encoder(
        path=args.graph_encoder_path,
        text_hidden_dim=text_encoder.config.hidden_size,
        device=device,
        dtype=text_encoder.dtype,
    )
    graph_encoder.eval()

    max_nodes = len(scene_graph["nodes"])
    if row is not None and image_size is not None:
        slot_targets, slot_mask = bbox_centers_after_crop(
            [row],
            [image_size],
            max_nodes=max_nodes,
            device=torch.device(device),
        )
        log_sigma_targets, _ = bbox_log_sigmas_after_crop(
            [row],
            [image_size],
            max_nodes=max_nodes,
            device=torch.device(device),
        )
    else:
        slot_targets = torch.zeros(
            1,
            max_nodes,
            3,
            device=torch.device(device),
            dtype=torch.float32,
        )
        slot_mask = torch.ones(
            1,
            max_nodes,
            device=torch.device(device),
            dtype=torch.bool,
        )
        log_sigma_targets = torch.zeros(
            1,
            max_nodes,
            2,
            device=torch.device(device),
            dtype=torch.float32,
        )
    batched_graph = build_batched_scene_graphs(
        [scene_graph],
        slot_targets=slot_targets,
        slot_mask=slot_mask,
    )
    with torch.no_grad():
        conditioning = build_slot_conditioning(
            tokenizer=tokenizer,
            text_encoder=text_encoder,
            scene_graph_batch=batched_graph,
            graph_encoder=graph_encoder,
            device=device,
        )

    labels = [str(node["label"]) for node in scene_graph["nodes"]]
    pred_centers = [_to_float_list(value) for value in conditioning.slot_positions[0, :max_nodes]]
    pred_sigmas = [_to_float_list(value.exp()) for value in conditioning.slot_log_sigmas[0, :max_nodes]]

    if row is not None and row_index is not None:
        gt_centers = [_to_float_list(value) for value in slot_targets[0, :max_nodes]]
        gt_sigmas = [_to_float_list(value.exp()) for value in log_sigma_targets[0, :max_nodes]]
        report, table_lines, payload = _build_report(
            row_index=row_index,
            row=row,
            prompt=prompt,
            scene_graph=scene_graph,
            labels=labels,
            gt_centers=gt_centers,
            gt_sigmas=gt_sigmas,
            pred_centers=pred_centers,
            pred_sigmas=pred_sigmas,
        )
    else:
        gt_centers = []
        gt_sigmas = []
        report, table_lines, payload = _build_prediction_only_report(
            prompt=prompt,
            scene_graph=scene_graph,
            labels=labels,
            pred_centers=pred_centers,
            pred_sigmas=pred_sigmas,
        )

    (args.output_dir / "gnn_layout_report.txt").write_text(report)
    (args.output_dir / "gnn_layout_values.json").write_text(json.dumps(payload, indent=2))
    _render_scene_graph(
        prompt=prompt,
        scene_graph=scene_graph,
        output_path=args.output_dir / "scene_graph.png",
    )
    if row is not None and args.dataset_dir is not None:
        image_path = args.dataset_dir / row["file_name"]
        _render_layout_comparison(
            labels=labels,
            gt_centers=gt_centers,
            gt_sigmas=gt_sigmas,
            pred_centers=pred_centers,
            pred_sigmas=pred_sigmas,
            output_path=args.output_dir / "layout_comparison.png",
        )
        _draw_overlay_on_crop(
            image_path=image_path,
            labels=labels,
            gt_centers=gt_centers,
            gt_sigmas=gt_sigmas,
            pred_centers=pred_centers,
            pred_sigmas=pred_sigmas,
            output_path=args.output_dir / "crop_overlay_comparison.png",
        )
    else:
        _render_prediction_layout(
            labels=labels,
            pred_centers=pred_centers,
            pred_sigmas=pred_sigmas,
            output_path=args.output_dir / "predicted_layout.png",
        )
    _render_summary_image(
        prompt=prompt,
        report_lines=table_lines,
        output_path=args.output_dir / "gnn_layout_summary.png",
    )
    print(report)
    print("")
    print(f"Saved scene graph image to {args.output_dir / 'scene_graph.png'}")
    if row is not None:
        print(f"Saved layout comparison to {args.output_dir / 'layout_comparison.png'}")
        print(f"Saved crop overlay comparison to {args.output_dir / 'crop_overlay_comparison.png'}")
    else:
        print(f"Saved predicted layout to {args.output_dir / 'predicted_layout.png'}")
    print(f"Saved summary image to {args.output_dir / 'gnn_layout_summary.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
