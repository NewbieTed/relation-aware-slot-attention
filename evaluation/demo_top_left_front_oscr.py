"""Preview a fixed-corner 3D OSCR variant from GNN-predicted boxes.

This script is intentionally separate from training and generation. It uses the
same prompt parser and graph encoder as the FLUX path, then renders diagnostic
images that compare the current lightweight OSCR with a fixed top-left/front
cuboid projection. The output is only for visual inspection.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from PIL import Image, ImageDraw, ImageFont

from evaluation.prompt_parser import parse_prompt_to_scene_graph
from training.graph_modules import build_slot_conditioning
from training.oscr_renderer import render_oscr_boxes
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


CANVAS_SIZE = 512
COLORS = ["#00d5ff", "#ff3366", "#2a9d8f", "#f77f00", "#ffd166", "#9b5de5"]


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Render side-by-side OSCR demos from GNN-predicted 3D boxes without "
            "changing the training/eval renderer."
        )
    )
    parser.add_argument("--prompt", action="append", default=[], help="Prompt to visualize. Can be repeated.")
    parser.add_argument("--prompt-file", type=Path, default=None, help="Optional newline prompt file.")
    parser.add_argument("--graph-encoder-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-id", type=str, default=DEFAULT_FLUX_MODEL_ID)
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    parser.add_argument("--image-size", type=int, default=CANVAS_SIZE)
    parser.add_argument(
        "--layout-sample-mode",
        choices=("prior_mean", "prior_sample"),
        default="prior_mean",
        help="Use prior_mean for deterministic layout or prior_sample for stochastic CVAE samples.",
    )
    parser.add_argument(
        "--num-layout-samples",
        type=int,
        default=1,
        help="Number of GNN layout samples to render per prompt.",
    )
    parser.add_argument(
        "--layout-seed",
        type=int,
        default=42,
        help="Base seed for reproducible stochastic layout samples.",
    )
    parser.add_argument(
        "--layout-z-scale",
        type=float,
        default=1.0,
        help="Debug multiplier for sampled latent displacement around the prior/posterior mean.",
    )
    parser.add_argument(
        "--debug-latents",
        action="store_true",
        help="Print prior/posterior latent stats and sampled z values for each layout sample.",
    )
    parser.add_argument(
        "--depth-offset-scale",
        type=float,
        default=0.18,
        help="How strongly predicted z-size shifts the back face up-left.",
    )
    parser.add_argument(
        "--front-alpha",
        type=int,
        default=18,
        help="RGBA alpha for front cuboid faces in the fixed-corner demo.",
    )
    parser.add_argument(
        "--side-alpha",
        type=int,
        default=12,
        help="RGBA alpha for side/top cuboid faces in the fixed-corner demo.",
    )
    parser.add_argument(
        "--back-alpha",
        type=int,
        default=8,
        help="RGBA alpha for back cuboid faces in the fixed-corner demo.",
    )
    parser.add_argument(
        "--edge-alpha",
        type=int,
        default=210,
        help="RGBA alpha for cuboid edges in the fixed-corner demo.",
    )
    return parser


def _load_font(size: int) -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size)
    except OSError:
        return ImageFont.load_default()


def _read_prompts(args: argparse.Namespace) -> list[str]:
    prompts = list(args.prompt)
    if args.prompt_file is not None:
        prompts.extend(line.strip() for line in args.prompt_file.read_text().splitlines() if line.strip())
    if not prompts:
        raise ValueError("Pass at least one --prompt or --prompt-file.")
    return prompts


def _safe_name(prompt: str, index: int) -> str:
    safe = "".join(ch.lower() if ch.isalnum() else "_" for ch in prompt).strip("_")
    while "__" in safe:
        safe = safe.replace("__", "_")
    return f"{index:03d}_{safe[:80] or 'prompt'}"


def _tensor_oscr_to_pil(oscr: torch.Tensor) -> Image.Image:
    image = (
        oscr.detach()
        .cpu()
        .to(torch.float32)
        .add(1.0)
        .mul(127.5)
        .clamp(0, 255)
        .byte()
        .permute(1, 2, 0)
        .numpy()
    )
    return Image.fromarray(image, mode="RGB")


def _coord_to_px(x: float, y: float, image_size: int) -> tuple[float, float]:
    return (x + 1.0) * 0.5 * image_size, (y + 1.0) * 0.5 * image_size


def _rect_from_center_size(
    center: torch.Tensor,
    size: torch.Tensor,
    *,
    image_size: int,
    dx: float = 0.0,
    dy: float = 0.0,
) -> tuple[float, float, float, float]:
    cx, cy = _coord_to_px(float(center[0]), float(center[1]), image_size)
    half_w = max(4.0, float(size[0]) * image_size * 0.5)
    half_h = max(4.0, float(size[1]) * image_size * 0.5)
    return (
        max(0.0, cx - half_w + dx),
        max(0.0, cy - half_h + dy),
        min(float(image_size), cx + half_w + dx),
        min(float(image_size), cy + half_h + dy),
    )


def _hex_to_rgba(hex_color: str, alpha: int) -> tuple[int, int, int, int]:
    value = hex_color.lstrip("#")
    return tuple(int(value[idx : idx + 2], 16) for idx in (0, 2, 4)) + (alpha,)


def _draw_text_box(
    draw: ImageDraw.ImageDraw,
    lines: list[str],
    xy: tuple[int, int],
    *,
    font: ImageFont.ImageFont,
    fill: str = "white",
) -> None:
    if not lines:
        return
    x, y = xy
    padding = 6
    line_gap = 3
    boxes = [draw.textbbox((0, 0), line, font=font) for line in lines]
    width = max(right - left for left, top, right, bottom in boxes) + padding * 2
    height = sum(bottom - top for left, top, right, bottom in boxes) + line_gap * (len(lines) - 1) + padding * 2
    draw.rectangle([x, y, x + width, y + height], fill=(0, 0, 0, 210))
    cursor = y + padding
    for line, box in zip(lines, boxes):
        draw.text((x + padding, cursor), line, font=font, fill=fill)
        cursor += box[3] - box[1] + line_gap


def render_fixed_corner_oscr(
    *,
    centers: torch.Tensor,
    log_sizes: torch.Tensor,
    slot_mask: torch.Tensor,
    labels: list[str],
    image_size: int,
    depth_offset_scale: float,
    front_alpha: int,
    side_alpha: int,
    back_alpha: int,
    edge_alpha: int,
) -> Image.Image:
    """Render a deterministic top-left/front cuboid projection.

    The front face is the predicted x/y box. The back face is shifted up-left by
    a distance derived from predicted z-size and z-center. Side polygons connect
    the two faces, so larger/deeper boxes produce more visible 3D structure.
    """

    image = Image.new("RGB", (image_size, image_size), "#101114")
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay, "RGBA")
    font = _load_font(15)

    centers_cpu = centers.detach().cpu().to(torch.float32)
    sizes_cpu = log_sizes.detach().cpu().to(torch.float32).exp().clamp(min=0.03, max=2.0)
    mask_cpu = slot_mask.detach().cpu()
    valid_indices = [
        index
        for index in range(centers_cpu.shape[1])
        if bool(mask_cpu[0, index].item())
    ]
    valid_indices.sort(key=lambda idx: float(centers_cpu[0, idx, 2].item()))

    for draw_order, slot_index in enumerate(valid_indices):
        center = centers_cpu[0, slot_index]
        size = sizes_cpu[0, slot_index]
        z_center = (float(center[2]) + 1.0) * 0.5
        z_size = float(size[2])
        offset = max(6.0, image_size * depth_offset_scale * (0.35 * z_center + 0.65 * z_size))
        back_dx = -offset
        back_dy = -offset
        front = _rect_from_center_size(center, size, image_size=image_size)
        back = _rect_from_center_size(center, size, image_size=image_size, dx=back_dx, dy=back_dy)
        x0, y0, x1, y1 = front
        bx0, by0, bx1, by1 = back
        color = COLORS[slot_index % len(COLORS)]
        face = _hex_to_rgba(color, max(0, min(255, front_alpha)))
        side_face = _hex_to_rgba(color, max(0, min(255, side_alpha)))
        back_face = _hex_to_rgba(color, max(0, min(255, back_alpha)))
        edge = _hex_to_rgba(color, max(0, min(255, edge_alpha)))

        # Visible side faces: top and left are emphasized to make the fixed
        # top-left/front corner readable.
        draw.polygon([(bx0, by0), (bx1, by1), (x1, y1), (x0, y0)], fill=side_face)
        draw.polygon([(bx0, by0), (x0, y0), (x0, y1), (bx0, by1)], fill=side_face)
        draw.rectangle(back, fill=back_face, outline=edge, width=2)
        draw.rectangle(front, fill=face, outline=edge, width=3)
        for start, end in [((bx0, by0), (x0, y0)), ((bx1, by1), (x1, y1)), ((bx0, by1), (x0, y1)), ((bx1, by0), (x1, y0))]:
            draw.line([start, end], fill=edge, width=2)

        cx, cy = _coord_to_px(float(center[0]), float(center[1]), image_size)
        draw.ellipse([cx - 4, cy - 4, cx + 4, cy + 4], fill=edge)
        label_x = int(max(4, min(image_size - 150, cx + 8)))
        label_y = int(max(4, min(image_size - 72, cy + 8)))
        _draw_text_box(
            draw,
            [
                labels[slot_index],
                f"c=({float(center[0]):+.2f},{float(center[1]):+.2f},{float(center[2]):+.2f})",
                f"s=({float(size[0]):.2f},{float(size[1]):.2f},{float(size[2]):.2f})",
            ],
            (label_x, label_y),
            font=font,
        )

    return Image.alpha_composite(image.convert("RGBA"), overlay).convert("RGB")


def _make_contact_sheet(
    *,
    prompt: str,
    current: Image.Image,
    fixed_corner: Image.Image,
    output_path: Path,
) -> None:
    title_font = _load_font(22)
    label_font = _load_font(18)
    width = current.width + fixed_corner.width
    height = current.height + 96
    sheet = Image.new("RGB", (width, height), "#0b0d10")
    draw = ImageDraw.Draw(sheet)
    draw.text((14, 12), prompt, font=title_font, fill="white")
    draw.text((14, 54), "Current lightweight OSCR", font=label_font, fill="#d0d7de")
    draw.text((current.width + 14, 54), "Fixed top-left/front cuboid OSCR demo", font=label_font, fill="#d0d7de")
    sheet.paste(current, (0, 96))
    sheet.paste(fixed_corner, (current.width, 96))
    sheet.save(output_path)


def _short_vector(tensor: torch.Tensor, *, count: int = 3) -> str:
    values = tensor.detach().cpu().to(torch.float32).flatten()
    shown = ", ".join(f"{float(value):+.4f}" for value in values[:count])
    suffix = ", ..." if values.numel() > count else ""
    return f"[{shown}{suffix}]"


def _latent_debug_payload(conditioning: object, node_count: int) -> dict[str, Any]:
    def _maybe_list(value: torch.Tensor | None) -> list[float] | None:
        if value is None:
            return None
        return value.detach().cpu().to(torch.float32).flatten()[:3].tolist()

    prior_mu = getattr(conditioning, "prior_mu", None)
    prior_logvar = getattr(conditioning, "prior_logvar", None)
    sampled_z = getattr(conditioning, "sampled_z", None)
    object_prior_mu = getattr(conditioning, "object_prior_mu", None)
    object_prior_logvar = getattr(conditioning, "object_prior_logvar", None)
    sampled_object_z = getattr(conditioning, "sampled_object_z", None)

    scene_std = torch.exp(0.5 * prior_logvar) if prior_logvar is not None else None
    scene_eps = (
        (sampled_z - prior_mu) / scene_std.clamp_min(1e-8)
        if sampled_z is not None and prior_mu is not None and scene_std is not None
        else None
    )
    object_std = (
        torch.exp(0.5 * object_prior_logvar[:, :node_count])
        if object_prior_logvar is not None
        else None
    )
    object_eps = (
        (sampled_object_z[:, :node_count] - object_prior_mu[:, :node_count])
        / object_std.clamp_min(1e-8)
        if sampled_object_z is not None
        and object_prior_mu is not None
        and object_std is not None
        else None
    )
    return {
        "scene_prior_mu_first3": _maybe_list(prior_mu),
        "scene_prior_std_first3": _maybe_list(scene_std),
        "scene_eps_first3": _maybe_list(scene_eps),
        "scene_sampled_z_first3": _maybe_list(sampled_z),
        "object_prior_mu_first3": _maybe_list(object_prior_mu[:, :node_count] if object_prior_mu is not None else None),
        "object_prior_std_first3": _maybe_list(object_std),
        "object_eps_first3": _maybe_list(object_eps),
        "object_sampled_z_first3": _maybe_list(
            sampled_object_z[:, :node_count] if sampled_object_z is not None else None
        ),
    }


def _print_latent_debug(
    *,
    prompt_index: int,
    sample_index: int,
    sample_seed: int,
    conditioning: object,
    centers: torch.Tensor,
    log_sizes: torch.Tensor,
    slot_mask: torch.Tensor,
    labels: list[str],
) -> None:
    prior_mu = getattr(conditioning, "prior_mu", None)
    prior_logvar = getattr(conditioning, "prior_logvar", None)
    sampled_z = getattr(conditioning, "sampled_z", None)
    object_prior_mu = getattr(conditioning, "object_prior_mu", None)
    object_prior_logvar = getattr(conditioning, "object_prior_logvar", None)
    sampled_object_z = getattr(conditioning, "sampled_object_z", None)
    print(f"\nLayout latent debug | prompt={prompt_index} sample={sample_index} seed={sample_seed}")
    if prior_mu is not None and prior_logvar is not None and sampled_z is not None:
        scene_std = torch.exp(0.5 * prior_logvar)
        scene_eps = (sampled_z - prior_mu) / scene_std.clamp_min(1e-8)
        print(f"  scene prior_mu first3: {_short_vector(prior_mu)}")
        print(f"  scene prior_std first3: {_short_vector(scene_std)}")
        print(f"  scene eps first3:      {_short_vector(scene_eps)}")
        print(f"  scene sampled_z first3:{_short_vector(sampled_z)}")
        print(
            "  scene summary: "
            f"mu_abs_mean={prior_mu.detach().abs().mean().item():.6f}, "
            f"std_mean={scene_std.detach().mean().item():.6f}, "
            f"z_std={sampled_z.detach().to(torch.float32).std().item():.6f}"
        )
    if object_prior_mu is not None and object_prior_logvar is not None and sampled_object_z is not None:
        valid_indices = torch.where(slot_mask[0].detach().cpu().to(torch.bool))[0].tolist()
        object_std = torch.exp(0.5 * object_prior_logvar)
        object_eps = (sampled_object_z - object_prior_mu) / object_std.clamp_min(1e-8)
        for slot_index in valid_indices:
            label = labels[slot_index] if slot_index < len(labels) else f"slot{slot_index}"
            print(f"  object[{slot_index}] {label} prior_mu first3: {_short_vector(object_prior_mu[:, slot_index])}")
            print(f"  object[{slot_index}] {label} prior_std first3:{_short_vector(object_std[:, slot_index])}")
            print(f"  object[{slot_index}] {label} eps first3:      {_short_vector(object_eps[:, slot_index])}")
            print(f"  object[{slot_index}] {label} sampled_z first3:{_short_vector(sampled_object_z[:, slot_index])}")
    sizes = log_sizes.detach().exp()
    for slot_index, label in enumerate(labels):
        print(
            f"  output[{slot_index}] {label}: "
            f"center={_short_vector(centers[:, slot_index], count=3)} "
            f"size={_short_vector(sizes[:, slot_index], count=3)}"
        )


@torch.no_grad()
def _predict_prompt(
    *,
    prompt: str,
    tokenizer: object,
    text_encoder: object,
    graph_encoder: torch.nn.Module,
    device: str,
    layout_sample_mode: str,
    layout_z_scale: float,
) -> tuple[dict[str, Any], list[str], torch.Tensor, torch.Tensor, torch.Tensor, object]:
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
    return (
        scene_graph,
        labels,
        conditioning.slot_positions[:, :node_count],
        conditioning.slot_log_sizes_3d[:, :node_count],
        conditioning.slot_mask[:, :node_count],
        conditioning,
    )


def main() -> int:
    args = make_parser().parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    prompts = _read_prompts(args)
    device = resolve_torch_device(args.device)

    state_dict = normalize_graph_encoder_state_dict(torch.load(args.graph_encoder_path, map_location="cpu"))
    _slot_dim, text_hidden_dim, _gnn_layers, _layout_mode, _latent_dim, _decoder_mode, _decoder_box_residual = infer_graph_encoder_config(state_dict)
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

    records: list[dict[str, Any]] = []
    current_dir = args.output_dir / "current_oscr"
    fixed_dir = args.output_dir / "top_left_front_oscr"
    comparison_dir = args.output_dir / "comparison"
    current_dir.mkdir(parents=True, exist_ok=True)
    fixed_dir.mkdir(parents=True, exist_ok=True)
    comparison_dir.mkdir(parents=True, exist_ok=True)
    for index, prompt in enumerate(prompts):
        for sample_index in range(max(1, args.num_layout_samples)):
            sample_seed = int(args.layout_seed) + index * 1000 + sample_index
            if args.layout_sample_mode == "prior_sample":
                torch.manual_seed(sample_seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(sample_seed)
            scene_graph, labels, centers, log_sizes, slot_mask, conditioning = _predict_prompt(
                prompt=prompt,
                tokenizer=tokenizer,
                text_encoder=text_encoder,
                graph_encoder=graph_encoder,
                device=device,
                layout_sample_mode=args.layout_sample_mode,
                layout_z_scale=args.layout_z_scale,
            )
            if args.debug_latents:
                _print_latent_debug(
                    prompt_index=index,
                    sample_index=sample_index,
                    sample_seed=sample_seed,
                    conditioning=conditioning,
                    centers=centers,
                    log_sizes=log_sizes,
                    slot_mask=slot_mask,
                    labels=labels,
                )
            current_oscr = _tensor_oscr_to_pil(
                render_oscr_boxes(
                    centers=centers.detach().cpu(),
                    log_sizes=log_sizes.detach().cpu(),
                    slot_mask=slot_mask.detach().cpu(),
                    image_size=args.image_size,
                )[0]
            )
            fixed_corner = render_fixed_corner_oscr(
                centers=centers,
                log_sizes=log_sizes,
                slot_mask=slot_mask,
                labels=labels,
                image_size=args.image_size,
                depth_offset_scale=args.depth_offset_scale,
                front_alpha=args.front_alpha,
                side_alpha=args.side_alpha,
                back_alpha=args.back_alpha,
                edge_alpha=args.edge_alpha,
            )
            stem = f"{_safe_name(prompt, index)}_sample{sample_index:02d}"
            current_path = current_dir / f"{stem}.png"
            fixed_path = fixed_dir / f"{stem}.png"
            sheet_path = comparison_dir / f"{stem}.png"
            current_oscr.save(current_path)
            fixed_corner.save(fixed_path)
            _make_contact_sheet(
                prompt=f"{prompt} | {args.layout_sample_mode} seed={sample_seed}",
                current=current_oscr,
                fixed_corner=fixed_corner,
                output_path=sheet_path,
            )
            record = {
                "prompt": prompt,
                "sample_index": sample_index,
                "layout_sample_mode": args.layout_sample_mode,
                "layout_z_scale": args.layout_z_scale,
                "layout_seed": sample_seed,
                "scene_graph": scene_graph,
                "labels": labels,
                "predicted_centers": centers[0].detach().cpu().to(torch.float32).tolist(),
                "predicted_sizes": log_sizes[0].detach().cpu().to(torch.float32).exp().tolist(),
                "latent_debug": _latent_debug_payload(conditioning, len(labels)),
                "current_oscr": str(current_path),
                "top_left_front_oscr": str(fixed_path),
                "comparison": str(sheet_path),
            }
            records.append(record)
            print(f"Saved demo for prompt {index} sample {sample_index}: {sheet_path}")

    (args.output_dir / "top_left_front_oscr_records.json").write_text(json.dumps(records, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
