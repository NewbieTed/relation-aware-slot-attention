"""Render projected 3D OSCR demos from saved GNN layout JSON records.

This is a local, dependency-light renderer for OSCR design iteration. It does
not use Blender and does not run the GNN. It reads predicted centers/sizes from
``top_left_front_oscr_records.json`` and draws true projected 3D cuboids:
transparent faces first, then all 12 edges on top so back/through-lines remain
visible in the diagnostic image.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont


COLORS = [
    (0, 213, 255),
    (255, 51, 102),
    (42, 157, 143),
    (247, 127, 0),
    (255, 209, 102),
    (155, 93, 229),
]


FACES = [
    (0, 1, 3, 2),  # -x
    (4, 6, 7, 5),  # +x
    (0, 4, 5, 1),  # -y / front
    (2, 3, 7, 6),  # +y / back
    (0, 2, 6, 4),  # -z / bottom
    (1, 5, 7, 3),  # +z / top
]


EDGES = [
    (0, 1),
    (0, 2),
    (0, 4),
    (1, 3),
    (1, 5),
    (2, 3),
    (2, 6),
    (3, 7),
    (4, 5),
    (4, 6),
    (5, 7),
    (6, 7),
]


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Render projected cuboid OSCR demos from saved GNN JSON records.")
    parser.add_argument("--records-json", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--image-size", type=int, default=768)
    parser.add_argument("--world-scale", type=float, default=3.5)
    parser.add_argument("--depth-scale", type=float, default=3.0)
    parser.add_argument("--face-alpha", type=int, default=10)
    parser.add_argument("--edge-alpha", type=int, default=235)
    parser.add_argument("--edge-width", type=int, default=3)
    parser.add_argument("--orthographic-scale", type=float, default=7.0)
    parser.add_argument("--camera-x", type=float, default=4.2)
    parser.add_argument("--camera-y", type=float, default=-7.0)
    parser.add_argument("--camera-z", type=float, default=4.3)
    parser.add_argument("--target-z", type=float, default=0.7)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--no-labels", action="store_true")
    return parser


def _safe_name(prompt: str, index: int) -> str:
    safe = "".join(ch.lower() if ch.isalnum() else "_" for ch in prompt).strip("_")
    while "__" in safe:
        safe = safe.replace("__", "_")
    return f"{index:03d}_{safe[:80] or 'prompt'}"


def _load_font(size: int) -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size)
    except OSError:
        return ImageFont.load_default()


def _normalize(v: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(v))
    if norm <= 1e-8:
        return v
    return v / norm


def _camera_basis(args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    camera = np.array([args.camera_x, args.camera_y, args.camera_z], dtype=np.float32)
    target = np.array([0.0, 0.0, args.target_z], dtype=np.float32)
    forward = _normalize(target - camera)
    right = _normalize(np.cross(forward, np.array([0.0, 0.0, 1.0], dtype=np.float32)))
    up = _normalize(np.cross(right, forward))
    return camera, right, up, forward


def _to_world(
    center_xyz: list[float],
    size_xyz: list[float],
    *,
    world_scale: float,
    depth_scale: float,
) -> tuple[np.ndarray, np.ndarray]:
    x, y, z = center_xyz
    sx, sy, sz = size_xyz
    dims = np.array(
        [
            max(0.12, sx * world_scale),
            max(0.12, sz * depth_scale),
            max(0.12, sy * world_scale),
        ],
        dtype=np.float32,
    )
    center = np.array(
        [
            x * world_scale,
            -z * depth_scale,
            -y * world_scale + dims[2] * 0.5,
        ],
        dtype=np.float32,
    )
    return center, dims


def _corners(center: np.ndarray, dims: np.ndarray) -> np.ndarray:
    hx, hy, hz = dims * 0.5
    return np.array(
        [
            [center[0] - hx, center[1] - hy, center[2] - hz],
            [center[0] - hx, center[1] - hy, center[2] + hz],
            [center[0] - hx, center[1] + hy, center[2] - hz],
            [center[0] - hx, center[1] + hy, center[2] + hz],
            [center[0] + hx, center[1] - hy, center[2] - hz],
            [center[0] + hx, center[1] - hy, center[2] + hz],
            [center[0] + hx, center[1] + hy, center[2] - hz],
            [center[0] + hx, center[1] + hy, center[2] + hz],
        ],
        dtype=np.float32,
    )


def _project(
    points: np.ndarray,
    *,
    camera: np.ndarray,
    right: np.ndarray,
    up: np.ndarray,
    forward: np.ndarray,
    image_size: int,
    orthographic_scale: float,
) -> tuple[list[tuple[float, float]], np.ndarray]:
    rel = points - camera[None, :]
    x = rel @ right
    y = rel @ up
    depth = rel @ forward
    px = (x / orthographic_scale + 0.5) * image_size
    py = (0.5 - y / orthographic_scale) * image_size
    return list(zip(px.tolist(), py.tolist())), depth


def _text_box(draw: ImageDraw.ImageDraw, xy: tuple[int, int], lines: list[str], font: ImageFont.ImageFont) -> None:
    padding = 6
    boxes = [draw.textbbox((0, 0), line, font=font) for line in lines]
    width = max(right - left for left, top, right, bottom in boxes) + padding * 2
    height = sum(bottom - top for left, top, right, bottom in boxes) + padding * 2 + 3 * (len(lines) - 1)
    x, y = xy
    draw.rectangle([x, y, x + width, y + height], fill=(0, 0, 0, 190))
    cursor = y + padding
    for line, box in zip(lines, boxes):
        draw.text((x + padding, cursor), line, fill=(255, 255, 255, 255), font=font)
        cursor += box[3] - box[1] + 3


def _render_record(record: dict[str, Any], *, args: argparse.Namespace, index: int) -> dict[str, Any]:
    image = Image.new("RGB", (args.image_size, args.image_size), (8, 10, 13))
    face_layer = Image.new("RGBA", image.size, (0, 0, 0, 0))
    edge_layer = Image.new("RGBA", image.size, (0, 0, 0, 0))
    face_draw = ImageDraw.Draw(face_layer, "RGBA")
    edge_draw = ImageDraw.Draw(edge_layer, "RGBA")
    label_font = _load_font(18)
    title_font = _load_font(22)

    camera, right, up, forward = _camera_basis(args)
    cuboids = []
    for slot_index, (label, center_xyz, size_xyz) in enumerate(
        zip(record["labels"], record["predicted_centers"], record["predicted_sizes"])
    ):
        center, dims = _to_world(
            center_xyz,
            size_xyz,
            world_scale=args.world_scale,
            depth_scale=args.depth_scale,
        )
        corners = _corners(center, dims)
        projected, depths = _project(
            corners,
            camera=camera,
            right=right,
            up=up,
            forward=forward,
            image_size=args.image_size,
            orthographic_scale=args.orthographic_scale,
        )
        cuboids.append(
            {
                "slot_index": slot_index,
                "label": label,
                "center": center,
                "dims": dims,
                "projected": projected,
                "depths": depths,
                "color": COLORS[slot_index % len(COLORS)],
            }
        )

    # Faces first, sorted from far to near. Then all edges are drawn after all
    # faces, which is the key thing the earlier PIL demo failed to guarantee.
    face_items = []
    for cuboid in cuboids:
        for face_index, face in enumerate(FACES):
            mean_depth = float(np.mean(cuboid["depths"][list(face)]))
            face_items.append((mean_depth, cuboid, face_index, face))
    face_items.sort(key=lambda item: item[0])
    for _, cuboid, face_index, face in face_items:
        points = [cuboid["projected"][idx] for idx in face]
        color = cuboid["color"]
        # Slightly brighter top face helps depth readability while preserving
        # very low OSCR-style opacity.
        alpha = args.face_alpha + (4 if face_index == 5 else 0)
        face_draw.polygon(points, fill=(*color, max(0, min(255, alpha))))

    for cuboid in cuboids:
        color = cuboid["color"]
        edge_color = (*color, max(0, min(255, args.edge_alpha)))
        for start, end in EDGES:
            edge_draw.line(
                [cuboid["projected"][start], cuboid["projected"][end]],
                fill=edge_color,
                width=args.edge_width,
            )
        center_2d, _ = _project(
            cuboid["center"][None, :],
            camera=camera,
            right=right,
            up=up,
            forward=forward,
            image_size=args.image_size,
            orthographic_scale=args.orthographic_scale,
        )
        cx, cy = center_2d[0]
        edge_draw.ellipse([cx - 5, cy - 5, cx + 5, cy + 5], fill=edge_color)
        if not args.no_labels:
            _text_box(
                edge_draw,
                (int(max(4, min(args.image_size - 210, cx + 10))), int(max(42, min(args.image_size - 90, cy + 10)))),
                [
                    cuboid["label"],
                    f"c=({record['predicted_centers'][cuboid['slot_index']][0]:+.2f},"
                    f"{record['predicted_centers'][cuboid['slot_index']][1]:+.2f},"
                    f"{record['predicted_centers'][cuboid['slot_index']][2]:+.2f})",
                    f"s=({record['predicted_sizes'][cuboid['slot_index']][0]:.2f},"
                    f"{record['predicted_sizes'][cuboid['slot_index']][1]:.2f},"
                    f"{record['predicted_sizes'][cuboid['slot_index']][2]:.2f})",
                ],
                label_font,
            )

    composed = Image.alpha_composite(image.convert("RGBA"), face_layer)
    composed = Image.alpha_composite(composed, edge_layer)
    draw = ImageDraw.Draw(composed)
    draw.text((14, 14), record["prompt"], fill=(255, 255, 255, 255), font=title_font)
    draw.text((14, 46), "Projected 3D cuboids: transparent faces, all 12 edges drawn last", fill=(210, 215, 222, 255), font=label_font)

    stem = _safe_name(record["prompt"], index)
    output_path = args.output_dir / f"{stem}_projected_oscr.png"
    composed.convert("RGB").save(output_path)
    return {
        "prompt": record["prompt"],
        "output": str(output_path),
    }


def main() -> int:
    args = make_parser().parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    records = json.loads(args.records_json.read_text())
    if args.limit is not None:
        records = records[: args.limit]

    manifest = []
    for index, record in enumerate(records):
        result = _render_record(record, args=args, index=index)
        manifest.append(result)
        print(f"Rendered {result['output']}")
    (args.output_dir / "projected_oscr_manifest.json").write_text(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
