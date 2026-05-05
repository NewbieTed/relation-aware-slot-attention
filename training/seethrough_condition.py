"""SeeThrough3D-aligned OSCR rendering, cuboid masks, and token binding.

The functions here keep our project's main difference intact: object cuboids are
predicted by the GNN. Downstream of those boxes, the condition format follows
SeeThrough3D more closely: a rendered cuboid condition image, per-object cuboid
masks from the same projection, and object-token spans from an explicit subject
list prompt.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw


PAPER_RED = (128, 0, 0)
PAPER_BLUE = (0, 0, 128)
PAPER_GREEN = (0, 128, 0)


@dataclass(frozen=True)
class BindingPrompt:
    """Prompt text plus character spans for object labels in the subject list."""

    prompt: str
    subject_spans: list[tuple[int, int]]


def build_binding_prompt(
    *,
    original_prompt: str,
    scene_graph: dict[str, Any],
    prefix: str = "a photo of",
) -> BindingPrompt:
    """Build a SeeThrough-style prompt with explicit object phrase anchors.

    SeeThrough3D stores a ``PLACEHOLDER`` prompt and later replaces it with a
    deterministic subject list. We mirror the useful part directly:
    ``a photo of dog and bicycle, a dog behind a bicycle``.
    """

    labels = [str(node["label"]).replace("_", " ") for node in scene_graph["nodes"]]
    prompt = f"{prefix.strip()} "
    spans: list[tuple[int, int]] = []
    for index, label in enumerate(labels):
        if index > 0:
            prompt += " and "
        start = len(prompt)
        prompt += label
        spans.append((start, len(prompt)))
    prompt += f", {original_prompt.strip()}"
    return BindingPrompt(prompt=prompt, subject_spans=spans)


def call_ids_from_binding_prompt(
    *,
    tokenizer: Any,
    binding_prompt: BindingPrompt,
    max_sequence_length: int,
    device: str,
) -> list[torch.Tensor]:
    """Return T5 token positions corresponding to each anchored subject span."""

    try:
        encoded = tokenizer(
            binding_prompt.prompt,
            padding="max_length",
            max_length=max_sequence_length,
            truncation=True,
            return_offsets_mapping=True,
            return_tensors="pt",
        )
    except (NotImplementedError, TypeError):
        encoded = tokenizer(
            binding_prompt.prompt,
            padding="max_length",
            max_length=max_sequence_length,
            truncation=True,
            return_tensors="pt",
        )

    if hasattr(encoded, "offset_mapping"):
        offsets = encoded.offset_mapping[0].tolist()
        call_ids: list[torch.Tensor] = []
        for start, end in binding_prompt.subject_spans:
            ids = [
                index
                for index, (token_start, token_end) in enumerate(offsets)
                if token_end > start and token_start < end
            ]
            call_ids.append(torch.tensor(ids, device=device, dtype=torch.long))
        return call_ids

    # Fallback for non-fast tokenizers. It is less exact, but the subject list is
    # deterministic, so sequential matching is still safer than whole-prompt
    # label search.
    prompt_ids = encoded.input_ids[0].tolist()
    cursor = 0
    call_ids = []
    for start, end in binding_prompt.subject_spans:
        label = binding_prompt.prompt[start:end]
        label_ids = tokenizer(label, add_special_tokens=False, return_tensors="pt").input_ids[0].tolist()
        found: list[int] = []
        for index in range(cursor, max(0, len(prompt_ids) - len(label_ids) + 1)):
            if prompt_ids[index : index + len(label_ids)] == label_ids:
                found = list(range(index, index + len(label_ids)))
                cursor = index + len(label_ids)
                break
        call_ids.append(torch.tensor(found, device=device, dtype=torch.long))
    return call_ids


def _world_box(center_xyz: list[float], size_xyz: list[float]) -> tuple[list[float], list[float]]:
    x, y, z = center_xyz
    sx, sy, sz = size_xyz
    world_scale = 3.5
    depth_scale = 3.0
    dims = [
        max(0.12, sx * world_scale),
        max(0.12, sz * depth_scale),
        max(0.12, sy * world_scale),
    ]
    center = [
        x * world_scale,
        -z * depth_scale,
        -y * world_scale + dims[2] * 0.5,
    ]
    return center, dims


def _normalize(values: list[float]) -> list[float]:
    norm = sum(value * value for value in values) ** 0.5
    if norm <= 1e-8:
        return values
    return [value / norm for value in values]


def _cross(a: list[float], b: list[float]) -> list[float]:
    return [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]


def _dot(a: list[float], b: list[float]) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _camera_basis() -> tuple[list[float], list[float], list[float], list[float]]:
    camera = [-4.2, -7.0, 4.3]
    target = [0.0, 0.0, 0.7]
    forward = _normalize([target[i] - camera[i] for i in range(3)])
    right = _normalize(_cross(forward, [0.0, 0.0, 1.0]))
    up = _normalize(_cross(right, forward))
    return camera, right, up, forward


def _corners(center: list[float], dims: list[float], *, azimuth_degrees: float = 0.0) -> list[list[float]]:
    hx, hy, hz = dims[0] * 0.5, dims[1] * 0.5, dims[2] * 0.5
    angle = torch.deg2rad(torch.tensor(float(azimuth_degrees))).item()
    cos_a = torch.cos(torch.tensor(angle)).item()
    sin_a = torch.sin(torch.tensor(angle)).item()
    local_offsets = [
        [sx * hx, sy * hy, sz * hz]
        for sx in (-1.0, 1.0)
        for sy in (-1.0, 1.0)
        for sz in (-1.0, 1.0)
    ]
    return [
        [
            center[0] + offset[0] * cos_a - offset[1] * sin_a,
            center[1] + offset[0] * sin_a + offset[1] * cos_a,
            center[2] + offset[2],
        ]
        for offset in local_offsets
    ]


def _project(points: list[list[float]], image_size: int) -> tuple[list[tuple[float, float]], list[float]]:
    camera, right, up, forward = _camera_basis()
    orthographic_scale = 7.0
    projected: list[tuple[float, float]] = []
    depths: list[float] = []
    for point in points:
        rel = [point[i] - camera[i] for i in range(3)]
        px = (_dot(rel, right) / orthographic_scale + 0.5) * image_size
        py = (0.5 - _dot(rel, up) / orthographic_scale) * image_size
        projected.append((px, py))
        depths.append(_dot(rel, forward))
    return projected, depths


FACES: tuple[tuple[int, int, int, int, tuple[int, int, int]], ...] = (
    # SeeThrough3D colors Blender cube faces by material index, not by
    # camera-facing geometry. Its reordered material list is:
    # [blue, red, green, green, green, green].
    (0, 1, 3, 2, PAPER_BLUE),
    (4, 6, 7, 5, PAPER_RED),
    (0, 4, 5, 1, PAPER_GREEN),
    (2, 3, 7, 6, PAPER_GREEN),
    (0, 2, 6, 4, PAPER_GREEN),
    (1, 5, 7, 3, PAPER_GREEN),
)


def _max_pool_masks(mask_tensor: torch.Tensor, output_hw: tuple[int, int]) -> torch.Tensor:
    out_h, out_w = output_hw
    _, height, width = mask_tensor.shape
    if height == out_h and width == out_w:
        return mask_tensor
    kernel_h = max(1, height // out_h)
    kernel_w = max(1, width // out_w)
    if height % out_h == 0 and width % out_w == 0:
        return F.max_pool2d(
            mask_tensor[:, None].float(),
            kernel_size=(kernel_h, kernel_w),
            stride=(kernel_h, kernel_w),
        ).squeeze(1).to(torch.uint8)
    resized = F.interpolate(mask_tensor[:, None].float(), size=output_hw, mode="nearest")
    return resized.squeeze(1).to(torch.uint8)


def render_seethrough_oscr_and_masks(
    *,
    centers: torch.Tensor,
    log_sizes: torch.Tensor,
    slot_mask: torch.Tensor,
    image_size: int,
    mask_size: tuple[int, int] | None = None,
    face_alpha: float = 0.10,
    azimuth_degrees: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Render paper-style cuboid OSCR images and per-object masks.

    Returns:
        ``oscr``: ``[B, 3, image_size, image_size]`` in ``[-1, 1]``.
        ``masks``: ``[B, S, H, W]`` binary masks, downsampled with max pooling
        when ``mask_size`` is provided.
    """

    centers_cpu = centers.detach().cpu().float()
    sizes_cpu = log_sizes.detach().cpu().float().exp().clamp(min=0.03, max=2.0)
    mask_cpu = slot_mask.detach().cpu()
    batch_size, slot_count, _ = centers_cpu.shape
    oscr_images: list[torch.Tensor] = []
    all_masks: list[torch.Tensor] = []
    alpha = int(max(0, min(255, round(face_alpha * 255))))

    for batch_index in range(batch_size):
        image = Image.new("RGBA", (image_size, image_size), (255, 255, 255, 255))
        face_layer = Image.new("RGBA", image.size, (0, 0, 0, 0))
        face_draw = ImageDraw.Draw(face_layer, "RGBA")
        subject_masks: list[Image.Image] = []
        face_items: list[tuple[float, int, tuple[tuple[float, float], ...], tuple[int, int, int]]] = []

        for slot_index in range(slot_count):
            subject_mask = Image.new("L", image.size, 0)
            subject_masks.append(subject_mask)
            if not bool(mask_cpu[batch_index, slot_index].item()):
                continue
            center, dims = _world_box(
                centers_cpu[batch_index, slot_index].tolist(),
                sizes_cpu[batch_index, slot_index].tolist(),
            )
            projected, depths = _project(_corners(center, dims, azimuth_degrees=azimuth_degrees), image_size)
            mask_draw = ImageDraw.Draw(subject_mask)
            for face in FACES:
                indices = face[:4]
                color = face[4]
                points = tuple(projected[index] for index in indices)
                mean_depth = sum(depths[index] for index in indices) / len(indices)
                face_items.append((mean_depth, slot_index, points, color))
                mask_draw.polygon(points, fill=255)

        for _depth, _slot_index, points, color in sorted(face_items, key=lambda item: item[0], reverse=True):
            face_draw.polygon(points, fill=(*color, alpha))
        image = Image.alpha_composite(image, face_layer).convert("RGB")
        oscr = torch.from_numpy(__import__("numpy").array(image)).permute(2, 0, 1).float()
        oscr_images.append(oscr.div(127.5).sub(1.0))
        stacked_masks = torch.stack(
            [torch.from_numpy(__import__("numpy").array(subject_mask)).gt(0).to(torch.uint8) for subject_mask in subject_masks],
            dim=0,
        )
        if mask_size is not None:
            stacked_masks = _max_pool_masks(stacked_masks, mask_size)
        all_masks.append(stacked_masks)

    return torch.stack(oscr_images, dim=0).to(centers.device), torch.stack(all_masks, dim=0).to(centers.device)


def _labels_from_scene_graph(scene_graph: dict[str, Any]) -> list[str]:
    return [str(node["label"]).replace("_", " ") for node in scene_graph["nodes"]]


def _blender_cache_key(
    *,
    labels: list[str],
    centers: list[list[float]],
    sizes: list[list[float]],
    image_size: int,
    face_alpha: float,
    azimuth_degrees: float,
) -> str:
    payload = {
        "labels": labels,
        "centers": [[round(float(value), 5) for value in row] for row in centers],
        "sizes": [[round(float(value), 5) for value in row] for row in sizes],
        "image_size": image_size,
        "face_alpha": round(float(face_alpha), 6),
        "azimuth_degrees": round(float(azimuth_degrees), 4),
        "renderer_version": "blender_condition_v1_paper_face_order",
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:24]


def _pil_to_condition_tensor(path: Path) -> torch.Tensor:
    image = Image.open(path).convert("RGB")
    array = torch.from_numpy(__import__("numpy").array(image)).permute(2, 0, 1).float()
    return array.div(127.5).sub(1.0)


def render_blender_oscr_conditions(
    *,
    centers: torch.Tensor,
    log_sizes: torch.Tensor,
    slot_mask: torch.Tensor,
    scene_graphs: list[dict[str, Any]],
    prompts: list[str],
    image_size: int,
    face_alpha: float,
    azimuth_degrees: float,
    blender_bin: str,
    cache_dir: Path,
) -> torch.Tensor:
    """Render SeeThrough3D-style Blender OSCR images and return tensors in ``[-1, 1]``.

    The model-condition path uses this when ``condition_renderer=blender``.
    Rendered PNGs are cached by predicted boxes, labels, alpha, azimuth, and
    image size because the frozen GNN makes these deterministic.
    """

    centers_cpu = centers.detach().cpu().float()
    sizes_cpu = log_sizes.detach().cpu().float().exp().clamp(min=0.03, max=2.0)
    mask_cpu = slot_mask.detach().cpu()
    cache_dir.mkdir(parents=True, exist_ok=True)
    image_paths: list[Path] = []
    records_to_render: list[dict[str, Any]] = []
    cache_targets: list[Path] = []

    for batch_index, scene_graph in enumerate(scene_graphs):
        labels = _labels_from_scene_graph(scene_graph)
        valid_centers: list[list[float]] = []
        valid_sizes: list[list[float]] = []
        valid_labels: list[str] = []
        for slot_index, label in enumerate(labels):
            if slot_index >= mask_cpu.shape[1] or not bool(mask_cpu[batch_index, slot_index].item()):
                continue
            valid_labels.append(label)
            valid_centers.append(centers_cpu[batch_index, slot_index].tolist())
            valid_sizes.append(sizes_cpu[batch_index, slot_index].tolist())
        cache_key = _blender_cache_key(
            labels=valid_labels,
            centers=valid_centers,
            sizes=valid_sizes,
            image_size=image_size,
            face_alpha=face_alpha,
            azimuth_degrees=azimuth_degrees,
        )
        target_path = cache_dir / f"{cache_key}.png"
        image_paths.append(target_path)
        if target_path.exists():
            continue
        cache_targets.append(target_path)
        records_to_render.append(
            {
                "prompt": f"{batch_index}_{cache_key}_{prompts[batch_index]}",
                "scene_graph": scene_graph,
                "labels": valid_labels,
                "predicted_centers": valid_centers,
                "predicted_sizes": valid_sizes,
            }
        )

    if records_to_render:
        render_root = cache_dir / f"render_{hashlib.sha256(str(cache_targets).encode('utf-8')).hexdigest()[:12]}"
        render_root.mkdir(parents=True, exist_ok=True)
        records_path = render_root / "records.json"
        records_path.write_text(json.dumps(records_to_render, indent=2))
        script_path = Path(__file__).resolve().parents[1] / "evaluation" / "render_blender_oscr_demo.py"
        cmd = [
            blender_bin,
            "--background",
            "--python",
            str(script_path),
            "--",
            "--records-json",
            str(records_path),
            "--output-dir",
            str(render_root),
            "--image-size",
            str(image_size),
            "--face-alpha",
            str(face_alpha),
            "--azimuth-degrees",
            str(azimuth_degrees),
            "--background",
            "white",
            "--engine",
            "eevee",
            "--samples",
            "64",
            "--no-edges",
            "--no-labels",
            "--no-ground",
            "--no-shadows",
        ]
        try:
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        except FileNotFoundError as exc:
            raise RuntimeError(f"Blender executable not found for condition rendering: {blender_bin}") from exc
        except subprocess.CalledProcessError as exc:
            stderr_tail = "\n".join((exc.stderr or "").splitlines()[-20:])
            raise RuntimeError(f"Blender condition rendering failed with exit code {exc.returncode}:\n{stderr_tail}") from exc
        manifest_path = render_root / "blender_oscr_manifest.json"
        manifest = json.loads(manifest_path.read_text())
        if len(manifest) != len(cache_targets):
            raise RuntimeError(
                f"Blender rendered {len(manifest)} OSCR images, expected {len(cache_targets)}."
            )
        for item, target_path in zip(manifest, cache_targets):
            shutil.copyfile(item["output"], target_path)
        if result.stdout:
            print(result.stdout.strip().splitlines()[-1])

    images = [_pil_to_condition_tensor(path) for path in image_paths]
    return torch.stack(images, dim=0).to(centers.device)
