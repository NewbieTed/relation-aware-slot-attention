"""SeeThrough3D-aligned OSCR rendering, cuboid masks, and token binding.

The functions here keep our project's main difference intact: object cuboids are
predicted by the GNN. Downstream of those boxes, the condition format follows
SeeThrough3D more closely: a rendered cuboid condition image, per-object cuboid
masks from the same projection, and object-token spans from an explicit subject
list prompt.
"""

from __future__ import annotations

from dataclasses import dataclass
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


def _corners(center: list[float], dims: list[float]) -> list[list[float]]:
    hx, hy, hz = dims[0] * 0.5, dims[1] * 0.5, dims[2] * 0.5
    return [
        [center[0] + sx * hx, center[1] + sy * hy, center[2] + sz * hz]
        for sx in (-1.0, 1.0)
        for sy in (-1.0, 1.0)
        for sz in (-1.0, 1.0)
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
            projected, depths = _project(_corners(center, dims), image_size)
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
