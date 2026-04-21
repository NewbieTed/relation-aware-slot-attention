from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from scop_depth.geometry import segmentation_to_mask
from scop_depth.models import CocoInstanceAnnotation


def _square_crop_bounds(image_size: tuple[int, int]) -> tuple[int, int, int]:
    width, height = image_size
    crop_size = min(width, height)
    left = (width - crop_size) // 2
    top = (height - crop_size) // 2
    return left, top, crop_size


def _bbox_to_mask(
    bbox: tuple[float, float, float, float],
    image_size: tuple[int, int],
) -> np.ndarray:
    width, height = image_size
    mask = np.zeros((height, width), dtype=np.float32)
    x, y, w, h = bbox
    x0 = max(0, int(np.floor(x)))
    y0 = max(0, int(np.floor(y)))
    x1 = min(width, int(np.ceil(x + w)))
    y1 = min(height, int(np.ceil(y + h)))
    if x1 > x0 and y1 > y0:
        mask[y0:y1, x0:x1] = 1.0
    return mask


def _annotation_mask(
    annot_dict: dict[str, Any],
    image_size: tuple[int, int],
) -> np.ndarray:
    annot = CocoInstanceAnnotation.from_dict(annot_dict)
    mask = segmentation_to_mask(annot, image_size[0], image_size[1])
    if mask is None:
        mask = _bbox_to_mask(annot.bbox, image_size)
    else:
        mask = mask.astype(np.float32)
    return mask


def _crop_and_resize_mask(
    mask: np.ndarray,
    image_size: tuple[int, int],
    resolution: tuple[int, int],
    *,
    device: torch.device,
) -> torch.Tensor:
    left, top, crop_size = _square_crop_bounds(image_size)
    cropped = mask[top : top + crop_size, left : left + crop_size]
    tensor = torch.from_numpy(cropped).to(device=device, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    resized = F.interpolate(tensor, size=resolution, mode="nearest")
    return (resized > 0).to(dtype=torch.float32).squeeze(0).squeeze(0)


def build_slot_target_masks(
    *,
    metadata: list[dict[str, Any]],
    image_sizes: list[tuple[int, int]],
    slot_mask: torch.Tensor,
    resolution: tuple[int, int],
    device: torch.device,
) -> torch.Tensor:
    batch_size, max_nodes = slot_mask.shape
    masks = torch.zeros(batch_size, max_nodes, resolution[0], resolution[1], device=device, dtype=torch.float32)
    for batch_index, (row, image_size) in enumerate(zip(metadata, image_sizes)):
        annots = row.get("annots", [])
        for slot_index in range(min(len(annots), max_nodes)):
            if not bool(slot_mask[batch_index, slot_index].item()):
                continue
            mask = _annotation_mask(annots[slot_index], image_size)
            masks[batch_index, slot_index] = _crop_and_resize_mask(
                mask,
                image_size,
                resolution,
                device=device,
            )
    return masks


def clear_attention_cache(processors: dict[str, torch.nn.Module]) -> None:
    for processor in processors.values():
        clear_fn = getattr(processor, "clear_attention_cache", None)
        if clear_fn is not None:
            clear_fn()


def collect_slot_attention_maps(
    processors: dict[str, torch.nn.Module],
) -> dict[tuple[int, int], list[torch.Tensor]]:
    collected: dict[tuple[int, int], list[torch.Tensor]] = defaultdict(list)
    for processor in processors.values():
        attn_map = getattr(processor, "latest_slot_attention_map", None)
        query_hw = getattr(processor, "latest_query_hw", None)
        if attn_map is None or query_hw is None:
            continue
        height, width = query_hw
        reshaped = attn_map.reshape(attn_map.shape[0], height, width, attn_map.shape[-1])
        collected[(height, width)].append(reshaped)
    return dict(collected)


def compute_slot_attention_losses(
    *,
    attention_maps: dict[tuple[int, int], list[torch.Tensor]],
    metadata: list[dict[str, Any]],
    image_sizes: list[tuple[int, int]],
    slot_mask: torch.Tensor,
    device: torch.device,
    return_debug: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any] | None]:
    if not attention_maps:
        zero = slot_mask.new_tensor(0.0, dtype=torch.float32)
        debug = {
            "num_attention_maps": 0,
            "resolutions": [],
            "mask_sum_min": 0.0,
            "mask_sum_max": 0.0,
            "mask_sum_mean": 0.0,
            "attn_sum_min": 0.0,
            "attn_sum_max": 0.0,
            "attn_sum_mean": 0.0,
            "valid_slots": int(slot_mask.sum().item()),
        } if return_debug else None
        return zero, zero, debug

    token_losses: list[torch.Tensor] = []
    pixel_losses: list[torch.Tensor] = []
    debug_mask_sums: list[torch.Tensor] = []
    debug_attn_sums: list[torch.Tensor] = []
    eps = 1e-6

    for resolution, maps_at_resolution in attention_maps.items():
        target_masks = build_slot_target_masks(
            metadata=metadata,
            image_sizes=image_sizes,
            slot_mask=slot_mask,
            resolution=resolution,
            device=device,
        )
        valid_mask = slot_mask.to(device=device, dtype=torch.bool)
        if not valid_mask.any():
            continue
        debug_mask_sums.append(target_masks[valid_mask].flatten(1).sum(dim=-1))

        for attn_map in maps_at_resolution:
            attn_map = attn_map.to(dtype=torch.float32)
            attn_by_slot = attn_map.permute(0, 3, 1, 2)
            debug_attn_sums.append(attn_by_slot[valid_mask].flatten(1).sum(dim=-1))
            numerator = (attn_by_slot * target_masks).flatten(2).sum(dim=-1)
            denominator = attn_by_slot.flatten(2).sum(dim=-1).clamp_min(eps)
            inside_mass = numerator / denominator
            token_losses.append((1.0 - inside_mass[valid_mask]).pow(2).mean())

        mean_attn = torch.stack([m.to(dtype=torch.float32) for m in maps_at_resolution], dim=0).mean(dim=0)
        mean_attn = mean_attn.permute(0, 3, 1, 2).clamp(min=eps, max=1.0 - eps)
        pixel_input = mean_attn[valid_mask]
        pixel_target = target_masks[valid_mask]
        pixel_losses.append(
            -(
                pixel_target * torch.log(pixel_input)
                + (1.0 - pixel_target) * torch.log(1.0 - pixel_input)
            ).mean()
        )

    token_loss = torch.stack(token_losses).mean() if token_losses else slot_mask.new_tensor(0.0, dtype=torch.float32)
    pixel_loss = torch.stack(pixel_losses).mean() if pixel_losses else slot_mask.new_tensor(0.0, dtype=torch.float32)

    debug: dict[str, Any] | None = None
    if return_debug:
        if debug_mask_sums:
            mask_sums = torch.cat(debug_mask_sums, dim=0)
        else:
            mask_sums = slot_mask.new_zeros((0,), dtype=torch.float32)
        if debug_attn_sums:
            attn_sums = torch.cat(debug_attn_sums, dim=0)
        else:
            attn_sums = slot_mask.new_zeros((0,), dtype=torch.float32)
        debug = {
            "num_attention_maps": sum(len(maps) for maps in attention_maps.values()),
            "resolutions": [f"{height}x{width}" for height, width in sorted(attention_maps.keys())],
            "mask_sum_min": float(mask_sums.min().item()) if mask_sums.numel() else 0.0,
            "mask_sum_max": float(mask_sums.max().item()) if mask_sums.numel() else 0.0,
            "mask_sum_mean": float(mask_sums.mean().item()) if mask_sums.numel() else 0.0,
            "attn_sum_min": float(attn_sums.min().item()) if attn_sums.numel() else 0.0,
            "attn_sum_max": float(attn_sums.max().item()) if attn_sums.numel() else 0.0,
            "attn_sum_mean": float(attn_sums.mean().item()) if attn_sums.numel() else 0.0,
            "valid_slots": int(slot_mask.sum().item()),
        }
    return token_loss, pixel_loss, debug
