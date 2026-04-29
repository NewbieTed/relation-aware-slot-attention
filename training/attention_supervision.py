from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

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
    return _bbox_to_mask(tuple(annot_dict["bbox"]), image_size)


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


def compute_mean_slot_usage(attention_maps: dict[tuple[int, int], list[torch.Tensor]]) -> torch.Tensor:
    slot_masses: list[torch.Tensor] = []
    device: torch.device | None = None
    for maps_at_resolution in attention_maps.values():
        for attn_map in maps_at_resolution:
            device = attn_map.device
            slot_masses.append(attn_map.sum(dim=-1).reshape(-1))
    if not slot_masses:
        if device is None:
            device = torch.device("cpu")
        return torch.zeros((), device=device, dtype=torch.float32)
    return torch.cat(slot_masses, dim=0).to(dtype=torch.float32).mean()


def compute_region_slot_loss(
    *,
    attention_maps: dict[tuple[int, int], list[torch.Tensor]],
    slot_centers: torch.Tensor,
    slot_log_sigmas: torch.Tensor,
    slot_mask: torch.Tensor,
    device: torch.device,
    target_usage: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Bind object regions to their matching appended slots.

    Inside object i's target ellipse, slot i should dominate over the other
    slots. Outside that ellipse, slot i should not receive high attention. The
    target_usage argument is kept for CLI/backward compatibility and is not used
    by this contrastive objective.
    """
    del target_usage

    pos_losses: list[torch.Tensor] = []
    neg_in_losses: list[torch.Tensor] = []
    neg_out_losses: list[torch.Tensor] = []
    region_usages: list[torch.Tensor] = []
    valid_slot_mask = slot_mask.to(device=device, dtype=torch.bool)
    eps = 1e-6

    for resolution, maps_at_resolution in attention_maps.items():
        height, width = resolution
        ys = torch.linspace(-1.0, 1.0, height, device=device)
        xs = torch.linspace(-1.0, 1.0, width, device=device)
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
        grid = torch.stack([grid_x, grid_y], dim=-1).view(1, 1, height, width, 2)

        centers = slot_centers.to(device=device, dtype=torch.float32)[..., :2]
        sigmas = slot_log_sigmas.to(device=device, dtype=torch.float32).exp().clamp(min=0.03, max=2.0)
        normalized_delta = (grid - centers[:, :, None, None, :]) / sigmas[:, :, None, None, :]
        target_regions = torch.exp(-0.5 * normalized_delta.pow(2).sum(dim=-1))
        target_regions = target_regions * valid_slot_mask[:, :, None, None].to(dtype=torch.float32)

        mask_area = target_regions.flatten(2).sum(dim=-1)
        outside_regions = (1.0 - target_regions).clamp(min=0.0)
        outside_regions = outside_regions * valid_slot_mask[:, :, None, None].to(dtype=torch.float32)
        outside_area = outside_regions.flatten(2).sum(dim=-1)
        valid_mask = valid_slot_mask & (mask_area > 0)
        if not valid_mask.any():
            continue

        for attn_map in maps_at_resolution:
            attn_by_slot = attn_map.to(dtype=torch.float32).permute(0, 3, 1, 2)
            attn_by_slot = attn_by_slot * valid_slot_mask[:, :, None, None].to(dtype=torch.float32)
            slot_prob = attn_by_slot / attn_by_slot.sum(dim=1, keepdim=True).clamp_min(eps)

            correct_slot_prob = (slot_prob * target_regions).flatten(2).sum(dim=-1)
            correct_slot_prob = correct_slot_prob / mask_area.clamp_min(eps)
            pos_losses.append(-torch.log(correct_slot_prob[valid_mask].clamp_min(eps)).mean())
            region_usages.append(correct_slot_prob[valid_mask])

            total_slot_prob_inside = (slot_prob.sum(dim=1, keepdim=True) * target_regions).flatten(2).sum(dim=-1)
            total_slot_prob_inside = total_slot_prob_inside / mask_area.clamp_min(eps)
            other_slot_prob = (total_slot_prob_inside - correct_slot_prob).clamp_min(0.0)
            neg_in_losses.append(other_slot_prob[valid_mask].mean())

            attention_outside_region = (attn_by_slot * outside_regions).flatten(2).sum(dim=-1)
            outside_usage = attention_outside_region / outside_area.clamp_min(eps)
            valid_outside_mask = valid_mask & (outside_area > 0)
            if valid_outside_mask.any():
                neg_out_losses.append(outside_usage[valid_outside_mask].mean())

    if not pos_losses:
        zero = slot_mask.new_tensor(0.0, dtype=torch.float32)
        return zero, zero, zero, zero, zero

    pos_loss = torch.stack(pos_losses).mean()
    neg_in_loss = torch.stack(neg_in_losses).mean() if neg_in_losses else pos_loss.new_tensor(0.0)
    neg_out_loss = torch.stack(neg_out_losses).mean() if neg_out_losses else pos_loss.new_tensor(0.0)
    loss = pos_loss + 0.5 * neg_in_loss + 0.1 * neg_out_loss
    usage = torch.cat(region_usages, dim=0).mean() * 100.0
    return loss, usage, pos_loss, neg_in_loss, neg_out_loss
