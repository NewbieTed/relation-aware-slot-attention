"""Utilities for turning GNN layout predictions into ControlNet inputs.

The relation-aware GNN predicts one center and one axis-aligned uncertainty
ellipse per object slot. ControlNet, however, expects an image-like condition.
This module bridges that gap by rasterizing each slot into a Gaussian heatmap
in normalized image coordinates.
"""

from __future__ import annotations

import torch


def build_gaussian_layout_maps(
    *,
    slot_centers: torch.Tensor,
    slot_log_sigmas: torch.Tensor,
    slot_mask: torch.Tensor,
    image_size: int,
    channels: int = 3,
    sigma_scale: float = 1.0,
) -> torch.Tensor:
    """Build image-shaped Gaussian layout maps from slot center/sigma predictions.

    Args:
        slot_centers: Tensor with shape ``[batch, slots, 3]``. The first two
            coordinates are normalized x/y centers in the ``[-1, 1]`` image
            coordinate system; z is ignored for this 2D layout map.
        slot_log_sigmas: Tensor with shape ``[batch, slots, 2]``. Values are
            log standard deviations for the x/y Gaussian axes.
        slot_mask: Boolean tensor with shape ``[batch, slots]`` marking real
            object slots. Masked slots contribute zero heatmap mass.
        image_size: Square side length of the rasterized condition map.
        channels: Number of output channels. SD1.5 ControlNet commonly uses
            three-channel conditions, so the default is 3.
        sigma_scale: Optional multiplier for widening/narrowing the predicted
            Gaussian ellipses at train or inference time.

    Returns:
        Tensor with shape ``[batch, channels, image_size, image_size]`` and
        values clamped to ``[0, 1]``.

    Channel 0 receives slot 0, channel 1 receives slot 1, and the final channel
    receives the union of all valid slots. This keeps the condition compatible
    with SD ControlNet's default 3-channel conditioning input.
    """

    if channels < 3:
        raise ValueError("Gaussian layout conditioning expects at least 3 channels")

    # Build the coordinate grid in the same normalized image space used by the
    # GNN targets: x=-1 is left, x=+1 is right, y=-1 is top, y=+1 is bottom.
    device = slot_centers.device
    dtype = torch.float32
    batch_size, max_slots, _ = slot_centers.shape
    ys = torch.linspace(-1.0, 1.0, image_size, device=device, dtype=dtype)
    xs = torch.linspace(-1.0, 1.0, image_size, device=device, dtype=dtype)
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
    grid = torch.stack([grid_x, grid_y], dim=-1).view(1, 1, image_size, image_size, 2)

    # Convert log sigmas back to positive radii-like standard deviations. The
    # clamp prevents tiny sigmas from becoming needle-thin maps and huge sigmas
    # from washing out the whole condition image.
    centers = slot_centers.to(device=device, dtype=dtype)[..., :2]
    sigmas = (
        slot_log_sigmas.to(device=device, dtype=dtype)
        .exp()
        .mul(float(sigma_scale))
        .clamp(min=0.03, max=2.0)
    )
    # Standard diagonal Gaussian: exp(-0.5 * ((dx/sx)^2 + (dy/sy)^2)).
    normalized_delta = (grid - centers[:, :, None, None, :]) / sigmas[:, :, None, None, :]
    maps_by_slot = torch.exp(-0.5 * normalized_delta.pow(2).sum(dim=-1))
    maps_by_slot = maps_by_slot * slot_mask.to(device=device, dtype=dtype)[:, :, None, None]

    # ControlNet's default conditioning stem accepts 3 channels. For our common
    # two-object prompts, channels 0 and 1 are object-specific, while channel 2
    # is a union map that says "some object belongs here."
    layout = torch.zeros(
        batch_size,
        channels,
        image_size,
        image_size,
        device=device,
        dtype=dtype,
    )
    if max_slots > 0:
        layout[:, 0] = maps_by_slot[:, 0]
    if max_slots > 1:
        layout[:, 1] = maps_by_slot[:, 1]
    layout[:, 2] = maps_by_slot.max(dim=1).values

    # If a future dataset has more than two slots and the caller asks for more
    # channels, expose additional slot maps after the union channel.
    for slot_index in range(2, min(max_slots, channels - 1)):
        layout[:, slot_index + 1] = maps_by_slot[:, slot_index]
    return layout.clamp(0.0, 1.0)
