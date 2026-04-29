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

    Channel 0 receives slot 0, channel 1 receives slot 1, and the final channel
    receives the union of all valid slots. This keeps the condition compatible
    with SD ControlNet's default 3-channel conditioning input.
    """

    if channels < 3:
        raise ValueError("Gaussian layout conditioning expects at least 3 channels")

    device = slot_centers.device
    dtype = torch.float32
    batch_size, max_slots, _ = slot_centers.shape
    ys = torch.linspace(-1.0, 1.0, image_size, device=device, dtype=dtype)
    xs = torch.linspace(-1.0, 1.0, image_size, device=device, dtype=dtype)
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
    grid = torch.stack([grid_x, grid_y], dim=-1).view(1, 1, image_size, image_size, 2)

    centers = slot_centers.to(device=device, dtype=dtype)[..., :2]
    sigmas = (
        slot_log_sigmas.to(device=device, dtype=dtype)
        .exp()
        .mul(float(sigma_scale))
        .clamp(min=0.03, max=2.0)
    )
    normalized_delta = (grid - centers[:, :, None, None, :]) / sigmas[:, :, None, None, :]
    maps_by_slot = torch.exp(-0.5 * normalized_delta.pow(2).sum(dim=-1))
    maps_by_slot = maps_by_slot * slot_mask.to(device=device, dtype=dtype)[:, :, None, None]

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
    for slot_index in range(2, min(max_slots, channels - 1)):
        layout[:, slot_index + 1] = maps_by_slot[:, slot_index]
    return layout.clamp(0.0, 1.0)
