"""Render predicted 3D object boxes into SeeThrough3D-style OSCR maps.

This is a lightweight approximation of the paper's occlusion-aware scene
representation. Each object is drawn as a translucent 3D cuboid projected into
the image plane. The front/back faces and depth ordering are encoded in RGB-like
channels so the FLUX transformer receives more than a flat 2D heatmap.
"""

from __future__ import annotations

import torch


OSCR_COLORS = torch.tensor(
    [
        [0.95, 0.20, 0.10],
        [0.10, 0.55, 0.95],
        [0.20, 0.85, 0.35],
        [0.95, 0.75, 0.15],
    ],
    dtype=torch.float32,
)


def render_oscr_boxes(
    *,
    centers: torch.Tensor,
    log_sizes: torch.Tensor,
    slot_mask: torch.Tensor,
    image_size: int = 512,
    alpha: float = 0.55,
) -> torch.Tensor:
    """Rasterize approximate translucent 3D boxes into an image tensor.

    Args:
        centers: ``[B, S, 3]`` normalized x/y/z centers in ``[-1, 1]``.
        log_sizes: ``[B, S, 3]`` log box sizes in normalized coordinates.
        slot_mask: ``[B, S]`` boolean mask for valid object slots.
        image_size: Output square side length.
        alpha: Per-object transparency used when compositing boxes.

    Returns:
        ``[B, 3, image_size, image_size]`` tensor in ``[-1, 1]`` suitable for
        the FLUX VAE.
    """

    device = centers.device
    dtype = torch.float32
    batch_size, slot_count, _ = centers.shape
    canvas = torch.zeros(batch_size, 3, image_size, image_size, device=device, dtype=dtype)
    colors = OSCR_COLORS.to(device=device, dtype=dtype)
    sizes = log_sizes.to(device=device, dtype=dtype).exp().clamp(min=0.03, max=2.0)
    centers = centers.to(device=device, dtype=dtype)

    for batch_index in range(batch_size):
        valid_indices = [
            index
            for index in range(slot_count)
            if bool(slot_mask[batch_index, index].item())
        ]
        # Draw far boxes first, near boxes later. Our depth convention follows
        # existing targets: larger z is visually nearer after normalization.
        valid_indices.sort(key=lambda idx: float(centers[batch_index, idx, 2].item()))
        for draw_order, slot_index in enumerate(valid_indices):
            cx, cy, cz = centers[batch_index, slot_index]
            sx, sy, sz = sizes[batch_index, slot_index]
            px = int(((float(cx) + 1.0) * 0.5) * image_size)
            py = int(((float(cy) + 1.0) * 0.5) * image_size)
            half_w = max(2, int(float(sx) * image_size * 0.5))
            half_h = max(2, int(float(sy) * image_size * 0.5))
            depth_offset = max(1, int(float(sz) * image_size * 0.12))
            x0 = max(0, px - half_w)
            x1 = min(image_size, px + half_w)
            y0 = max(0, py - half_h)
            y1 = min(image_size, py + half_h)
            bx0 = max(0, x0 - depth_offset)
            bx1 = min(image_size, x1 - depth_offset)
            by0 = max(0, y0 - depth_offset)
            by1 = min(image_size, y1 - depth_offset)
            color = colors[slot_index % len(colors)].clone()
            depth_tint = torch.tensor(
                [(float(cz) + 1.0) * 0.5, 0.25 + 0.15 * draw_order, 1.0 - (float(cz) + 1.0) * 0.5],
                device=device,
                dtype=dtype,
            ).clamp(0.0, 1.0)
            back_color = (color * 0.45 + depth_tint * 0.55).view(3, 1, 1)
            front_color = color.view(3, 1, 1)
            if bx1 > bx0 and by1 > by0:
                canvas[batch_index, :, by0:by1, bx0:bx1] = (
                    canvas[batch_index, :, by0:by1, bx0:bx1] * (1.0 - alpha * 0.55)
                    + back_color * (alpha * 0.55)
                )
            if x1 > x0 and y1 > y0:
                canvas[batch_index, :, y0:y1, x0:x1] = (
                    canvas[batch_index, :, y0:y1, x0:x1] * (1.0 - alpha)
                    + front_color * alpha
                )

    return canvas.mul(2.0).sub(1.0)
