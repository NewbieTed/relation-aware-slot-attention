from __future__ import annotations

from typing import Any

import torch


def bbox_centers_after_crop(
    metadata_rows: list[dict[str, Any]],
    image_sizes: list[tuple[int, int]],
    max_nodes: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    targets = torch.zeros(len(metadata_rows), max_nodes, 3, device=device)
    mask = torch.zeros(len(metadata_rows), max_nodes, dtype=torch.bool, device=device)
    for batch_index, (row, (width, height)) in enumerate(zip(metadata_rows, image_sizes)):
        crop_size = min(width, height)
        left = (width - crop_size) / 2.0
        top = (height - crop_size) / 2.0
        depth = row.get("depth")
        depth_values = [
            float(depth["bbox1"]["median"]) if depth else 0.0,
            float(depth["bbox2"]["median"]) if depth else 0.0,
        ]
        for node_index, annot in enumerate(row["annots"][:max_nodes]):
            x, y, w, h = annot["bbox"]
            cx = ((x + w / 2.0) - left) / crop_size
            cy = ((y + h / 2.0) - top) / crop_size
            cx = max(0.0, min(1.0, cx))
            cy = max(0.0, min(1.0, cy))
            cz = depth_values[node_index] if node_index < len(depth_values) else 0.0
            targets[batch_index, node_index] = torch.tensor(
                [cx * 2 - 1, cy * 2 - 1, cz * 2 - 1],
                device=device,
            )
            mask[batch_index, node_index] = True
    return targets, mask


def bbox_log_sigmas_after_crop(
    metadata_rows: list[dict[str, Any]],
    image_sizes: list[tuple[int, int]],
    max_nodes: int,
    device: torch.device,
    *,
    min_sigma: float = 0.03,
) -> tuple[torch.Tensor, torch.Tensor]:
    targets = torch.zeros(len(metadata_rows), max_nodes, 2, device=device)
    mask = torch.zeros(len(metadata_rows), max_nodes, dtype=torch.bool, device=device)
    for batch_index, (row, (width, height)) in enumerate(zip(metadata_rows, image_sizes)):
        crop_size = min(width, height)
        for node_index, annot in enumerate(row["annots"][:max_nodes]):
            _, _, bbox_w, bbox_h = annot["bbox"]
            sigma_x = max(float(bbox_w) / crop_size, min_sigma)
            sigma_y = max(float(bbox_h) / crop_size, min_sigma)
            targets[batch_index, node_index] = torch.tensor(
                [sigma_x, sigma_y],
                device=device,
            ).log()
            mask[batch_index, node_index] = True
    return targets, mask


def _depth_extent_from_row(row: dict[str, Any], node_index: int, fallback: float) -> float:
    """Return an approximate normalized z-extent for one object.

    SCOP-depth stores monocular depth statistics, not full metric 3D object
    meshes. When a spread statistic is available we use it; otherwise the z-size
    falls back to a conservative fraction of the 2D object scale.
    """

    depth = row.get("depth")
    if not depth:
        return fallback
    key = f"bbox{node_index + 1}"
    stats = depth.get(key, {})
    for field in ("iqr", "std", "mad", "range"):
        if field in stats:
            return max(float(stats[field]), fallback)
    return fallback


def bbox_log_sizes_3d_after_crop(
    metadata_rows: list[dict[str, Any]],
    image_sizes: list[tuple[int, int]],
    max_nodes: int,
    device: torch.device,
    *,
    min_size: float = 0.03,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return approximate log 3D bbox sizes in normalized scene coordinates.

    The x/y size comes directly from the cropped COCO bbox. The z size uses
    depth-spread metadata when present, otherwise a small fallback tied to the
    2D object scale. Values are log-space so the GNN can predict unconstrained
    real values while the decoded sizes stay positive.
    """

    targets = torch.zeros(len(metadata_rows), max_nodes, 3, device=device)
    mask = torch.zeros(len(metadata_rows), max_nodes, dtype=torch.bool, device=device)
    for batch_index, (row, (width, height)) in enumerate(zip(metadata_rows, image_sizes)):
        crop_size = min(width, height)
        for node_index, annot in enumerate(row["annots"][:max_nodes]):
            _, _, bbox_w, bbox_h = annot["bbox"]
            size_x = max(float(bbox_w) / crop_size, min_size)
            size_y = max(float(bbox_h) / crop_size, min_size)
            fallback_z = max((size_x + size_y) * 0.25, min_size)
            size_z = max(_depth_extent_from_row(row, node_index, fallback_z), min_size)
            targets[batch_index, node_index] = torch.tensor(
                [size_x, size_y, size_z],
                device=device,
            ).log()
            mask[batch_index, node_index] = True
    return targets, mask


def bbox_minmax_3d_after_crop(
    metadata_rows: list[dict[str, Any]],
    image_sizes: list[tuple[int, int]],
    max_nodes: int,
    device: torch.device,
    *,
    min_size: float = 0.03,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return 3D_SLN-style normalized min/max boxes.

    The target format is ``[x0, y0, z0, x1, y1, z1]`` in ``[0, 1]``. For x/y,
    coordinates are measured after the square center-crop used by training
    images. For z, SCOP-depth gives monocular normalized depth statistics, so
    the min/max bounds are approximated from the median depth and available
    depth spread.
    """

    targets = torch.zeros(len(metadata_rows), max_nodes, 6, device=device)
    mask = torch.zeros(len(metadata_rows), max_nodes, dtype=torch.bool, device=device)
    for batch_index, (row, (width, height)) in enumerate(zip(metadata_rows, image_sizes)):
        crop_size = min(width, height)
        left = (width - crop_size) / 2.0
        top = (height - crop_size) / 2.0
        depth = row.get("depth")
        depth_values = [
            float(depth["bbox1"]["median"]) if depth else 0.5,
            float(depth["bbox2"]["median"]) if depth else 0.5,
        ]
        for node_index, annot in enumerate(row["annots"][:max_nodes]):
            x, y, bbox_w, bbox_h = annot["bbox"]
            x0 = max(0.0, min(1.0, (float(x) - left) / crop_size))
            y0 = max(0.0, min(1.0, (float(y) - top) / crop_size))
            x1 = max(0.0, min(1.0, (float(x) + float(bbox_w) - left) / crop_size))
            y1 = max(0.0, min(1.0, (float(y) + float(bbox_h) - top) / crop_size))
            if x1 < x0:
                x0, x1 = x1, x0
            if y1 < y0:
                y0, y1 = y1, y0

            cz = depth_values[node_index] if node_index < len(depth_values) else 0.5
            size_x = max(x1 - x0, min_size)
            size_y = max(y1 - y0, min_size)
            fallback_z = max((size_x + size_y) * 0.25, min_size)
            size_z = max(_depth_extent_from_row(row, node_index, fallback_z), min_size)
            z0 = max(0.0, min(1.0, cz - size_z * 0.5))
            z1 = max(0.0, min(1.0, cz + size_z * 0.5))

            targets[batch_index, node_index] = torch.tensor(
                [x0, y0, z0, x1, y1, z1],
                device=device,
            )
            mask[batch_index, node_index] = True
    return targets, mask
