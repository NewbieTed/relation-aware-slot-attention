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
