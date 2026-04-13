from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageDraw

from .models import CocoInstanceAnnotation


@dataclass(frozen=True)
class AnnotationGeometry:
    """Lightweight geometry container for per-instance rasterized masks."""

    mask: np.ndarray | None = None


def _decode_uncompressed_rle(counts: list[int], height: int, width: int) -> np.ndarray:
    """Decode COCO's uncompressed RLE encoding into a boolean mask."""

    total = height * width
    values = np.zeros(total, dtype=np.uint8)
    cursor = 0
    value = 0

    for run_length in counts:
        if run_length <= 0:
            value = 1 - value
            continue
        end = min(cursor + run_length, total)
        if value == 1:
            values[cursor:end] = 1
        cursor = end
        value = 1 - value
        if cursor >= total:
            break

    return values.reshape((height, width), order="F").astype(bool)


def _polygon_to_mask(
    polygons: list[list[float]], image_width: int, image_height: int
) -> np.ndarray | None:
    """Rasterize polygon segmentations into a full-image boolean mask."""

    if not polygons:
        return None

    mask_image = Image.new("1", (image_width, image_height), 0)
    draw = ImageDraw.Draw(mask_image)

    drew_any_polygon = False
    for polygon in polygons:
        if len(polygon) < 6:
            continue
        points = list(zip(polygon[0::2], polygon[1::2]))
        draw.polygon(points, fill=1)
        drew_any_polygon = True

    if not drew_any_polygon:
        return None

    return np.asarray(mask_image, dtype=bool)


def segmentation_to_mask(
    annot: CocoInstanceAnnotation, image_width: int, image_height: int
) -> np.ndarray | None:
    """Return a boolean mask for a COCO instance annotation when possible."""

    segmentation = annot.segmentation

    if isinstance(segmentation, list):
        return _polygon_to_mask(segmentation, image_width, image_height)

    if isinstance(segmentation, CocoInstanceAnnotation.Segmentationdict):
        if isinstance(segmentation.counts, list):
            mask_height, mask_width = segmentation.size
            return _decode_uncompressed_rle(
                segmentation.counts, mask_height, mask_width
            )

    return None
