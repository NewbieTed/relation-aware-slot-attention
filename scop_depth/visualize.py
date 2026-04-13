import random
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .dataset_reader import DatasetReader
from .depth import DepthAnythingV2Estimator, DepthConfig
from .geometry import segmentation_to_mask
from .models import CocoInstanceAnnotation


def _load_font(size: int = 24, font_path: str | None = None) -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype(font_path if font_path else "DejaVuSerif.ttf", size)
    except IOError:
        return ImageFont.load_default()


def _draw_text_block(
    draw: ImageDraw.ImageDraw,
    lines: list[str],
    origin: tuple[int, int],
    font: ImageFont.ImageFont,
    line_fill: str = "white",
) -> None:
    x, y = origin
    if not lines:
        return

    line_heights: list[int] = []
    max_width = 0
    for line in lines:
        left, top, right, bottom = draw.textbbox((0, 0), line, font=font)
        max_width = max(max_width, right - left)
        line_heights.append(bottom - top)

    padding = 6
    line_gap = 4
    block_height = sum(line_heights) + line_gap * (len(lines) - 1) + padding * 2
    draw.rectangle(
        [x, y, x + max_width + padding * 2, y + block_height],
        fill="black",
    )

    cursor_y = y + padding
    for line, line_height in zip(lines, line_heights):
        draw.text((x + padding, cursor_y), line, font=font, fill=line_fill)
        cursor_y += line_height + line_gap


def visualize_object_pair(
    annot_pair: tuple[CocoInstanceAnnotation, CocoInstanceAnnotation],
    reader: DatasetReader,
    category_dict: dict[int, str],
    font_path: str | None = None,
) -> tuple[Image.Image, Image.Image]:
    """
    Visualize a pair of objects with both bbox outlines and COCO segmentation overlays.

    Args:
        annot_pair: Tuple of two annotations to visualize
        reader: DatasetReader instance to access images
        category_dict: Dictionary mapping category IDs to names
        font_path: Optional path to font file

    Returns:
        tuple: (original_image, image_with_annotations)
    """
    a1, a2 = annot_pair
    assert a1.image_id == a2.image_id

    # Get the image using the reader
    image = reader.get_image(a1.image_id)
    # Try to load font, fallback to default if specified font isn't available
    font = _load_font(size=24, font_path=font_path)

    colors = ["cyan", "magenta"]
    overlay = np.asarray(image.copy()).copy()

    for i, annot in enumerate([a1, a2]):
        rgb = (0, 255, 255) if i == 0 else (255, 0, 255)
        # The overlay is informational only. SCOP's 2D constraints still come from
        # bbox geometry; we show the visible-instance masks to make the depth pooling
        # regions easier to inspect in samples.
        mask = segmentation_to_mask(annot, image.width, image.height)
        if mask is not None and np.any(mask):
            tint = np.zeros_like(overlay)
            tint[..., 0] = rgb[0]
            tint[..., 1] = rgb[1]
            tint[..., 2] = rgb[2]
            overlay[mask] = (0.45 * overlay[mask] + 0.55 * tint[mask]).astype(
                np.uint8
            )

    image_with_annotations = Image.fromarray(overlay)
    draw = ImageDraw.Draw(image_with_annotations)

    for i, annot in enumerate([a1, a2]):
        x, y, w, h = annot.bbox
        category = category_dict[annot.category_id]
        color = colors[i % len(colors)]

        draw.rectangle([x, y, x + w, y + h], outline=color, width=3)
        text_bbox = draw.textbbox((x, y), category, font=font)
        draw.rectangle(text_bbox, fill="black")
        draw.text((x, y), category, font=font, fill=color)

    return image, image_with_annotations


def create_sample_visualization(
    reader: DatasetReader,
    image_id_to_relationships: dict[int, list[dict[str, Any]]],
    output_dir: Path,
    num_samples: int = 5,
) -> None:
    """
    Create sample visualizations of spatial relationships.

    Args:
        reader: DatasetReader instance to access data
        image_id_to_relationships: Dictionary mapping image IDs to relationships
        output_dir: Directory to save visualizations
        num_samples: Number of samples to create
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for old_sample in output_dir.glob("*.jpg"):
        old_sample.unlink()

    # Load category information from reader
    coco_inst = reader.get_instances_annotations()
    category_dict = {cat["id"]: cat["name"] for cat in coco_inst["categories"]}
    depth_estimator: DepthAnythingV2Estimator | None = None

    # Get sample image IDs
    sample_image_ids = random.sample(
        list(image_id_to_relationships.keys()),
        min(num_samples, len(image_id_to_relationships)),
    )

    for image_id in sample_image_ids:
        # Prefer an occlusion example when available so depth-specific labels
        # like "hidden by" show up in the sample set.
        image_relationships = image_id_to_relationships[image_id]
        relationship = next(
            (
                rel
                for rel in image_relationships
                if any(item[1] == "hidden by" for item in rel["relative_positions"])
            ),
            image_relationships[0],
        )
        annot_pair = relationship["annotations"]
        relative_positions = relationship["relative_positions"]
        relation_kind = "mixed"
        if any(item[1] == "hidden by" for item in relative_positions):
            relation_kind = "hidden_by"
        elif any(item[1] in {"in front of", "behind"} for item in relative_positions):
            relation_kind = "depth"
        elif relative_positions:
            relation_kind = "planar"

        # Create visualization
        _, img_with_annotations = visualize_object_pair(
            (annot_pair[0], annot_pair[1]), reader, category_dict
        )

        font = _load_font(size=22)

        # Separate 2D and depth relations so the visualization makes the new
        # front/behind labels obvious at a glance.
        depth_phrases = {"in front of", "behind", "hidden by"}
        planar_lines = [
            f"{subj} {rel} {obj}"
            for subj, rel, obj in relative_positions
            if rel not in depth_phrases
        ]
        depth_lines = [
            f"{subj} {rel} {obj}"
            for subj, rel, obj in relative_positions
            if rel in depth_phrases
        ]

        draw = ImageDraw.Draw(img_with_annotations)
        text_lines = []
        if planar_lines:
            text_lines.append("2D:")
            text_lines.extend(planar_lines)
        if depth_lines:
            text_lines.append("Depth:")
            text_lines.extend(depth_lines)
        _draw_text_block(draw, text_lines, (10, 10), font)

        final_image = img_with_annotations
        depth_summary = relationship.get("depth")
        depth_preview = relationship.get("depth_preview")

        if depth_summary is not None and depth_preview is None:
            if depth_estimator is None:
                depth_estimator = DepthAnythingV2Estimator(DepthConfig())
            raw_depth_map = depth_estimator.predict_depth(reader.get_image(image_id))
            depth_preview = depth_estimator.render_depth_preview(raw_depth_map)

        if depth_preview is not None:
            depth_panel = depth_preview.resize(
                img_with_annotations.size, resample=Image.Resampling.BILINEAR
            )
            panel_width = img_with_annotations.width + depth_panel.width
            panel_height = max(img_with_annotations.height, depth_panel.height)
            combined = Image.new("RGB", (panel_width, panel_height), "black")
            combined.paste(img_with_annotations, (0, 0))
            combined.paste(depth_panel, (img_with_annotations.width, 0))

            combined_draw = ImageDraw.Draw(combined)
            right_lines = ["Depth Anything V2"]

            if depth_summary is not None:
                ordering = depth_summary["ordering"]
                delta = depth_summary["delta_median"]
                # Convert internal slot ordering into object names so sample panels are
                # readable without needing to know which annotation was "bbox1" or "bbox2".
                if ordering == "bbox1_closer":
                    left_name = category_dict[annot_pair[0].category_id]
                    right_name = category_dict[annot_pair[1].category_id]
                    right_lines.append(
                        f"depth: {left_name} closer than {right_name} ({delta:+.3f})"
                    )
                elif ordering == "bbox2_closer":
                    left_name = category_dict[annot_pair[0].category_id]
                    right_name = category_dict[annot_pair[1].category_id]
                    right_lines.append(
                        f"depth: {right_name} closer than {left_name} ({delta:+.3f})"
                    )
                else:
                    right_lines.append(f"depth: ambiguous ({delta:+.3f})")
                right_lines.extend(depth_lines)
            _draw_text_block(
                combined_draw,
                right_lines,
                (img_with_annotations.width + 10, 10),
                font,
            )

            final_image = combined

        # Save the image
        output_path = output_dir / f"{relation_kind}_image_{image_id}.jpg"
        final_image.save(output_path)

    print(f"Created {len(sample_image_ids)} sample visualizations in {output_dir}")
