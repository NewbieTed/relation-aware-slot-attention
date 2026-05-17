from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from .prompts import prompt_from_scop_depth_row, scene_graph_payload_from_row


def load_metadata_rows(dataset_dir: Path) -> list[dict[str, Any]]:
    metadata_path = dataset_dir / "metadata.jsonl"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing SCOP-Depth metadata file: {metadata_path}")

    rows: list[dict[str, Any]] = []
    for line in metadata_path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        rows.append(json.loads(line))
    if not rows:
        raise ValueError(f"No training rows found in {metadata_path}")
    return rows


def _center_crop_to_square(image: Image.Image, size: int) -> Image.Image:
    width, height = image.size
    crop_size = min(width, height)
    left = (width - crop_size) // 2
    top = (height - crop_size) // 2
    image = image.crop((left, top, left + crop_size, top + crop_size))
    return image.resize((size, size), resample=Image.Resampling.BICUBIC)


def _image_to_tensor(image: Image.Image) -> torch.Tensor:
    array = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    array = (array * 2.0) - 1.0
    array = np.transpose(array, (2, 0, 1))
    return torch.from_numpy(array)


@dataclass(frozen=True)
class TrainingItem:
    pixel_values: torch.Tensor
    prompt: str
    scene_graph: dict[str, Any]
    metadata: dict[str, Any]
    image_size: tuple[int, int]


class SCOPDepthTextToImageDataset(Dataset[TrainingItem]):
    """PyTorch dataset for SCOP-Depth text-to-image baseline training."""

    def __init__(
        self,
        dataset_dir: str | Path,
        *,
        image_size: int = 512,
        prompt_prefix: str = "a photo of",
        limit_rows: int | None = None,
        shuffle_rows: bool = False,
        seed: int = 42,
        rows: list[dict[str, Any]] | None = None,
        load_images: bool = True,
    ) -> None:
        self.dataset_dir = Path(dataset_dir)
        self.image_size = image_size
        self.prompt_prefix = prompt_prefix
        self.load_images = load_images

        if rows is None:
            rows = load_metadata_rows(self.dataset_dir)
        if shuffle_rows:
            rng = random.Random(seed)
            rng.shuffle(rows)
        if limit_rows is not None:
            rows = rows[:limit_rows]
        if not rows:
            raise ValueError("SCOP-Depth dataset is empty after applying row limits")

        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> TrainingItem:
        row = self.rows[index]
        if self.load_images:
            image_path = self.dataset_dir / row["file_name"]
            if not image_path.exists():
                raise FileNotFoundError(
                    "Missing SCOP-Depth image file: "
                    f"{image_path}. "
                    "If metadata.jsonl exists but images were deleted or never materialized, "
                    "rebuild the subset with "
                    "`python -m training.materialize_images --dataset-dir <scop_depth_dir> "
                    "--coco-root <coco_root>`."
                )
            image = Image.open(image_path).convert("RGB")
            original_image_size = image.size
            image = _center_crop_to_square(image, self.image_size)
            pixel_values = _image_to_tensor(image)
        else:
            original_image_size = tuple(row.get("image_size") or (self.image_size, self.image_size))
            pixel_values = torch.empty(0)

        return TrainingItem(
            pixel_values=pixel_values,
            prompt=prompt_from_scop_depth_row(row, prefix=self.prompt_prefix),
            scene_graph=scene_graph_payload_from_row(row),
            metadata=row,
            image_size=original_image_size,
        )


def collate_training_items(items: list[TrainingItem]) -> dict[str, Any]:
    if all(item.pixel_values.numel() == 0 for item in items):
        pixel_values = torch.empty(0)
    else:
        pixel_values = torch.stack([item.pixel_values for item in items]).contiguous()
    return {
        "pixel_values": pixel_values,
        "prompts": [item.prompt for item in items],
        "scene_graphs": [item.scene_graph for item in items],
        "metadata": [item.metadata for item in items],
        "image_sizes": [item.image_size for item in items],
    }


def _compute_split_counts(
    total_rows: int,
    eval_fraction: float,
    test_fraction: float,
) -> tuple[int, int, int]:
    if total_rows <= 0:
        raise ValueError("Cannot split an empty dataset")
    if not (0.0 <= eval_fraction < 1.0 and 0.0 <= test_fraction < 1.0):
        raise ValueError("Eval/test fractions must lie in [0, 1)")
    if eval_fraction + test_fraction >= 1.0:
        raise ValueError("Eval/test fractions must sum to less than 1")

    eval_count = int(round(total_rows * eval_fraction))
    test_count = int(round(total_rows * test_fraction))

    if eval_fraction > 0 and eval_count == 0 and total_rows >= 3:
        eval_count = 1
    if test_fraction > 0 and test_count == 0 and total_rows - eval_count >= 2:
        test_count = 1

    while eval_count + test_count >= total_rows:
        if eval_count >= test_count and eval_count > 0:
            eval_count -= 1
        elif test_count > 0:
            test_count -= 1
        else:
            break

    train_count = total_rows - eval_count - test_count
    if train_count <= 0:
        raise ValueError("Split configuration left no rows for training")
    return train_count, eval_count, test_count


def build_dataset_splits(
    dataset_dir: str | Path,
    *,
    image_size: int = 512,
    prompt_prefix: str = "a photo of",
    limit_rows: int | None = None,
    seed: int = 42,
    eval_fraction: float = 0.1,
    test_fraction: float = 0.1,
    load_images: bool = True,
    prompt_filter: str | None = None,
) -> dict[str, SCOPDepthTextToImageDataset]:
    dataset_path = Path(dataset_dir)
    rows = load_metadata_rows(dataset_path)
    if prompt_filter:
        normalized_filter = " ".join(prompt_filter.split()).lower()
        rows = [
            row
            for row in rows
            if " ".join(prompt_from_scop_depth_row(row, prefix=prompt_prefix).split()).lower()
            == normalized_filter
        ]
    rng = random.Random(seed)
    rng.shuffle(rows)
    if limit_rows is not None:
        rows = rows[:limit_rows]
    if not rows:
        raise ValueError("SCOP-Depth dataset is empty after applying row limits")

    train_count, eval_count, test_count = _compute_split_counts(
        len(rows),
        eval_fraction=eval_fraction,
        test_fraction=test_fraction,
    )

    train_rows = rows[:train_count]
    eval_rows = rows[train_count : train_count + eval_count]
    test_rows = rows[train_count + eval_count : train_count + eval_count + test_count]

    return {
        "train": SCOPDepthTextToImageDataset(
            dataset_path,
            image_size=image_size,
            prompt_prefix=prompt_prefix,
            rows=train_rows,
            load_images=load_images,
        ),
        "eval": SCOPDepthTextToImageDataset(
            dataset_path,
            image_size=image_size,
            prompt_prefix=prompt_prefix,
            rows=eval_rows,
            load_images=load_images,
        ),
        "test": SCOPDepthTextToImageDataset(
            dataset_path,
            image_size=image_size,
            prompt_prefix=prompt_prefix,
            rows=test_rows,
            load_images=load_images,
        ),
    }
