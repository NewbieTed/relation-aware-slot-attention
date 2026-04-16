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
    ) -> None:
        self.dataset_dir = Path(dataset_dir)
        self.image_size = image_size
        self.prompt_prefix = prompt_prefix

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
        image_path = self.dataset_dir / row["file_name"]
        image = Image.open(image_path).convert("RGB")
        image = _center_crop_to_square(image, self.image_size)

        return TrainingItem(
            pixel_values=_image_to_tensor(image),
            prompt=prompt_from_scop_depth_row(row, prefix=self.prompt_prefix),
            scene_graph=scene_graph_payload_from_row(row),
            metadata=row,
        )


def collate_training_items(items: list[TrainingItem]) -> dict[str, Any]:
    return {
        "pixel_values": torch.stack([item.pixel_values for item in items]).contiguous(),
        "prompts": [item.prompt for item in items],
        "scene_graphs": [item.scene_graph for item in items],
        "metadata": [item.metadata for item in items],
    }

