from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset

CACHE_VERSION = 1


def file_sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def manifest_path(cache_dir: Path) -> Path:
    return cache_dir / "manifest.json"


def write_manifest(cache_dir: Path, manifest: dict[str, Any]) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    manifest_path(cache_dir).write_text(json.dumps(manifest, indent=2, sort_keys=True))


def load_manifest(cache_dir: Path) -> dict[str, Any]:
    path = manifest_path(cache_dir)
    if not path.exists():
        raise FileNotFoundError(f"Missing FLUX training cache manifest: {path}")
    return json.loads(path.read_text())


def build_expected_manifest(args: Any, *, graph_sha256: str, dtype_name: str) -> dict[str, Any]:
    return {
        "cache_version": CACHE_VERSION,
        "dataset_dir": str(args.dataset_dir),
        "model_id": args.model_id,
        "init_graph_encoder": str(args.init_graph_encoder),
        "init_graph_encoder_sha256": graph_sha256,
        "image_size": args.image_size,
        "oscr_size": args.oscr_size,
        "oscr_render_size": getattr(args, "oscr_render_size", None),
        "condition_renderer": args.condition_renderer,
        "oscr_face_alpha": args.oscr_face_alpha,
        "oscr_azimuth_degrees": args.oscr_azimuth_degrees,
        "prompt_prefix": args.prompt_prefix,
        "max_sequence_length": args.max_sequence_length,
        "seed": args.seed,
        "eval_fraction": args.eval_fraction,
        "test_fraction": args.test_fraction,
        "limit_rows": args.limit_rows,
        "slot_dim": args.slot_dim,
        "gnn_layers": args.gnn_layers,
        "dtype": dtype_name,
    }


def validate_manifest(cache_dir: Path, expected: dict[str, Any]) -> dict[str, Any]:
    manifest = load_manifest(cache_dir)
    mismatches = []
    for key, expected_value in expected.items():
        actual_value = manifest.get(key)
        if actual_value != expected_value:
            mismatches.append(f"{key}: cache={actual_value!r}, expected={expected_value!r}")
    if mismatches:
        joined = "\n  - ".join(mismatches)
        raise ValueError(
            "Precomputed FLUX cache does not match this training configuration:\n"
            f"  - {joined}\n"
            "Rebuild the cache or pass matching training arguments."
        )
    return manifest


@dataclass(frozen=True)
class CachedFluxTrainingItem:
    path: Path
    payload: dict[str, Any]


class CachedFluxTrainingDataset(Dataset[CachedFluxTrainingItem]):
    def __init__(self, cache_dir: str | Path, split: str) -> None:
        self.cache_dir = Path(cache_dir)
        self.split = split
        split_dir = self.cache_dir / split
        if not split_dir.exists():
            raise FileNotFoundError(f"Missing cached split directory: {split_dir}")
        self.paths = sorted(split_dir.glob("*.pt"))
        if not self.paths:
            raise ValueError(f"Cached split is empty: {split_dir}")
        self.rows = [torch.load(path, map_location="cpu")["metadata"] for path in self.paths]

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> CachedFluxTrainingItem:
        path = self.paths[index]
        return CachedFluxTrainingItem(path=path, payload=torch.load(path, map_location="cpu"))


def _stack(items: list[CachedFluxTrainingItem], key: str) -> torch.Tensor:
    return torch.stack([item.payload[key] for item in items]).contiguous()


def collate_cached_flux_training_items(items: list[CachedFluxTrainingItem]) -> dict[str, Any]:
    return {
        "clean_latents": _stack(items, "clean_latents"),
        "cond_latents": _stack(items, "cond_latents"),
        "prompt_embeds": _stack(items, "prompt_embeds"),
        "pooled_prompt_embeds": _stack(items, "pooled_prompt_embeds"),
        "text_ids": items[0].payload["text_ids"],
        "image_and_condition_ids": items[0].payload["image_and_condition_ids"],
        "call_ids": [
            [token_ids.to(dtype=torch.long) for token_ids in item.payload["call_ids"]]
            for item in items
        ],
        "cuboids_segmasks": _stack(items, "cuboids_segmasks").to(dtype=torch.uint8),
        "pred_center_abs_mean": torch.tensor(
            [float(item.payload["pred_center_abs_mean"]) for item in items],
            dtype=torch.float32,
        ),
        "pred_log_size_mean": torch.tensor(
            [float(item.payload["pred_log_size_mean"]) for item in items],
            dtype=torch.float32,
        ),
        "condition_latent_norm": torch.tensor(
            [float(item.payload["condition_latent_norm"]) for item in items],
            dtype=torch.float32,
        ),
        "binding_token_count": torch.tensor(
            [float(item.payload["binding_token_count"]) for item in items],
            dtype=torch.float32,
        ),
        "binding_mask_pct": torch.tensor(
            [float(item.payload["binding_mask_pct"]) for item in items],
            dtype=torch.float32,
        ),
        "prompts": [item.payload["prompt"] for item in items],
        "binding_prompts": [item.payload["binding_prompt"] for item in items],
        "scene_graphs": [item.payload["scene_graph"] for item in items],
        "metadata": [item.payload["metadata"] for item in items],
    }
