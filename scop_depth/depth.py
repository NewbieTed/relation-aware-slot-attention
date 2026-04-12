from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageOps
from transformers import AutoImageProcessor, AutoModelForDepthEstimation

from .models import BoundingBox

DEFAULT_DEPTH_MODEL_ID = "depth-anything/Depth-Anything-V2-Base-hf"


def _has_cuda() -> bool:
    return torch.cuda.is_available()


def _has_mps() -> bool:
    return torch.backends.mps.is_built() and torch.backends.mps.is_available()


def resolve_torch_device(device_preference: str = "auto") -> str:
    """Resolve the torch device to use for depth inference."""
    if device_preference == "auto":
        if _has_cuda():
            return "cuda"
        if _has_mps():
            return "mps"
        return "cpu"

    if device_preference == "cuda":
        if _has_cuda():
            return "cuda"
        print("Depth Anything: CUDA requested but not available, falling back to CPU.")
        return "cpu"

    if device_preference == "mps":
        if _has_mps():
            return "mps"
        print("Depth Anything: MPS requested but not available, falling back to CPU.")
        return "cpu"

    return "cpu"


@dataclass(frozen=True)
class DepthConfig:
    model_id: str = DEFAULT_DEPTH_MODEL_ID
    device: str = "auto"
    min_separation: float = 0.2
    center_crop_ratio: float = 0.6
    hidden_overlap_threshold: float = 0.4
    include_order_labels: bool = False


class DepthAnythingV2Estimator:
    """
    Thin wrapper around the Hugging Face Depth Anything V2 checkpoints.

    The relative-depth model is used conservatively here:
    - we summarize depth inside each bounding box
    - we only emit front/behind labels when the separation is clearly large
    """

    def __init__(self, config: DepthConfig):
        self.config = config
        self.device = resolve_torch_device(config.device)
        self.processor = None
        self.model = None
        self._warned_cpu_fallback = False
        self._mps_fallback_enabled = False

    def _enable_mps_fallback(self) -> None:
        if self.device != "mps" or self._mps_fallback_enabled:
            return

        os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
        self._mps_fallback_enabled = True

    def load(self) -> None:
        if self.processor is not None and self.model is not None:
            return

        self._enable_mps_fallback()
        try:
            self.processor = AutoImageProcessor.from_pretrained(
                self.config.model_id, use_fast=False
            )
            self.model = AutoModelForDepthEstimation.from_pretrained(
                self.config.model_id
            ).to(self.device)
        except Exception:
            # Fall back to locally cached files so visualization rerenders and
            # offline runs still work after the checkpoint has been downloaded.
            self.processor = AutoImageProcessor.from_pretrained(
                self.config.model_id, use_fast=False, local_files_only=True
            )
            self.model = AutoModelForDepthEstimation.from_pretrained(
                self.config.model_id, local_files_only=True
            ).to(self.device)
        self.model.eval()

    def _run_model(self, pixel_values: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            outputs = self.model(pixel_values=pixel_values)
            return outputs.predicted_depth

    def predict_depth(self, image: Image.Image) -> np.ndarray:
        self.load()

        rgb_image = image.convert("RGB")
        inputs = self.processor(images=rgb_image, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(self.device)

        try:
            predicted_depth = self._run_model(pixel_values)
        except NotImplementedError as exc:
            if self.device != "mps":
                raise

            # Some Apple Silicon torch builds still miss ops used by the DINOv2
            # backbone. We first try the official MPS CPU-op fallback path and,
            # if that still fails, move the whole model to CPU.
            if not self._warned_cpu_fallback:
                print(
                    "Depth Anything: MPS hit an unsupported op; retrying with CPU fallback enabled."
                )
                self._warned_cpu_fallback = True
            self._enable_mps_fallback()
            try:
                predicted_depth = self._run_model(pixel_values)
            except Exception:
                print(
                    "Depth Anything: MPS fallback was insufficient, moving depth inference to CPU."
                )
                self.device = "cpu"
                self.model = self.model.to(self.device)
                pixel_values = inputs["pixel_values"].to(self.device)
                predicted_depth = self._run_model(pixel_values)
        except RuntimeError as exc:
            if self.device != "mps" or "MPS" not in str(exc):
                raise

            if not self._warned_cpu_fallback:
                print(
                    "Depth Anything: MPS raised a runtime error; retrying with CPU fallback enabled."
                )
                self._warned_cpu_fallback = True
            self._enable_mps_fallback()
            try:
                predicted_depth = self._run_model(pixel_values)
            except Exception:
                print(
                    "Depth Anything: MPS fallback was insufficient, moving depth inference to CPU."
                )
                self.device = "cpu"
                self.model = self.model.to(self.device)
                pixel_values = inputs["pixel_values"].to(self.device)
                predicted_depth = self._run_model(pixel_values)

        predicted_depth = F.interpolate(
            predicted_depth.unsqueeze(1),
            size=rgb_image.size[::-1],
            mode="bicubic",
            align_corners=False,
        ).squeeze(0).squeeze(0)

        return predicted_depth.detach().cpu().numpy().astype(np.float32)

    @staticmethod
    def normalize_depth(depth_map: np.ndarray) -> np.ndarray:
        depth_min = float(depth_map.min())
        depth_max = float(depth_map.max())
        denom = depth_max - depth_min
        if denom <= 1e-8:
            return np.zeros_like(depth_map, dtype=np.float32)
        return ((depth_map - depth_min) / denom).astype(np.float32)

    @staticmethod
    def summarize_bbox_depth(
        depth_map: np.ndarray, bbox: BoundingBox, center_crop_ratio: float = 1.0
    ) -> dict[str, float] | None:
        x, y, w, h = bbox
        if center_crop_ratio <= 0 or center_crop_ratio > 1:
            raise ValueError("center_crop_ratio must be in the interval (0, 1]")

        crop_w = w * center_crop_ratio
        crop_h = h * center_crop_ratio
        crop_x = x + (w - crop_w) / 2
        crop_y = y + (h - crop_h) / 2

        x0 = max(0, int(round(crop_x)))
        y0 = max(0, int(round(crop_y)))
        x1 = min(depth_map.shape[1], int(round(crop_x + crop_w)))
        y1 = min(depth_map.shape[0], int(round(crop_y + crop_h)))

        if x1 <= x0 or y1 <= y0:
            return None

        patch = depth_map[y0:y1, x0:x1]
        if patch.size == 0:
            return None

        return {
            "mean": float(np.mean(patch)),
            "median": float(np.median(patch)),
            "min": float(np.min(patch)),
            "max": float(np.max(patch)),
            "std": float(np.std(patch)),
            "center_crop_ratio": float(center_crop_ratio),
        }

    def compare_depth_stats(
        self, stats1: dict[str, float], stats2: dict[str, float]
    ) -> dict[str, Any] | None:
        # For the relative Depth Anything checkpoints, larger values behave like
        # stronger inverse depth, so we interpret larger values as "closer".
        delta = stats1["median"] - stats2["median"]
        abs_delta = abs(delta)

        if abs_delta < self.config.min_separation:
            ordering = "ambiguous"
            closer_index = None
        elif delta > 0:
            ordering = "bbox1_closer"
            closer_index = 0
        else:
            ordering = "bbox2_closer"
            closer_index = 1

        return {
            "bbox1": stats1,
            "bbox2": stats2,
            "delta_median": float(delta),
            "abs_delta_median": float(abs_delta),
            "ordering": ordering,
            "closer_index": closer_index,
            "threshold": self.config.min_separation,
        }

    def compare_bboxes(
        self, normalized_depth_map: np.ndarray, bbox1: BoundingBox, bbox2: BoundingBox
    ) -> dict[str, Any] | None:
        stats1 = self.summarize_bbox_depth(
            normalized_depth_map, bbox1, center_crop_ratio=self.config.center_crop_ratio
        )
        stats2 = self.summarize_bbox_depth(
            normalized_depth_map, bbox2, center_crop_ratio=self.config.center_crop_ratio
        )
        if stats1 is None or stats2 is None:
            return None

        return self.compare_depth_stats(stats1, stats2)

    @staticmethod
    def render_depth_preview(depth_map: np.ndarray) -> Image.Image:
        normalized = DepthAnythingV2Estimator.normalize_depth(depth_map)
        depth_uint8 = np.clip(normalized * 255.0, 0, 255).astype(np.uint8)
        grayscale = Image.fromarray(depth_uint8, mode="L")
        return ImageOps.colorize(grayscale, black="#0b132b", white="#f4f1de")
