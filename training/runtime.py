"""Shared runtime helpers for SCOP-Depth training and FLUX evaluation."""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch

from .graph_modules import GraphSlotEncoder

DEFAULT_FLUX_MODEL_ID = "black-forest-labs/FLUX.1-dev"


def resolve_torch_device(device_preference: str = "auto") -> str:
    """Resolve a requested device string into an available PyTorch backend."""

    if device_preference == "auto":
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_built() and torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    if device_preference == "cuda" and not torch.cuda.is_available():
        print("CUDA requested but not available, falling back to CPU.")
        return "cpu"
    if (
        device_preference == "mps"
        and not (torch.backends.mps.is_built() and torch.backends.mps.is_available())
    ):
        print("MPS requested but not available, falling back to CPU.")
        return "cpu"
    return device_preference


def choose_weight_dtype(device: str, mixed_precision: str) -> torch.dtype:
    """Choose the weight dtype used for model loading on the active device."""

    if mixed_precision == "no":
        return torch.float32
    if device == "cuda" and mixed_precision == "fp16":
        return torch.float16
    if device == "cuda" and mixed_precision == "bf16":
        return torch.bfloat16
    return torch.float32


def set_seed(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch RNGs."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def is_tqdm_disabled(disable_tqdm_arg: object) -> bool:
    """Return whether progress bars should be hidden."""

    return bool(getattr(disable_tqdm_arg, "disable_tqdm", False))


def infer_graph_encoder_config(state_dict: dict[str, torch.Tensor]) -> tuple[int, int]:
    """Infer slot dimension and GNN depth from a saved graph encoder state."""

    slot_dim = int(state_dict["node_in.weight"].shape[0])
    layer_indices = {
        int(key.split(".")[1])
        for key in state_dict
        if key.startswith("layers.") and key.split(".")[1].isdigit()
    }
    return slot_dim, len(layer_indices)


def load_graph_encoder(
    *,
    path: Path,
    text_hidden_dim: int,
    device: str,
    dtype: torch.dtype,
) -> GraphSlotEncoder:
    """Load a saved graph encoder with its inferred architecture."""

    state_dict = torch.load(path, map_location="cpu")
    slot_dim, gnn_layers = infer_graph_encoder_config(state_dict)
    encoder = GraphSlotEncoder(
        text_hidden_dim=text_hidden_dim,
        slot_dim=slot_dim,
        num_layers=gnn_layers,
    ).to(device=device, dtype=dtype)
    encoder.load_state_dict(state_dict, strict=False)
    encoder.eval()
    return encoder
