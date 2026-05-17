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


def normalize_graph_encoder_state_dict(checkpoint: object) -> dict[str, torch.Tensor]:
    """Return a plain graph encoder state dict from common checkpoint shapes."""

    if not isinstance(checkpoint, dict):
        raise TypeError(f"Expected graph encoder checkpoint dict, got {type(checkpoint)!r}")

    for wrapper_key in ("graph_encoder", "graph_encoder_state_dict", "state_dict", "model"):
        nested = checkpoint.get(wrapper_key)
        if isinstance(nested, dict):
            checkpoint = nested
            break

    state_dict: dict[str, torch.Tensor] = {}
    for key, value in checkpoint.items():
        if not isinstance(key, str) or not torch.is_tensor(value):
            continue
        normalized_key = key
        for prefix in ("module.", "graph_encoder."):
            if normalized_key.startswith(prefix):
                normalized_key = normalized_key[len(prefix) :]
        if normalized_key.startswith("node_in."):
            normalized_key = "node_proj." + normalized_key[len("node_in.") :]
        state_dict[normalized_key] = value

    if not state_dict:
        available = ", ".join(str(key) for key in list(checkpoint.keys())[:12])
        raise ValueError(f"No tensor state dict entries found in graph checkpoint. Keys: {available}")
    return state_dict


def infer_graph_encoder_config(state_dict: dict[str, torch.Tensor]) -> tuple[int, int, int, str, int, str, bool]:
    """Infer graph encoder dimensions and layout mode from a saved state."""

    node_weight = state_dict.get("node_proj.weight")
    if node_weight is None:
        node_weight = state_dict.get("node_in.weight")
    if node_weight is None:
        available = ", ".join(list(state_dict.keys())[:12])
        raise KeyError(
            "Could not infer graph encoder slot dimension because neither "
            f"'node_proj.weight' nor 'node_in.weight' exists. Keys: {available}"
        )
    slot_dim = int(node_weight.shape[0])
    text_hidden_dim = int(node_weight.shape[1])
    layer_indices = {
        int(key.split(".")[1])
        for key in state_dict
        if key.startswith("layers.") and key.split(".")[1].isdigit()
    }
    if "triple_prior_scene_head.3.weight" in state_dict:
        layout_mode = "triple_cvae"
        latent_dim = int(state_dict["triple_prior_scene_head.3.weight"].shape[0] // 2)
        decoder_mode = "triple_gnn" if "triple_decoder_layers.0.triple_mlp.0.weight" in state_dict else "mlp"
        decoder_box_residual = "triple_box_3d_delta_head.3.weight" in state_dict
    elif "prior_head.3.weight" in state_dict:
        layout_mode = "cvae"
        latent_dim = int(state_dict["prior_head.3.weight"].shape[0] // 2)
        decoder_mode = "triple_gnn"
        decoder_box_residual = False
    else:
        layout_mode = "deterministic"
        latent_dim = 64
        decoder_mode = "triple_gnn"
        decoder_box_residual = False
    return slot_dim, text_hidden_dim, len(layer_indices), layout_mode, latent_dim, decoder_mode, decoder_box_residual


def infer_text_encoder_type(text_hidden_dim: int) -> str:
    """Infer which FLUX text encoder produced graph node embeddings."""

    # FLUX.1-dev uses CLIP-L/14 hidden size 768 and T5-XXL hidden size 4096.
    if text_hidden_dim == 4096:
        return "t5"
    if text_hidden_dim == 768:
        return "clip"
    return "custom"


def load_graph_label_encoder(
    *,
    model_id: str,
    text_encoder_type: str,
    torch_dtype: torch.dtype,
    device: str,
) -> tuple[object, object, int]:
    """Load the frozen text encoder used to create GNN object-label embeddings."""

    normalized = text_encoder_type.lower()
    if normalized == "clip":
        from transformers import CLIPTextModel, CLIPTokenizer

        tokenizer = CLIPTokenizer.from_pretrained(model_id, subfolder="tokenizer")
        text_encoder = CLIPTextModel.from_pretrained(
            model_id,
            subfolder="text_encoder",
            torch_dtype=torch_dtype,
        )
    elif normalized == "t5":
        from transformers import T5EncoderModel, T5TokenizerFast

        tokenizer = T5TokenizerFast.from_pretrained(model_id, subfolder="tokenizer_2")
        text_encoder = T5EncoderModel.from_pretrained(
            model_id,
            subfolder="text_encoder_2",
            torch_dtype=torch_dtype,
        )
    else:
        raise ValueError(f"Unsupported graph text encoder type: {text_encoder_type}")

    text_encoder.requires_grad_(False)
    text_encoder.to(device)
    text_encoder.eval()
    return tokenizer, text_encoder, int(text_encoder.config.hidden_size)


def load_graph_encoder(
    *,
    path: Path,
    text_hidden_dim: int,
    device: str,
    dtype: torch.dtype,
) -> GraphSlotEncoder:
    """Load a saved graph encoder with its inferred architecture."""

    state_dict = normalize_graph_encoder_state_dict(torch.load(path, map_location="cpu"))
    (
        slot_dim,
        inferred_text_hidden_dim,
        gnn_layers,
        layout_mode,
        latent_dim,
        decoder_mode,
        decoder_box_residual,
    ) = infer_graph_encoder_config(state_dict)
    if text_hidden_dim != inferred_text_hidden_dim:
        raise ValueError(
            "Graph encoder text embedding dimension mismatch: "
            f"checkpoint expects {inferred_text_hidden_dim}, caller provided {text_hidden_dim}."
        )
    encoder = GraphSlotEncoder(
        text_hidden_dim=text_hidden_dim,
        slot_dim=slot_dim,
        num_layers=gnn_layers,
        layout_mode=layout_mode,
        latent_dim=latent_dim,
        decoder_mode=decoder_mode,
        decoder_box_residual=decoder_box_residual,
    ).to(device=device, dtype=dtype)
    encoder.load_state_dict(state_dict, strict=False)
    encoder.eval()
    return encoder
