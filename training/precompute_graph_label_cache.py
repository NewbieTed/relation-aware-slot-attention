"""Precompute frozen object-label embeddings for graph pretraining.

The GNN pretraining loop only needs text encoder outputs for a small fixed
vocabulary of object labels. This script fills the cache referenced by the
GNN YAML config so training can skip repeated T5/CLIP label encoding.
"""

from __future__ import annotations

from pathlib import Path

import torch
from tqdm.auto import tqdm

from .config import parse_args_with_config
from .dataset import load_metadata_rows
from .graph_modules import mean_pool_hidden
from .pretrain_graph_encoder import (
    _load_label_embedding_cache,
    _save_label_embedding_cache,
    make_parser,
)
from .prompts import scene_graph_payload_from_row
from .runtime import choose_weight_dtype, load_graph_label_encoder, resolve_torch_device


def _unique_node_labels(rows: list[dict[str, object]]) -> list[str]:
    labels: set[str] = set()
    for row in rows:
        graph = scene_graph_payload_from_row(row)
        for node in graph["nodes"]:
            labels.add(str(node["label"]))
    return sorted(labels)


def main() -> int:
    args = parse_args_with_config(make_parser(), section="gnn")
    if args.label_embedding_cache is None:
        raise ValueError("Set paths.label_embedding_cache in the GNN config before precomputing.")

    device = resolve_torch_device(args.device)
    dtype = choose_weight_dtype(device, args.mixed_precision)
    rows = load_metadata_rows(Path(args.dataset_dir))
    if args.limit_rows is not None:
        rows = rows[: args.limit_rows]
    labels = _unique_node_labels(rows)

    tokenizer, text_encoder, hidden_dim = load_graph_label_encoder(
        model_id=args.model_id,
        text_encoder_type=args.text_encoder_type,
        torch_dtype=dtype,
        device=device,
    )
    if args.text_hidden_dim is not None and args.text_hidden_dim != hidden_dim:
        raise ValueError(
            f"Config text_hidden_dim={args.text_hidden_dim} does not match "
            f"{args.text_encoder_type} encoder hidden size {hidden_dim}."
        )

    cache = _load_label_embedding_cache(
        args.label_embedding_cache,
        model_id=args.model_id,
        text_encoder_type=args.text_encoder_type,
        text_hidden_dim=hidden_dim,
    )
    missing_labels = [label for label in labels if label not in cache]
    batch_size = max(1, int(args.batch_size))

    for start in tqdm(range(0, len(missing_labels), batch_size), desc="PrecomputeLabelEmbeddings"):
        batch_labels = missing_labels[start : start + batch_size]
        text_inputs = tokenizer(batch_labels, padding=True, truncation=True, return_tensors="pt")
        with torch.no_grad():
            encoded = text_encoder(
                text_inputs.input_ids.to(device),
                attention_mask=text_inputs.attention_mask.to(device),
            )[0]
        pooled = mean_pool_hidden(encoded, text_inputs.attention_mask.to(device)).to(dtype=dtype)
        for label, embedding in zip(batch_labels, pooled):
            cache[label] = embedding.detach().cpu()

    _save_label_embedding_cache(
        args.label_embedding_cache,
        cache,
        model_id=args.model_id,
        text_encoder_type=args.text_encoder_type,
        text_hidden_dim=hidden_dim,
    )
    print(
        f"Saved {len(cache)} label embeddings "
        f"({len(missing_labels)} newly computed) to {args.label_embedding_cache}."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
