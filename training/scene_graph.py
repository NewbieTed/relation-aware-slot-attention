from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

RELATION_VOCAB = {
    "left_of": 0,
    "right_of": 1,
    "above": 2,
    "below": 3,
    "on": 4,
    "in_front_of": 5,
    "behind": 6,
    "hidden_by": 7,
}


@dataclass(frozen=True)
class BatchedSceneGraphs:
    node_labels: list[list[str]]
    node_ids: list[list[str]]
    edge_index: list[torch.Tensor]
    edge_types: list[torch.Tensor]
    position_targets: torch.Tensor
    position_mask: torch.Tensor
    relation_triplets: list[list[tuple[int, int, str]]]


def _relation_triplets(scene_graph: dict[str, Any]) -> list[tuple[int, int, str]]:
    node_id_to_index = {node["id"]: i for i, node in enumerate(scene_graph["nodes"])}
    triplets: list[tuple[int, int, str]] = []
    for edge in scene_graph["edges"]:
        triplets.append(
            (
                node_id_to_index[edge["source_id"]],
                node_id_to_index[edge["target_id"]],
                edge["relation"],
            )
        )
    return triplets


def build_batched_scene_graphs(
    scene_graphs: list[dict[str, Any]],
    slot_targets: torch.Tensor,
    slot_mask: torch.Tensor,
) -> BatchedSceneGraphs:
    edge_index: list[torch.Tensor] = []
    edge_types: list[torch.Tensor] = []
    node_labels: list[list[str]] = []
    node_ids: list[list[str]] = []
    relation_triplets: list[list[tuple[int, int, str]]] = []

    for scene_graph in scene_graphs:
        labels = [node["label"] for node in scene_graph["nodes"]]
        ids = [node["id"] for node in scene_graph["nodes"]]
        node_labels.append(labels)
        node_ids.append(ids)

        triplets = _relation_triplets(scene_graph)
        relation_triplets.append(triplets)
        if triplets:
            edge_index.append(
                torch.tensor([[src, dst] for src, dst, _ in triplets], dtype=torch.long)
            )
            edge_types.append(
                torch.tensor([RELATION_VOCAB[relation] for _, _, relation in triplets], dtype=torch.long)
            )
        else:
            edge_index.append(torch.zeros((0, 2), dtype=torch.long))
            edge_types.append(torch.zeros((0,), dtype=torch.long))

    return BatchedSceneGraphs(
        node_labels=node_labels,
        node_ids=node_ids,
        edge_index=edge_index,
        edge_types=edge_types,
        position_targets=slot_targets,
        position_mask=slot_mask,
        relation_triplets=relation_triplets,
    )

