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

INVERSE_RELATION = {
    "left_of": "right_of",
    "right_of": "left_of",
    "above": "below",
    "below": "above",
    "on": "below",
    "in_front_of": "behind",
    "behind": "in_front_of",
    "hidden_by": "in_front_of",
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


def _message_passing_triplets(
    triplets: list[tuple[int, int, str]],
) -> list[tuple[int, int, str]]:
    expanded: list[tuple[int, int, str]] = []
    seen: set[tuple[int, int, str]] = set()
    for src, dst, relation in triplets:
        for triplet in (
            (src, dst, relation),
            (dst, src, INVERSE_RELATION[relation]),
        ):
            if triplet not in seen:
                expanded.append(triplet)
                seen.add(triplet)
    return expanded


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
        message_passing_triplets = _message_passing_triplets(triplets)
        if message_passing_triplets:
            edge_index.append(
                torch.tensor(
                    [[src, dst] for src, dst, _ in message_passing_triplets],
                    dtype=torch.long,
                )
            )
            edge_types.append(
                torch.tensor(
                    [RELATION_VOCAB[relation] for _, _, relation in message_passing_triplets],
                    dtype=torch.long,
                )
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
