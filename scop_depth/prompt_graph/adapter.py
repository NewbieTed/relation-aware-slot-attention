from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..models import CocoInstanceAnnotation
from .schema import SceneEdge, SceneGraph, SceneNode

DATASET_RELATION_TO_GRAPH = {
    "to the left of": "left_of",
    "to the right of": "right_of",
    "above": "above",
    "below": "below",
    "on": "on",
    "in front of": "in_front_of",
    "behind": "behind",
    "hidden by": "hidden_by",
}


@dataclass(frozen=True)
class SCOPDepthExample:
    """Convenience wrapper around one exported metadata row."""

    seq: int
    file_name: str
    oros: list[list[str]]
    annots: tuple[CocoInstanceAnnotation, CocoInstanceAnnotation]
    depth: dict[str, Any] | None = None

    @classmethod
    def from_row(cls, row: dict[str, Any]) -> SCOPDepthExample:
        annots = tuple(CocoInstanceAnnotation.from_dict(a) for a in row["annots"])
        if len(annots) != 2:
            raise ValueError("SCOPDepthExample expects exactly two annotations")
        return cls(
            seq=row["seq"],
            file_name=row["file_name"],
            oros=row["oros"],
            annots=annots,
            depth=row.get("depth"),
        )


def _normalize_relation(raw_relation: str) -> str:
    try:
        return DATASET_RELATION_TO_GRAPH[raw_relation]
    except KeyError as exc:
        raise ValueError(f"Unsupported relation string: {raw_relation}") from exc


def _make_node(index: int, label: str, annot: CocoInstanceAnnotation) -> SceneNode:
    return SceneNode(
        id=f"obj{index}",
        label=label,
        annotation_id=annot.id,
        category_id=annot.category_id,
        bbox=annot.bbox,
        metadata={
            "image_id": annot.image_id,
            "area": annot.area,
            "iscrowd": annot.iscrowd,
        },
    )


def scene_graph_from_scop_depth_row(row: dict[str, Any]) -> SceneGraph:
    """Convert one SCOP-Depth metadata row into a validated two-node scene graph."""

    example = SCOPDepthExample.from_row(row)
    if len(example.oros) == 0:
        raise ValueError("SCOP-Depth row has no relations to convert")

    node_labels_in_order: list[str] = []
    for subj, _, obj in example.oros:
        if subj not in node_labels_in_order:
            node_labels_in_order.append(subj)
        if obj not in node_labels_in_order:
            node_labels_in_order.append(obj)

    if len(node_labels_in_order) != 2:
        raise ValueError(
            "SCOP-Depth adapter currently expects exactly two unique object labels"
        )

    label_to_node_id = {
        node_labels_in_order[0]: "obj0",
        node_labels_in_order[1]: "obj1",
    }

    nodes = (
        _make_node(0, node_labels_in_order[0], example.annots[0]),
        _make_node(1, node_labels_in_order[1], example.annots[1]),
    )

    edges = tuple(
        SceneEdge(
            source_id=label_to_node_id[subj],
            target_id=label_to_node_id[obj],
            relation=_normalize_relation(rel),  # type: ignore[arg-type]
            metadata={"raw_relation": rel},
        )
        for subj, rel, obj in example.oros
    )

    graph = SceneGraph(
        nodes=nodes,
        edges=edges,
        prompt=" ".join(f"{s} {r} {o}" for s, r, o in example.oros),
        metadata={
            "seq": example.seq,
            "file_name": example.file_name,
            "depth": example.depth,
        },
    )
    graph.validate()
    return graph

