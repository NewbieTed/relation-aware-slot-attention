from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

BoundingBox = tuple[float, float, float, float]
RelationType = Literal[
    "left_of",
    "right_of",
    "above",
    "below",
    "on",
    "in_front_of",
    "behind",
    "hidden_by",
]


@dataclass(frozen=True)
class SceneNode:
    """A grounded object node extracted from one SCOP-Depth example."""

    id: str
    label: str
    annotation_id: int | None = None
    category_id: int | None = None
    bbox: BoundingBox | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SceneEdge:
    """A directed relation between two scene nodes."""

    source_id: str
    target_id: str
    relation: RelationType
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SceneGraph:
    """Minimal scene-graph container used by the future graph-conditioning code."""

    nodes: tuple[SceneNode, ...]
    edges: tuple[SceneEdge, ...]
    prompt: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def node_ids(self) -> set[str]:
        return {node.id for node in self.nodes}

    def num_nodes(self) -> int:
        return len(self.nodes)

    def num_edges(self) -> int:
        return len(self.edges)

    def validate(self) -> None:
        ids = self.node_ids()
        if len(ids) != len(self.nodes):
            raise ValueError("SceneGraph contains duplicate node ids")

        for edge in self.edges:
            if edge.source_id not in ids:
                raise ValueError(f"Unknown source node id: {edge.source_id}")
            if edge.target_id not in ids:
                raise ValueError(f"Unknown target node id: {edge.target_id}")

