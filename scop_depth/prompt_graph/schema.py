from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


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


@dataclass(frozen=True, slots=True)
class SceneNode:
    """Object-centric node in a prompt scene graph."""

    id: int
    text: str
    head: str
    attributes: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class SceneEdge:
    """Directed relation between two scene-graph nodes."""

    source: int
    target: int
    relation: RelationType


@dataclass(frozen=True, slots=True)
class SceneGraph:
    """Prompt-derived scene graph used by the slot-conditioning path."""

    nodes: tuple[SceneNode, ...]
    edges: tuple[SceneEdge, ...]
    prompt: str | None = None
    metadata: dict[str, str] = field(default_factory=dict)

    def node_ids(self) -> tuple[int, ...]:
        return tuple(node.id for node in self.nodes)

    def validate(self) -> None:
        """
        Validate graph integrity.

        Raises:
            ValueError: if node IDs are duplicated or an edge references a
                non-existent node.
        """
        node_ids = self.node_ids()
        node_id_set = set(node_ids)

        if len(node_ids) != len(node_id_set):
            raise ValueError("SceneGraph contains duplicate node IDs")

        for edge in self.edges:
            if edge.source not in node_id_set:
                raise ValueError(f"Edge source {edge.source} is not a valid node ID")
            if edge.target not in node_id_set:
                raise ValueError(f"Edge target {edge.target} is not a valid node ID")

    def num_nodes(self) -> int:
        return len(self.nodes)

    def num_edges(self) -> int:
        return len(self.edges)
