from __future__ import annotations

from typing import Any

from scop_depth.prompt_graph import scene_graph_from_scop_depth_row

_RELATION_PRIORITY = {
    "hidden by": 0,
    "in front of": 1,
    "behind": 2,
    "on": 3,
    "to the left of": 4,
    "to the right of": 5,
    "above": 6,
    "below": 7,
}


def _article_for(noun_phrase: str) -> str:
    if not noun_phrase:
        return "a"
    return "an" if noun_phrase[0].lower() in {"a", "e", "i", "o", "u"} else "a"


def choose_primary_relation(oros: list[list[str]]) -> tuple[str, str, str]:
    """Choose one canonical relation triplet for baseline prompt synthesis."""

    if not oros:
        raise ValueError("Cannot build a prompt from an empty relation list")

    ranked = sorted(
        (tuple(triplet) for triplet in oros),
        key=lambda triplet: (
            _RELATION_PRIORITY.get(triplet[1], 999),
            str(triplet[0]),
            str(triplet[2]),
        ),
    )
    subject, relation, obj = ranked[0]
    return str(subject), str(relation), str(obj)


def prompt_from_scop_depth_row(row: dict[str, Any], prefix: str = "a photo of") -> str:
    """Turn one SCOP-Depth row into a concise text prompt for baseline training."""

    subject, relation, obj = choose_primary_relation(row["oros"])
    subject_phrase = f"{_article_for(subject)} {subject}"
    object_phrase = f"{_article_for(obj)} {obj}"
    prompt = f"{prefix} {subject_phrase} {relation} {object_phrase}"
    return " ".join(prompt.split())


def scene_graph_payload_from_row(row: dict[str, Any]) -> dict[str, Any]:
    """Serialize the validated two-node graph into a training-friendly dict."""

    graph = scene_graph_from_scop_depth_row(row)
    return {
        "prompt": graph.prompt,
        "nodes": [
            {
                "id": node.id,
                "label": node.label,
                "annotation_id": node.annotation_id,
                "category_id": node.category_id,
                "bbox": list(node.bbox) if node.bbox is not None else None,
                "metadata": dict(node.metadata),
            }
            for node in graph.nodes
        ],
        "edges": [
            {
                "source_id": edge.source_id,
                "target_id": edge.target_id,
                "relation": edge.relation,
                "metadata": dict(edge.metadata),
            }
            for edge in graph.edges
        ],
        "metadata": dict(graph.metadata),
    }

