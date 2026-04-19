from __future__ import annotations

import re

RELATION_PATTERNS: tuple[tuple[re.Pattern[str], str, str], ...] = (
    ("hidden by", "hidden_by"),
    ("in front of", "in_front_of"),
    ("to the left of", "left_of"),
    ("left of", "left_of"),
    ("to the right of", "right_of"),
    ("right of", "right_of"),
    ("on top of", "on"),
    ("above", "above"),
    ("below", "below"),
    ("under", "below"),
    ("behind", "behind"),
    ("on", "on"),
)

RELATION_PATTERNS = tuple(
    (
        re.compile(rf"\b{re.escape(phrase)}\b", re.IGNORECASE),
        phrase,
        relation,
    )
    for phrase, relation in RELATION_PATTERNS
)

PROMPT_PREFIX_RE = re.compile(
    r"^(?:a|an|the)\s+(?:photo|picture|image|rendering|painting|drawing|illustration)\s+of\s+",
    re.IGNORECASE,
)
LEADING_ARTICLE_RE = re.compile(r"^(?:a|an|the)\s+", re.IGNORECASE)
TRAILING_PUNCT_RE = re.compile(r"[.,;:!?]+$")


def _clean_phrase(text: str) -> str:
    cleaned = PROMPT_PREFIX_RE.sub("", text.strip())
    cleaned = TRAILING_PUNCT_RE.sub("", cleaned)
    cleaned = cleaned.strip()
    cleaned = LEADING_ARTICLE_RE.sub("", cleaned)
    return " ".join(cleaned.split())


def parse_prompt_to_scene_graph(prompt: str) -> dict[str, object]:
    lowered = prompt.lower().strip()
    match_span: tuple[int, int] | None = None
    relation_name: str | None = None
    matched_phrase: str | None = None

    for pattern, phrase, relation in RELATION_PATTERNS:
        match = pattern.search(lowered)
        if match is None:
            continue
        span = match.span()
        if match_span is None or span[0] < match_span[0] or (
            span[0] == match_span[0] and len(phrase) > len(matched_phrase or "")
        ):
            match_span = span
            relation_name = relation
            matched_phrase = phrase

    if match_span is None or relation_name is None or matched_phrase is None:
        raise ValueError(f"Could not parse a supported spatial relation from prompt: {prompt}")

    left = _clean_phrase(prompt[: match_span[0]])
    right = _clean_phrase(prompt[match_span[1] :])
    if not left or not right:
        raise ValueError(f"Could not parse subject/object phrases from prompt: {prompt}")

    return {
        "prompt": prompt,
        "nodes": [
            {"id": "obj0", "label": left},
            {"id": "obj1", "label": right},
        ],
        "edges": [
            {
                "source_id": "obj0",
                "target_id": "obj1",
                "relation": relation_name,
                "metadata": {
                    "raw_relation": matched_phrase,
                    "parser": "rule_based",
                },
            }
        ],
        "metadata": {
            "parser": "rule_based",
        },
    }
