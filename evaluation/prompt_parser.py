from __future__ import annotations

import re

RELATION_PATTERN_SPECS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("to the left of and above", ("left_of", "above")),
    ("to the left of and below", ("left_of", "below")),
    ("to the right of and above", ("right_of", "above")),
    ("to the right of and below", ("right_of", "below")),
    ("above and to the left of", ("above", "left_of")),
    ("above and to the right of", ("above", "right_of")),
    ("below and to the left of", ("below", "left_of")),
    ("below and to the right of", ("below", "right_of")),
    ("hidden by", ("hidden_by",)),
    ("in front of", ("in_front_of",)),
    ("next to", ("next_to",)),
    ("near", ("next_to",)),
    ("beside", ("next_to",)),
    ("on side of", ("next_to",)),
    ("side of", ("next_to",)),
    ("on the left of", ("left_of",)),
    ("to the left of", ("left_of",)),
    ("left of", ("left_of",)),
    ("on the right of", ("right_of",)),
    ("to the right of", ("right_of",)),
    ("right of", ("right_of",)),
    ("on the top of", ("above",)),
    ("on top of", ("on",)),
    ("above", ("above",)),
    ("on the bottom of", ("below",)),
    ("on bottom of", ("below",)),
    ("below", ("below",)),
    ("under", ("below",)),
    ("behind", ("behind",)),
    ("on", ("on",)),
)

RELATION_PATTERNS: tuple[tuple[re.Pattern[str], str, tuple[str, ...]], ...] = tuple(
    (
        re.compile(rf"\b{re.escape(phrase)}\b", re.IGNORECASE),
        phrase,
        relations,
    )
    for phrase, relations in RELATION_PATTERN_SPECS
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
    relation_names: tuple[str, ...] | None = None
    matched_phrase: str | None = None

    for pattern, phrase, relations in RELATION_PATTERNS:
        match = pattern.search(lowered)
        if match is None:
            continue
        span = match.span()
        if match_span is None or span[0] < match_span[0] or (
            span[0] == match_span[0] and len(phrase) > len(matched_phrase or "")
        ):
            match_span = span
            relation_names = relations
            matched_phrase = phrase

    if match_span is None or relation_names is None or matched_phrase is None:
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
            for relation_name in relation_names
        ],
        "metadata": {
            "parser": "rule_based",
        },
    }
