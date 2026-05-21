from __future__ import annotations

from dataclasses import dataclass
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
TRAILING_ADDON_RE = re.compile(
    r"(?P<object>.+?)"
    r"(?P<separator>"
    r"\s*[,;:]\s+"
    r"|\s+\b(?:in|inside|within|at|against|with|without|featuring|surrounded by|on)\b\s+"
    r")"
    r"(?P<addon>.+)$",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class PromptAdditions:
    """Generation-only text that should not change the parsed relation graph."""

    scene_prefix: str = ""
    background: str = ""
    style: str = ""
    quality: str = ""
    suffix: str = ""


def _clean_phrase(text: str) -> str:
    cleaned = PROMPT_PREFIX_RE.sub("", text.strip())
    cleaned = TRAILING_PUNCT_RE.sub("", cleaned)
    cleaned = cleaned.strip()
    cleaned = LEADING_ARTICLE_RE.sub("", cleaned)
    return " ".join(cleaned.split())


def _strip_trailing_addons(text: str) -> tuple[str, str]:
    """Split an object phrase from trailing scene/style text when present.

    Prompt files sometimes carry useful generation text after the relation,
    for example ``a dog left of a chair, in a kitchen``. The graph encoder only
    wants the object label, while the full text can still be used by FLUX.
    """

    text = text.strip()
    match = TRAILING_ADDON_RE.match(text)
    if match is None:
        return text, ""
    object_text = match.group("object").strip()
    separator = match.group("separator").strip()
    addon_text = match.group("addon").strip()
    if not object_text or not addon_text:
        return text, ""
    if separator and separator[0] not in ",;:":
        addon_text = f"{separator} {addon_text}"
    return object_text, addon_text


def _normalize_addition(text: str | None) -> str:
    if text is None:
        return ""
    return " ".join(str(text).strip(" ,;").split())


def _contains_normalized(text: str, phrase: str) -> bool:
    if not phrase:
        return True
    return _normalize_addition(phrase).lower() in _normalize_addition(text).lower()


def prompt_additions_from_args(args: object) -> PromptAdditions:
    """Read optional generation add-ons from an argparse namespace."""

    return PromptAdditions(
        scene_prefix=_normalize_addition(getattr(args, "generation_scene_prefix", "")),
        background=_normalize_addition(getattr(args, "background_prompt", "")),
        style=_normalize_addition(getattr(args, "style_prompt", "")),
        quality=_normalize_addition(getattr(args, "quality_prompt", "")),
        suffix=_normalize_addition(getattr(args, "generation_prompt_suffix", "")),
    )


def compose_generation_prompt(base_prompt: str, additions: PromptAdditions | None = None) -> str:
    """Compose the final text prompt with scene context front-loaded.

    ``scene_prefix`` is meant to be part of the main prompt rather than a weak
    tail add-on. Some callers already bake it into ``base_prompt`` to preserve
    token-binding offsets, so avoid duplicating it in that case.
    """

    additions = additions or PromptAdditions()
    base_prompt = _normalize_addition(base_prompt)
    leading_scene = "" if _contains_normalized(base_prompt, additions.scene_prefix) else additions.scene_prefix
    parts = [
        leading_scene,
        base_prompt,
        additions.background,
        additions.style,
        additions.quality,
        additions.suffix,
    ]
    return ", ".join(part for part in parts if part)


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
    right_raw, trailing_addon = _strip_trailing_addons(prompt[match_span[1] :])
    right = _clean_phrase(right_raw)
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
            "core_prompt": " ".join(f"{prompt[: match_span[1]]} {right_raw}".split()),
            "trailing_generation_addon": _normalize_addition(trailing_addon),
        },
    }
