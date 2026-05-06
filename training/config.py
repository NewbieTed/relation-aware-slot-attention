from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _load_raw_config(path: Path) -> dict[str, Any]:
    suffix = path.suffix.lower()
    if suffix == ".json":
        return json.loads(path.read_text())
    if suffix in {".yaml", ".yml"}:
        try:
            import yaml
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "YAML config files require PyYAML. Use JSON or install pyyaml."
            ) from exc
        loaded = yaml.safe_load(path.read_text())
        return loaded or {}
    raise ValueError(f"Unsupported config file extension: {path}. Use .json, .yaml, or .yml.")


def _flatten_config(config: dict[str, Any]) -> dict[str, Any]:
    flattened: dict[str, Any] = {}
    for key, value in config.items():
        normalized_key = str(key).replace("-", "_")
        if isinstance(value, dict):
            for nested_key, nested_value in _flatten_config(value).items():
                if nested_key in flattened:
                    raise ValueError(f"Duplicate config key after flattening: {nested_key}")
                flattened[nested_key] = nested_value
        else:
            if normalized_key in flattened:
                raise ValueError(f"Duplicate config key after flattening: {normalized_key}")
            flattened[normalized_key] = value
    return flattened


def _section_config(raw: dict[str, Any], section: str) -> dict[str, Any]:
    config: dict[str, Any] = {}
    common = raw.get("common")
    if isinstance(common, dict):
        config.update(common)
    selected = raw.get(section)
    if isinstance(selected, dict):
        config.update(selected)
    if not config:
        config = raw
    return _flatten_config(config)


def _coerce_value(action: argparse.Action, value: Any) -> Any:
    if value is None:
        return None
    if isinstance(action, argparse._StoreTrueAction):
        return bool(value)
    if isinstance(action, argparse._StoreFalseAction):
        return bool(value)
    if action.type is Path and not isinstance(value, Path):
        return Path(value)
    if action.type is not None and action.type is not Path:
        try:
            return action.type(value)
        except (TypeError, ValueError):
            return value
    return value


def parse_args_with_config(
    parser: argparse.ArgumentParser,
    *,
    section: str,
    args: list[str] | None = None,
) -> argparse.Namespace:
    """Parse either a complete config file or normal argparse flags.

    Config mode is intentionally minimal: when ``--config`` is supplied, the
    config is the source of truth and no extra CLI overrides are accepted.
    """

    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="JSON/YAML config file. Values use argparse destination names.",
    )
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument("--config", type=Path, default=None)
    known, remaining = config_parser.parse_known_args(args)
    if known.config is None:
        return parser.parse_args(args)
    if remaining:
        raise ValueError("--config mode does not accept additional CLI overrides")

    raw_config = _load_raw_config(known.config)
    config = _section_config(raw_config, section)
    actions = {action.dest: action for action in parser._actions}
    unknown_keys = sorted(key for key in config if key not in actions)
    if unknown_keys:
        raise ValueError(f"Unknown keys in {known.config} for section {section!r}: {unknown_keys}")

    values = {action.dest: action.default for action in parser._actions}
    values.update({key: _coerce_value(actions[key], value) for key, value in config.items()})
    values["config"] = known.config

    missing = [
        action.dest
        for action in parser._actions
        if getattr(action, "required", False)
        and values.get(action.dest) in (None, argparse.SUPPRESS)
    ]
    if missing:
        raise ValueError(f"Missing required config keys in {known.config}: {missing}")
    return argparse.Namespace(**values)
