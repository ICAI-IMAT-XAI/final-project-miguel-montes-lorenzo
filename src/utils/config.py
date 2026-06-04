"""Configuration loading helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Load a YAML file as a dictionary.

    Args:
        path: Path to a YAML configuration file.

    Returns:
        Parsed configuration dictionary.
    """
    with Path(path).open(mode="r", encoding="utf-8") as file:
        data: Any = yaml.safe_load(stream=file)
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise TypeError(f"Expected a mapping in {path}, got {type(data).__name__}.")
    return data


def merge_dicts(*configs: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge dictionaries from left to right.

    Args:
        *configs: Configuration dictionaries ordered from lowest to highest
            priority.

    Returns:
        Merged dictionary.
    """
    merged: dict[str, Any] = {}
    for config in configs:
        for key, value in config.items():
            if (
                key in merged
                and isinstance(merged[key], dict)
                and isinstance(value, dict)
            ):
                merged[key] = merge_dicts(merged[key], value)
            else:
                merged[key] = value
    return merged
