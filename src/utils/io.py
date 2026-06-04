"""Small filesystem helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def ensure_dir(path: str | Path) -> Path:
    """Create a directory if it does not exist.

    Args:
        path: Directory path.

    Returns:
        Directory path as a ``Path`` object.
    """
    directory: Path = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def write_json(data: dict[str, Any], path: str | Path) -> None:
    """Write a dictionary to JSON with stable formatting.

    Args:
        data: Dictionary to serialize.
        path: Output JSON path.
    """
    output_path: Path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open(mode="w", encoding="utf-8") as file:
        json.dump(obj=data, fp=file, indent=2, sort_keys=True)


def read_json(path: str | Path) -> dict[str, Any]:
    """Read a JSON file as a dictionary.

    Args:
        path: Input JSON path.

    Returns:
        Parsed JSON dictionary.
    """
    with Path(path).open(mode="r", encoding="utf-8") as file:
        data: Any = json.load(fp=file)
    if not isinstance(data, dict):
        raise TypeError(f"Expected a JSON object in {path}.")
    return data
