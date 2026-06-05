"""Filesystem helpers for XAI artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def ensure_xai_dirs(output_dir: str | Path) -> dict[str, Path]:
    """Create and return the required XAI output directories."""
    root = Path(output_dir)
    paths = {
        "root": root,
        "global": root / "global_explanations",
        "local": root / "local_explanations",
        "evaluation": root / "evaluation",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def write_json(data: dict[str, Any], path: str | Path) -> Path:
    """Write a JSON object with stable formatting."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2, sort_keys=True)
    return output_path


def manifest_entry(path: str | Path, kind: str, method: str) -> dict[str, str]:
    """Build a manifest entry for one generated artifact."""
    artifact_path = Path(path)
    return {
        "path": str(artifact_path),
        "kind": kind,
        "method": method,
    }


def write_manifest(entries: list[dict[str, str]], output_dir: str | Path) -> Path:
    """Write the XAI outputs manifest."""
    return write_json(
        {"artifacts": entries, "artifact_count": len(entries)},
        Path(output_dir) / "xai_outputs_manifest.json",
    )

