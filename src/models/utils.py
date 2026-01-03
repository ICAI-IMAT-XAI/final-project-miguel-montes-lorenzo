import json
import shutil
from collections.abc import Iterable
from pathlib import Path
from typing import Any


def save_results(results: dict[str, Any], store_path: Path) -> None:
    """Save results dictionary to a JSON file at <store_path>/results.json.

    Args:
        results: Dictionary containing serializable results.
        store_path: Directory where the results file will be stored.
    """
    assert isinstance(results, dict)
    assert isinstance(store_path, Path)

    store_path.mkdir(parents=True, exist_ok=True)

    output_path: Path = store_path / "results.json"

    with output_path.open(mode="w", encoding="utf-8") as f:
        json.dump(
            obj=results,
            fp=f,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
        )

    return None


def clean_model_saves(model_path: Path) -> None:
    """Remove directories under a model path that do not contain a `.keep` file.

    Any directory (at any depth) inside `model_path` is removed if it does not
    contain a `.keep` file directly within it.

    Args:
        model_path: Root directory containing model save subdirectories.
    """
    assert isinstance(model_path, Path)
    assert model_path.is_dir()

    directories: Iterable[Path] = (
        p for p in model_path.rglob(pattern="*") if p.is_dir()
    )

    for directory in directories:
        keep_file: Path = directory / ".keep"

        if not keep_file.exists():
            shutil.rmtree(directory)

    return None
