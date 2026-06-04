"""Console summary tables for data pipeline artifacts."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from rich import box
from rich.console import Console
from rich.table import Table


def format_summary_value(value: Any) -> str:
    """Format values compactly for console summary tables."""
    if value is None:
        return ""
    if isinstance(value, float):
        if np.isnan(value):
            return "nan"
        if np.isinf(value):
            return str(value)
        if value == 0.0:
            return "0"
        if abs(value) >= 1_000_000 or abs(value) < 0.001:
            return f"{value:.3e}"
        return f"{value:.4f}".rstrip("0").rstrip(".")
    if isinstance(value, pd.Timestamp):
        return str(value.date())
    return str(value)


def date_bounds(index: pd.Index) -> tuple[str, str]:
    """Return compact date bounds for a dataframe index."""
    if not isinstance(index, pd.DatetimeIndex) or index.empty:
        return "", ""
    return str(index.min().date()), str(index.max().date())


def missing_summary(frame: pd.DataFrame | pd.Series) -> tuple[int, str]:
    """Summarize missing values in a pandas object."""
    values: pd.DataFrame = frame.to_frame() if isinstance(frame, pd.Series) else frame
    total_cells: int = int(values.size)
    if total_cells == 0:
        return 0, "0.00%"
    missing_cells: int = int(values.isna().sum().sum())
    missing_pct: float = missing_cells / total_cells * 100.0
    return missing_cells, f"{missing_pct:.2f}%"


def summarize_frame(name: str, frame: pd.DataFrame | pd.Series) -> dict[str, str]:
    """Build one compact row describing a tabular artifact."""
    table: pd.DataFrame = frame.to_frame() if isinstance(frame, pd.Series) else frame
    first_date, last_date = date_bounds(index=table.index)
    missing_cells, missing_pct = missing_summary(frame=table)
    return {
        "table": name,
        "rows": str(table.shape[0]),
        "cols": str(table.shape[1]),
        "first_date": first_date,
        "last_date": last_date,
        "missing": str(missing_cells),
        "missing_pct": missing_pct,
    }


def summarize_numeric_frame(name: str, frame: pd.DataFrame | pd.Series) -> dict[str, str]:
    """Build one compact numerical summary row for a table."""
    values: pd.DataFrame = frame.to_frame() if isinstance(frame, pd.Series) else frame
    numeric: pd.DataFrame = values.select_dtypes(include=[np.number])
    if numeric.empty:
        return {"table": name, "mean": "", "std": "", "min": "", "max": ""}
    flattened: np.ndarray = numeric.to_numpy(dtype=np.float64).reshape(-1)
    flattened = flattened[np.isfinite(flattened)]
    if flattened.size == 0:
        return {"table": name, "mean": "nan", "std": "nan", "min": "nan", "max": "nan"}
    return {
        "table": name,
        "mean": format_summary_value(float(np.mean(a=flattened))),
        "std": format_summary_value(float(np.std(a=flattened, ddof=1))),
        "min": format_summary_value(float(np.min(a=flattened))),
        "max": format_summary_value(float(np.max(a=flattened))),
    }


def summarize_array(name: str, array: np.ndarray) -> dict[str, str]:
    """Build one compact row describing a saved NumPy artifact."""
    values: np.ndarray = np.asarray(a=array)
    row: dict[str, str] = {
        "artifact": name,
        "shape": " x ".join(str(dim) for dim in values.shape),
        "dtype": str(values.dtype),
        "missing": "",
        "mean": "",
        "std": "",
        "min": "",
        "max": "",
    }
    if np.issubdtype(values.dtype, np.number):
        numeric_values: np.ndarray = values.astype(dtype=np.float64, copy=False)
        finite_mask: np.ndarray = np.isfinite(numeric_values)
        missing_count: int = int(values.size - finite_mask.sum())
        row["missing"] = str(missing_count)
        if np.any(finite_mask):
            finite_values: np.ndarray = numeric_values[finite_mask]
            row.update(
                {
                    "mean": format_summary_value(float(finite_values.mean())),
                    "std": format_summary_value(float(finite_values.std())),
                    "min": format_summary_value(float(finite_values.min())),
                    "max": format_summary_value(float(finite_values.max())),
                }
            )
    return row


def format_symbol_preview(symbols: pd.Index, limit: int = 6) -> str:
    """Format a compact preview for a symbol list."""
    symbol_list: list[str] = [str(symbol) for symbol in symbols]
    if len(symbol_list) <= limit:
        return ", ".join(symbol_list)
    visible: str = ", ".join(symbol_list[:limit])
    return f"{visible}, +{len(symbol_list) - limit} more"


def print_summary_table(title: str, rows: list[dict[str, str]]) -> None:
    """Print a boxed summary table with Rich."""
    if not rows:
        return
    console: Console = Console(width=180)
    columns: list[str] = list(rows[0])
    table: Table = Table(title=title, box=box.ROUNDED, show_lines=False)
    for column in columns:
        table.add_column(column, no_wrap=True, overflow="ellipsis")
    for row in rows:
        table.add_row(*(str(row.get(column, "")) for column in columns))
    console.print(table)
