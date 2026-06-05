"""Transform raw prices into supervised FOREX allocation tensors."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.data.symbols import FX_USD_PAIRS, TARGET_CURRENCIES
from src.data.summary import (
    date_bounds,
    format_symbol_preview,
    print_summary_table,
    summarize_array,
    summarize_frame,
    summarize_numeric_frame,
)
from src.utils import ensure_dir, load_yaml, write_json


def build_node_summary_rows(
    node_returns: dict[str, pd.DataFrame],
    arrays: dict[str, np.ndarray],
) -> list[dict[str, str]]:
    """Summarize heterogeneous node blocks after filtering."""
    rows: list[dict[str, str]] = []
    portfolio_shape: str = " x ".join(
        str(dim) for dim in arrays["portfolio_features"].shape
    )
    rows.append(
        {
            "node": "portfolio_signal",
            "symbol_count": "0",
            "symbols": "-",
            "rows": str(arrays["portfolio_features"].shape[0]),
            "features": str(arrays["portfolio_features"].shape[-1]),
            "first_date": "",
            "last_date": "",
            "tensor_shape": portfolio_shape,
        }
    )
    for node_name in sorted(node_returns):
        block: pd.DataFrame = node_returns[node_name]
        first_date, last_date = date_bounds(index=block.index)
        tensor: np.ndarray = arrays[f"node::{node_name}"]
        tensor_shape: str = " x ".join(str(dim) for dim in tensor.shape)
        rows.append(
            {
                "node": node_name,
                "symbol_count": str(block.shape[1]),
                "symbols": format_symbol_preview(symbols=block.columns),
                "rows": str(block.shape[0]),
                "features": str(block.shape[1]),
                "first_date": first_date,
                "last_date": last_date,
                "tensor_shape": tensor_shape,
            }
        )
    return rows


def format_missing_pct(frame: pd.DataFrame) -> str:
    """Return a compact missing percentage for a dataframe."""
    if frame.empty or frame.size == 0:
        return ""
    return f"{float(frame.isna().mean().mean() * 100.0):.2f}%"


def load_symbol_metadata(path: str | Path) -> pd.DataFrame:
    """Load raw symbol metadata written by the downloader."""
    input_path: Path = Path(path)
    if input_path.suffix == ".parquet":
        return pd.read_parquet(path=input_path)
    return pd.read_csv(filepath_or_buffer=input_path)


def symbol_blocks_from_metadata(metadata: pd.DataFrame) -> dict[str, list[str]]:
    """Build configured symbol blocks from downloader metadata."""
    required_columns: set[str] = {"category", "symbol"}
    missing_columns: set[str] = required_columns.difference(metadata.columns)
    if missing_columns:
        raise KeyError(f"Raw metadata missing columns: {sorted(missing_columns)}.")

    blocks: dict[str, list[str]] = {}
    for category, block in metadata.groupby(by="category", sort=False):
        blocks[str(category)] = [str(symbol) for symbol in block["symbol"].tolist()]
    return blocks


def build_node_filter_row(
    category: str,
    configured_symbols: list[str],
    available_symbols: list[str],
    valid_symbols: list[str],
    removed_symbols: list[str],
    block: pd.DataFrame,
    filtered_block: pd.DataFrame,
    ffilled_block: pd.DataFrame,
) -> dict[str, str]:
    """Build one row summarizing transform-time symbol filtering."""
    return {
        "node": category,
        "configured": str(len(configured_symbols)),
        "available": str(len(available_symbols)),
        "kept": str(len(valid_symbols)),
        "removed": str(len(removed_symbols)),
        "removed_symbols": format_symbol_preview(symbols=pd.Index(data=removed_symbols)),
        "missing_before": format_missing_pct(frame=block),
        "missing_after_filter": format_missing_pct(frame=filtered_block),
        "missing_after_ffill": format_missing_pct(frame=ffilled_block),
    }


def build_split_summary_rows(metadata: dict[str, Any]) -> list[dict[str, str]]:
    """Summarize train/validation/test split bounds."""
    rows: list[dict[str, str]] = []
    for split_name, bounds in metadata["splits"].items():
        start, end = bounds
        row: dict[str, str] = {
            "split": split_name,
            "start": str(start),
            "end": str(end),
            "samples": str(end - start),
        }
        split_dates: dict[str, list[str]] = metadata.get("split_dates", {})
        if split_name in split_dates:
            date_start, date_end = split_dates[split_name]
            row["requested_start"] = date_start
            row["requested_end"] = date_end
        split_actual_dates: dict[str, list[str]] = metadata.get("split_actual_dates", {})
        if split_name in split_actual_dates:
            actual_start, actual_end = split_actual_dates[split_name]
            row["sample_start"] = actual_start
            row["sample_end"] = actual_end
        rows.append(row)
    return rows


def log_transform_summaries(
    prices: pd.DataFrame,
    log_returns: pd.DataFrame,
    currency_returns: pd.DataFrame,
    node_returns: dict[str, pd.DataFrame],
    arrays: dict[str, np.ndarray],
    metadata: dict[str, Any],
    node_filter_rows: list[dict[str, str]],
) -> None:
    """Print compact human-readable summaries for transform artifacts."""
    print_summary_table(
        title="Input and Intermediate Tables",
        rows=[
            summarize_frame(name="raw_prices", frame=prices),
            summarize_frame(name="log_returns", frame=log_returns),
            summarize_frame(name="currency_returns", frame=currency_returns),
        ],
    )
    print_summary_table(
        title="Numeric Ranges",
        rows=[
            summarize_numeric_frame(name="raw_prices", frame=prices),
            summarize_numeric_frame(name="log_returns", frame=log_returns),
            summarize_numeric_frame(name="currency_returns", frame=currency_returns),
        ],
    )
    print_summary_table(
        title="Node Filtering Summary",
        rows=node_filter_rows,
    )
    print_summary_table(
        title="Heterogeneous Nodes",
        rows=build_node_summary_rows(node_returns=node_returns, arrays=arrays),
    )
    output_rows: list[dict[str, str]] = [
        summarize_array(name=key, array=value)
        for key, value in arrays.items()
        if not key.startswith("node::")
    ]
    output_rows.extend(
        summarize_array(name=f"nodes/{key.split('::', maxsplit=1)[1]}", array=value)
        for key, value in sorted(arrays.items())
        if key.startswith("node::")
    )
    print_summary_table(title="Saved NumPy Artifacts", rows=output_rows)
    print_summary_table(
        title="Dataset Splits",
        rows=build_split_summary_rows(metadata=metadata),
    )
    print(
        "\nSaved metadata: "
        f"{metadata['num_samples']} samples, "
        f"{len(metadata['node_names'])} nodes, "
        f"{len(metadata['currencies'])} currencies."
    )


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the transformer.

    Returns:
        Parsed command-line namespace.
    """
    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        description="Create supervised tensors for FOREX portfolio allocation."
    )
    parser.add_argument(
        "--config",
        default="configs/transform.yaml",
        help="Path to the transform YAML configuration.",
    )
    return parser.parse_args()


def project_to_simplex(values: np.ndarray) -> np.ndarray:
    """Project a vector onto the probability simplex.

    Args:
        values: Raw score vector.

    Returns:
        Non-negative vector summing to one.
    """
    vector: np.ndarray = np.asarray(a=values, dtype=np.float64)
    if vector.ndim != 1:
        raise ValueError("The simplex projection expects a one-dimensional vector.")
    if vector.size == 0:
        raise ValueError("Cannot project an empty vector.")
    sorted_values: np.ndarray = np.sort(a=vector)[::-1]
    cumsum: np.ndarray = np.cumsum(a=sorted_values)
    rho_candidates: np.ndarray = sorted_values * np.arange(1, vector.size + 1) > (
        cumsum - 1.0
    )
    if not np.any(a=rho_candidates):
        return np.full(shape=vector.shape, fill_value=1.0 / vector.size)
    rho: int = int(np.flatnonzero(rho_candidates)[-1])
    theta: float = float((cumsum[rho] - 1.0) / (rho + 1))
    projected: np.ndarray = np.maximum(vector - theta, 0.0)
    total: float = float(projected.sum())
    if total <= 0.0 or not np.isfinite(total):
        return np.full(shape=vector.shape, fill_value=1.0 / vector.size)
    return projected / total


def compute_markowitz_weights(
    returns: np.ndarray,
    risk_aversion: float,
    ridge: float,
    allow_short: bool,
) -> tuple[np.ndarray, float]:
    """Compute a stable rolling Markowitz allocation target.

    Args:
        returns: Historical log-return matrix with shape ``(T, N)``.
        risk_aversion: Quadratic utility risk-aversion coefficient.
        ridge: Diagonal covariance regularization.
        allow_short: Whether negative weights are allowed.

    Returns:
        Pair containing portfolio weights and portfolio variance.
    """
    n_assets: int = returns.shape[1]
    if returns.shape[0] < 2 or not np.all(np.isfinite(returns)):
        weights: np.ndarray = np.full(shape=n_assets, fill_value=1.0 / n_assets)
        return weights, 0.0

    mu: np.ndarray = np.mean(a=returns, axis=0)
    covariance: np.ndarray = np.cov(m=returns, rowvar=False)
    covariance = np.atleast_2d(covariance) + ridge * np.eye(N=n_assets)
    try:
        raw_weights: np.ndarray = np.linalg.solve(
            a=risk_aversion * covariance,
            b=mu,
        )
    except np.linalg.LinAlgError:
        raw_weights = np.linalg.pinv(a=risk_aversion * covariance) @ mu

    if allow_short:
        denominator: float = float(np.sum(raw_weights))
        if abs(denominator) < 1.0e-12 or not np.isfinite(denominator):
            weights = np.full(shape=n_assets, fill_value=1.0 / n_assets)
        else:
            weights = raw_weights / denominator
    else:
        weights = project_to_simplex(values=raw_weights)

    variance: float = float(weights.T @ covariance @ weights)
    return weights.astype(np.float64), max(variance, 0.0)


def load_prices(path: str | Path) -> pd.DataFrame:
    """Load raw close prices from Parquet or CSV.

    Args:
        path: File created by ``src.data.download``.

    Returns:
        Date-indexed close-price dataframe.
    """
    input_path: Path = Path(path)
    if input_path.suffix == ".parquet":
        prices: pd.DataFrame = pd.read_parquet(path=input_path)
    else:
        prices = pd.read_csv(filepath_or_buffer=input_path, index_col="date")
    prices.index = pd.to_datetime(arg=prices.index)
    return prices.sort_index(axis=0).sort_index(axis=1)


def compute_log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Compute cleaned log returns from close prices.

    Args:
        prices: Close-price dataframe.

    Returns:
        Date-indexed log-return dataframe.
    """
    filled_prices: pd.DataFrame = prices.ffill(limit=3)
    positive_prices: pd.DataFrame = filled_prices.where(cond=filled_prices > 0.0)
    log_returns: pd.DataFrame = np.log(positive_prices).diff()
    log_returns = log_returns.replace(to_replace=[np.inf, -np.inf], value=np.nan)
    return log_returns


def build_currency_returns(log_returns: pd.DataFrame) -> pd.DataFrame:
    """Build target-currency returns measured in USD.

    Args:
        log_returns: Log returns for raw Yahoo symbols.

    Returns:
        Log returns for the target currency basket in USD terms.
    """
    currency_returns: pd.DataFrame = pd.DataFrame(index=log_returns.index)
    currency_returns["USD"] = 0.0
    missing_pairs: list[str] = []
    for currency, pair_symbol in FX_USD_PAIRS.items():
        if pair_symbol not in log_returns.columns:
            missing_pairs.append(pair_symbol)
            continue
        currency_returns[currency] = -log_returns[pair_symbol]
    if missing_pairs:
        raise KeyError(f"Missing FX pairs required for targets: {missing_pairs}.")
    return currency_returns.loc[:, list(TARGET_CURRENCIES)].dropna(how="any")


def center_currency_returns(currency_returns: pd.DataFrame) -> pd.DataFrame:
    """Center each date's currency returns around the cross-currency mean.

    This makes USD informative in relative-return inputs: although its raw USD
    return is always zero, after centering it represents the negative average
    return of the full currency basket on that date.
    """
    return currency_returns.sub(currency_returns.mean(axis=1), axis=0)


def filter_node_returns(
    log_returns: pd.DataFrame,
    symbol_blocks: dict[str, list[str]],
    max_missing_ratio: float,
    min_observations: int,
) -> tuple[dict[str, pd.DataFrame], list[dict[str, str]]]:
    """Create one return matrix per heterogeneous graph node.

    Args:
        log_returns: Log returns for all downloaded market symbols.
        symbol_blocks: Configured symbol groups keyed by node/category name.
        max_missing_ratio: Maximum tolerated missing ratio per symbol.
        min_observations: Minimum non-null observations per symbol.

    Returns:
        Mapping from node/category name to cleaned returns and filtering summary rows.
    """
    node_returns: dict[str, pd.DataFrame] = {}
    summary_rows: list[dict[str, str]] = []
    for category, symbols in symbol_blocks.items():
        available_symbols: list[str] = [
            symbol for symbol in symbols if symbol in log_returns
        ]
        if not available_symbols:
            summary_rows.append(
                build_node_filter_row(
                    category=category,
                    configured_symbols=symbols,
                    available_symbols=[],
                    valid_symbols=[],
                    removed_symbols=symbols,
                    block=pd.DataFrame(index=log_returns.index),
                    filtered_block=pd.DataFrame(index=log_returns.index),
                    ffilled_block=pd.DataFrame(index=log_returns.index),
                )
            )
            continue
        block: pd.DataFrame = log_returns.loc[:, available_symbols]
        valid_symbols: list[str] = []
        for symbol in available_symbols:
            series: pd.Series = block[symbol]
            missing_ratio: float = float(series.isna().mean())
            observations: int = int(series.notna().sum())
            if missing_ratio <= max_missing_ratio and observations >= min_observations:
                valid_symbols.append(symbol)
        removed_symbols: list[str] = [
            symbol for symbol in symbols if symbol not in valid_symbols
        ]
        filtered_block: pd.DataFrame = block.loc[:, valid_symbols]
        ffilled_block: pd.DataFrame = filtered_block.ffill(limit=3)
        summary_rows.append(
            build_node_filter_row(
                category=category,
                configured_symbols=symbols,
                available_symbols=available_symbols,
                valid_symbols=valid_symbols,
                removed_symbols=removed_symbols,
                block=block,
                filtered_block=filtered_block,
                ffilled_block=ffilled_block,
            )
        )
        if not valid_symbols:
            continue
        cleaned_block: pd.DataFrame = ffilled_block.fillna(value=0.0)
        node_returns[category] = cleaned_block
    return node_returns, summary_rows


def compute_rolling_targets(
    currency_returns: pd.DataFrame,
    target_window: int,
    risk_aversion: float,
    ridge: float,
    allow_short: bool,
) -> tuple[pd.DataFrame, pd.Series]:
    """Compute rolling optimal weights and variance targets.

    Args:
        currency_returns: Currency log returns in USD terms.
        target_window: Number of observations used by the rolling optimizer.
        risk_aversion: Markowitz risk-aversion coefficient.
        ridge: Covariance regularization.
        allow_short: Whether negative weights are allowed.

    Returns:
        Pair with optimal-weight dataframe and variance series.
    """
    weights: list[np.ndarray] = []
    variances: list[float] = []
    dates: list[pd.Timestamp] = []
    values: np.ndarray = currency_returns.to_numpy(dtype=np.float64)
    for end_idx in range(target_window, values.shape[0]):
        window: np.ndarray = values[end_idx - target_window : end_idx]
        weight, variance = compute_markowitz_weights(
            returns=window,
            risk_aversion=risk_aversion,
            ridge=ridge,
            allow_short=allow_short,
        )
        weights.append(weight)
        variances.append(variance)
        dates.append(currency_returns.index[end_idx])
    weight_df: pd.DataFrame = pd.DataFrame(
        data=np.vstack(tup=weights),
        index=pd.Index(data=dates, name="date"),
        columns=list(TARGET_CURRENCIES),
    )
    variance_series: pd.Series = pd.Series(
        data=variances,
        index=pd.Index(data=dates, name="date"),
        name="variance",
    )
    return weight_df, variance_series


def compute_standardization_stats(array: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compute per-feature standardization stats over sample and time axes.

    Args:
        array: Array whose final axis represents features.

    Returns:
        Tuple with broadcastable mean and std arrays.
    """
    reduce_axes: tuple[int, ...] = tuple(range(array.ndim - 1))
    mean: np.ndarray = np.mean(
        a=array,
        axis=reduce_axes,
        keepdims=True,
    )
    std: np.ndarray = np.std(
        a=array,
        axis=reduce_axes,
        keepdims=True,
    )
    return mean, np.maximum(std, 1.0e-8)


def apply_standardization(
    array: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
) -> np.ndarray:
    """Apply precomputed feature-wise standardization."""
    standardized: np.ndarray = (array - mean) / np.maximum(std, 1.0e-8)
    return np.nan_to_num(x=standardized, nan=0.0, posinf=0.0, neginf=0.0)


def build_supervised_arrays(
    currency_returns: pd.DataFrame,
    node_returns: dict[str, pd.DataFrame],
    lookback: int,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Create tensors for model training and evaluation.

    Args:
        currency_returns: Currency log returns in USD terms.
        node_returns: Mapping from node names to node return matrices.
        lookback: Number of previous days per input sample.

    Returns:
        Tuple with NumPy arrays and serializable metadata.
    """
    common_index: pd.DatetimeIndex = currency_returns.index
    for block in node_returns.values():
        common_index = common_index.intersection(other=block.index)
    common_index = common_index.sort_values()

    currency_aligned: pd.DataFrame = currency_returns.loc[common_index]
    node_aligned: dict[str, pd.DataFrame] = {
        node_name: block.loc[common_index]
        for node_name, block in node_returns.items()
    }

    portfolio_features: list[np.ndarray] = []
    portfolio_raw_returns: list[np.ndarray] = []
    next_log_returns: list[np.ndarray] = []
    dates: list[str] = []
    node_windows: dict[str, list[np.ndarray]] = {
        node_name: [] for node_name in node_aligned
    }

    currency_values: np.ndarray = currency_aligned.to_numpy(dtype=np.float64)
    node_values: dict[str, np.ndarray] = {
        node_name: block.to_numpy(dtype=np.float64)
        for node_name, block in node_aligned.items()
    }

    for end_idx in range(lookback - 1, len(common_index) - 1):
        start_idx: int = end_idx - lookback + 1
        target_idx: int = end_idx + 1
        portfolio_window: np.ndarray = currency_values[start_idx : end_idx + 1]
        if not np.all(np.isfinite(portfolio_window)):
            continue
        skip_sample: bool = False
        current_node_windows: dict[str, np.ndarray] = {}
        for node_name, values in node_values.items():
            node_window: np.ndarray = values[start_idx : end_idx + 1]
            if not np.all(np.isfinite(node_window)):
                skip_sample = True
                break
            current_node_windows[node_name] = node_window
        if skip_sample:
            continue
        portfolio_features.append(portfolio_window)
        portfolio_raw_returns.append(portfolio_window)
        next_log_returns.append(currency_values[target_idx])
        dates.append(str(common_index[target_idx].date()))
        for node_name, node_window in current_node_windows.items():
            node_windows[node_name].append(node_window)

    arrays: dict[str, np.ndarray] = {
        "portfolio_features": np.asarray(a=portfolio_features, dtype=np.float32),
        "portfolio_raw_returns": np.asarray(a=portfolio_raw_returns, dtype=np.float32),
        "next_log_returns": np.asarray(a=next_log_returns, dtype=np.float32),
        "dates": np.asarray(a=dates, dtype=str),
    }
    node_arrays: dict[str, np.ndarray] = {
        node_name: np.asarray(a=windows, dtype=np.float32)
        for node_name, windows in node_windows.items()
    }
    metadata: dict[str, Any] = {
        "currencies": list(TARGET_CURRENCIES),
        "lookback": lookback,
        "portfolio_input_dim": int(arrays["portfolio_features"].shape[-1]),
        "node_names": ["portfolio_signal", *sorted(node_arrays)],
        "node_input_dims": {
            "portfolio_signal": int(arrays["portfolio_features"].shape[-1]),
            **{
                node_name: int(node_array.shape[-1])
                for node_name, node_array in node_arrays.items()
            },
        },
        "node_symbols": {
            node_name: list(node_aligned[node_name].columns)
            for node_name in node_aligned
        },
        "num_samples": int(arrays["next_log_returns"].shape[0]),
    }
    arrays.update({f"node::{key}": value for key, value in node_arrays.items()})
    return arrays, metadata


def resolve_dataset_splits(
    arrays: dict[str, np.ndarray],
    metadata: dict[str, Any],
    train_fraction: float,
    val_fraction: float,
    split_date_ranges: dict[str, dict[str, str]] | None = None,
) -> dict[str, Any]:
    """Resolve dataset split metadata from fractions or explicit date ranges.

    Args:
        arrays: Dictionary of generated arrays.
        metadata: Serializable metadata.
        train_fraction: Fraction assigned to the training split.
        val_fraction: Fraction assigned to the validation split.
        split_date_ranges: Optional date ranges keyed by split name.

    Returns:
        Updated metadata dictionary.
    """
    num_samples: int = int(metadata["num_samples"])
    if split_date_ranges:
        sample_dates: pd.DatetimeIndex = pd.to_datetime(arg=arrays["dates"])
        metadata["splits"] = {}
        metadata["split_dates"] = {}
        metadata["split_actual_dates"] = {}
        for split_name, bounds in split_date_ranges.items():
            start_date: pd.Timestamp = pd.Timestamp(bounds["start"])
            end_date: pd.Timestamp = pd.Timestamp(bounds["end"])
            if end_date <= start_date:
                raise ValueError(
                    f"Split {split_name!r} end date must be after start date."
                )
            mask: np.ndarray = np.asarray(
                (sample_dates >= start_date) & (sample_dates < end_date)
            )
            indices: np.ndarray = np.flatnonzero(mask)
            if indices.size == 0:
                raise ValueError(
                    f"Split {split_name!r} has no samples in "
                    f"[{start_date.date()}, {end_date.date()})."
                )
            metadata["splits"][split_name] = [
                int(indices.min()),
                int(indices.max()) + 1,
            ]
            metadata["split_dates"][split_name] = [
                str(start_date.date()),
                str(end_date.date()),
            ]
            metadata["split_actual_dates"][split_name] = [
                str(sample_dates[indices.min()].date()),
                str(sample_dates[indices.max()].date()),
            ]
    else:
        train_end: int = int(num_samples * train_fraction)
        val_end: int = int(num_samples * (train_fraction + val_fraction))
        metadata["splits"] = {
            "train": [0, train_end],
            "val": [train_end, val_end],
            "test": [val_end, num_samples],
        }
    return metadata


def standardize_processed_arrays(
    arrays: dict[str, np.ndarray],
    metadata: dict[str, Any],
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Standardize model inputs using train-only statistics.

    Args:
        arrays: Raw processed arrays.
        metadata: Metadata containing split bounds.

    Returns:
        Updated arrays and metadata with normalization stats.
    """
    train_start, train_end = metadata["splits"]["train"]
    train_slice = slice(int(train_start), int(train_end))

    portfolio_mean, portfolio_std = compute_standardization_stats(
        array=arrays["portfolio_features"][train_slice]
    )
    arrays["portfolio_features"] = apply_standardization(
        array=arrays["portfolio_features"],
        mean=portfolio_mean,
        std=portfolio_std,
    )

    normalization_stats: dict[str, dict[str, list[float]]] = {
        "portfolio_signal": {
            "mean": portfolio_mean.reshape(-1).astype(float).tolist(),
            "std": portfolio_std.reshape(-1).astype(float).tolist(),
        }
    }

    for key in sorted(arrays):
        if not key.startswith("node::"):
            continue
        node_mean, node_std = compute_standardization_stats(
            array=arrays[key][train_slice]
        )
        arrays[key] = apply_standardization(
            array=arrays[key],
            mean=node_mean,
            std=node_std,
        )
        node_name = key.split("::", maxsplit=1)[1]
        normalization_stats[node_name] = {
            "mean": node_mean.reshape(-1).astype(float).tolist(),
            "std": node_std.reshape(-1).astype(float).tolist(),
        }

    metadata["normalization"] = {
        "fit_split": "train",
        "stats_by_node": normalization_stats,
    }
    return arrays, metadata


def save_processed_dataset(
    arrays: dict[str, np.ndarray],
    metadata: dict[str, Any],
    processed_dir: str | Path,
) -> None:
    """Save processed arrays and metadata to disk."""
    output_dir: Path = ensure_dir(path=processed_dir)
    nodes_dir: Path = ensure_dir(path=output_dir / "nodes")

    for key, array in arrays.items():
        if key.startswith("node::"):
            node_name: str = key.split("::", maxsplit=1)[1]
            np.save(file=nodes_dir / f"{node_name}.npy", arr=array)
        else:
            np.save(file=output_dir / f"{key}.npy", arr=array)
    write_json(data=metadata, path=output_dir / "metadata.json")


def transform_data(config: dict[str, Any], show_summaries: bool = False) -> dict[str, Any]:
    """Run the full data transformation pipeline.

    Args:
        config: Transformation configuration.
        show_summaries: Whether to print compact data-quality summaries.

    Returns:
        Metadata of the generated supervised dataset.
    """
    prices: pd.DataFrame = load_prices(path=config["raw_prices_path"])
    symbol_metadata: pd.DataFrame = load_symbol_metadata(path=config["raw_metadata_path"])
    symbol_blocks: dict[str, list[str]] = symbol_blocks_from_metadata(
        metadata=symbol_metadata
    )
    log_returns: pd.DataFrame = compute_log_returns(prices=prices)
    currency_returns: pd.DataFrame = build_currency_returns(log_returns=log_returns)
    center_returns = bool(config.get("center", False))
    if center_returns:
        currency_returns = center_currency_returns(currency_returns=currency_returns)
    node_returns, node_filter_rows = filter_node_returns(
        log_returns=log_returns,
        symbol_blocks=symbol_blocks,
        max_missing_ratio=float(config.get("max_missing_ratio", 0.25)),
        min_observations=int(config.get("min_observations_per_node", 120)),
    )
    arrays, metadata = build_supervised_arrays(
        currency_returns=currency_returns,
        node_returns=node_returns,
        lookback=int(config.get("lookback", 20)),
    )
    metadata["centered_currency_returns"] = center_returns
    metadata = resolve_dataset_splits(
        arrays=arrays,
        metadata=metadata,
        train_fraction=float(config.get("train_fraction", 0.70)),
        val_fraction=float(config.get("val_fraction", 0.15)),
        split_date_ranges=config.get("split_date_ranges"),
    )
    arrays, metadata = standardize_processed_arrays(
        arrays=arrays,
        metadata=metadata,
    )
    processed_dir: Path = Path(config.get("processed_dir", "data/processed"))
    ensure_dir(path=processed_dir)
    currency_returns.to_csv(path_or_buf=processed_dir / "currency_log_returns.csv")
    save_processed_dataset(
        arrays=arrays,
        metadata=metadata,
        processed_dir=processed_dir,
    )
    if show_summaries:
        log_transform_summaries(
            prices=prices,
            log_returns=log_returns,
            currency_returns=currency_returns,
            node_returns=node_returns,
            arrays=arrays,
            metadata=metadata,
            node_filter_rows=node_filter_rows,
        )
    return metadata


def main() -> None:
    """Create the processed supervised dataset from raw prices."""
    args: argparse.Namespace = parse_args()
    config: dict[str, Any] = load_yaml(path=args.config)
    transform_data(config=config, show_summaries=True)


if __name__ == "__main__":
    main()
