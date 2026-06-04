"""Download Yahoo Finance prices for the FOREX HTGNN universe."""

from __future__ import annotations

import argparse
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from datetime import date
import logging
import os
from pathlib import Path
from typing import Any

import pandas as pd
from tqdm import tqdm
import yfinance as yf

from src.data.symbols import symbol_names, symbols_by_category
from src.data.summary import (
    date_bounds,
    print_summary_table,
    summarize_frame,
    summarize_numeric_frame,
)
from src.utils import ensure_dir, load_yaml


@contextmanager
def suppress_yfinance_output() -> Any:
    """Temporarily silence yfinance logging and stream output."""
    logger: logging.Logger = logging.getLogger(name="yfinance")
    previous_disabled: bool = logger.disabled
    logger.disabled = True
    with open(file=os.devnull, mode="w", encoding="utf-8") as devnull:
        with redirect_stdout(new_target=devnull), redirect_stderr(new_target=devnull):
            try:
                yield
            finally:
                logger.disabled = previous_disabled


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the downloader.

    Returns:
        Parsed command-line namespace.
    """
    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        description="Download prices for the FOREX HTGNN symbol universe."
    )
    parser.add_argument(
        "--config",
        default="configs/download.yaml",
        help="Path to the downloader YAML configuration.",
    )
    return parser.parse_args()


def extract_close_prices(
    downloaded: pd.DataFrame, expected_symbols: list[str] | None = None
) -> pd.DataFrame:
    """Extract close or adjusted close prices from yfinance output.

    Args:
        downloaded: Dataframe returned by ``yf.download``.
        expected_symbols: Optional ordered ticker list used to normalize columns.

    Returns:
        Date-indexed dataframe with one column per ticker.
    """
    if downloaded.empty and expected_symbols is not None:
        return pd.DataFrame(
            index=pd.DatetimeIndex(data=[], name=downloaded.index.name),
            columns=expected_symbols,
            dtype=float,
        )
    if isinstance(downloaded.columns, pd.MultiIndex):
        first_level: pd.Index = downloaded.columns.get_level_values(level=0)
        if "Close" in first_level:
            prices: pd.DataFrame = downloaded["Close"]
        elif "Adj Close" in first_level:
            prices = downloaded["Adj Close"]
        else:
            raise KeyError("The yfinance output has no Close or Adj Close field.")
    else:
        if "Close" not in downloaded.columns:
            raise KeyError("The yfinance output has no Close column.")
        prices = downloaded[["Close"]]
        if expected_symbols is not None and len(expected_symbols) == 1:
            prices = prices.rename(columns={"Close": expected_symbols[0]})
    prices = prices.sort_index(axis=0)
    prices.index = pd.to_datetime(arg=prices.index).tz_localize(tz=None)
    if expected_symbols is not None:
        prices = prices.reindex(columns=expected_symbols)
    return prices


def configured_symbol_blocks(config: dict[str, Any]) -> dict[str, list[str]]:
    """Read Yahoo symbol blocks from config, falling back to defaults."""
    raw_blocks: Any = config.get("symbol_blocks")
    if raw_blocks is None:
        return symbols_by_category()
    if not isinstance(raw_blocks, dict):
        raise TypeError("download config 'symbol_blocks' must be a mapping.")

    blocks: dict[str, list[str]] = {}
    for block_name, symbols in raw_blocks.items():
        if not isinstance(block_name, str):
            raise TypeError("Every symbol block name must be a string.")
        if not isinstance(symbols, list) or not all(
            isinstance(symbol, str) for symbol in symbols
        ):
            raise TypeError(f"Block '{block_name}' must contain a list of strings.")
        seen: set[str] = set()
        blocks[block_name] = [
            symbol for symbol in symbols if not (symbol in seen or seen.add(symbol))
        ]
    if not blocks:
        raise ValueError("download config 'symbol_blocks' cannot be empty.")
    return blocks


def build_metadata(
    prices: pd.DataFrame, symbol_blocks: dict[str, list[str]]
) -> pd.DataFrame:
    """Build metadata with coverage information for every configured symbol.

    Args:
        prices: Downloaded close-price dataframe.
        symbol_blocks: Configured symbol groups keyed by category/block name.

    Returns:
        Metadata dataframe with category, symbol, name, date range, and rows.
    """
    records: list[dict[str, Any]] = []
    names: dict[str, str] = symbol_names()
    for category, symbols in symbol_blocks.items():
        for symbol in symbols:
            if symbol not in prices.columns:
                first_date: str | None = None
                last_date: str | None = None
                rows: int = 0
            else:
                series: pd.Series = prices[symbol].dropna()
                first_date = None if series.empty else str(series.index.min().date())
                last_date = None if series.empty else str(series.index.max().date())
                rows = int(series.shape[0])
            records.append(
                {
                    "category": category,
                    "symbol": symbol,
                    "name": names.get(symbol, symbol),
                    "first_date": first_date,
                    "last_date": last_date,
                    "rows": rows,
                }
            )
    return pd.DataFrame.from_records(data=records)


def summarize_price_block(name: str, prices: pd.DataFrame) -> dict[str, str]:
    """Build one row describing a downloaded raw price block."""
    first_date, last_date = date_bounds(index=prices.index)
    missing_cells: int = int(prices.isna().sum().sum())
    total_cells: int = int(prices.size)
    missing_pct: float = missing_cells / total_cells * 100.0 if total_cells else 0.0
    non_empty_symbols: int = int(prices.notna().any(axis=0).sum())
    return {
        "table": name,
        "rows": str(prices.shape[0]),
        "symbols": str(prices.shape[1]),
        "non_empty": str(non_empty_symbols),
        "first_date": first_date,
        "last_date": last_date,
        "missing": str(missing_cells),
        "missing_pct": f"{missing_pct:.2f}%",
    }


def build_coverage_summary_rows(metadata: pd.DataFrame) -> list[dict[str, str]]:
    """Summarize raw symbol coverage by category."""
    rows: list[dict[str, str]] = []
    for category, block in metadata.groupby(by="category", sort=True):
        rows.append(
            {
                "category": str(category),
                "symbols": str(block.shape[0]),
                "empty": str(int(block["rows"].eq(0).sum())),
                "min_rows": str(int(block["rows"].min())),
                "max_rows": str(int(block["rows"].max())),
                "first_date": str(block["first_date"].dropna().min() or ""),
                "last_date": str(block["last_date"].dropna().max() or ""),
            }
        )
    return rows


def log_raw_summaries(
    prices: pd.DataFrame,
    metadata: pd.DataFrame,
    block_prices: dict[str, pd.DataFrame],
) -> None:
    """Print boxed summaries for raw downloaded tables."""
    print_summary_table(
        title="Raw Price Tables",
        rows=[
            summarize_price_block(name="prices.parquet", prices=prices),
            *[
                summarize_price_block(
                    name=f"blocks/{category}.parquet", prices=block
                )
                for category, block in sorted(block_prices.items())
            ],
        ],
    )
    print_summary_table(
        title="Raw Numeric Ranges",
        rows=[
            summarize_numeric_frame(name="prices.parquet", frame=prices),
            summarize_numeric_frame(name="symbol_metadata.parquet", frame=metadata),
        ],
    )
    print_summary_table(
        title="Raw Symbol Coverage",
        rows=build_coverage_summary_rows(metadata=metadata),
    )
    print_summary_table(
        title="Raw Metadata Table",
        rows=[summarize_frame(name="symbol_metadata.parquet", frame=metadata)],
    )


def download_symbol_block(
    symbols: list[str], config: dict[str, Any], end: str
) -> tuple[pd.DataFrame, dict[str, str]]:
    """Download one configured symbol block.

    Args:
        symbols: Yahoo Finance ticker list for the block.
        config: Downloader configuration.
        end: Inclusive-looking end date passed to yfinance.

    Returns:
        Close-price dataframe and final failures for the requested symbols.
    """
    download_kwargs: dict[str, Any] = {
        "start": config.get("start", "2021-05-06"),
        "end": end,
        "interval": config.get("interval", "1d"),
        "auto_adjust": bool(config.get("auto_adjust", True)),
        "threads": bool(config.get("threads", True)),
        "progress": False,
        "group_by": "column",
    }
    with suppress_yfinance_output():
        downloaded: pd.DataFrame = yf.download(tickers=symbols, **download_kwargs)
    prices: pd.DataFrame = extract_close_prices(
        downloaded=downloaded, expected_symbols=symbols
    )

    failed_symbols: list[str] = [
        symbol for symbol in symbols if prices[symbol].dropna().empty
    ]
    final_failures: dict[str, str] = {}
    for symbol in failed_symbols:
        with suppress_yfinance_output():
            retry_downloaded: pd.DataFrame = yf.download(
                tickers=[symbol], **download_kwargs
            )
        retry_prices: pd.DataFrame = extract_close_prices(
            downloaded=retry_downloaded, expected_symbols=[symbol]
        )
        if retry_prices[symbol].dropna().empty:
            final_failures[symbol] = "no price data returned"
            continue
        prices = pd.concat(
            objs=[prices.drop(columns=[symbol]), retry_prices], axis=1, sort=True
        ).reindex(columns=symbols)
    return prices, final_failures


def download_prices(
    config: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, pd.DataFrame], dict[str, str]]:
    """Download configured prices and save their metadata.

    Args:
        config: Downloader configuration.

    Returns:
        Tuple containing prices, metadata, per-category prices, and failures.
    """
    end_value: str | None = config.get("end")
    end: str = end_value if end_value is not None else date.today().isoformat()
    grouped_symbols: dict[str, list[str]] = configured_symbol_blocks(config=config)
    block_prices: dict[str, pd.DataFrame] = {}
    progress = tqdm(
        grouped_symbols.items(),
        desc="Downloading Yahoo Finance blocks",
        unit="block",
    )
    failures: dict[str, str] = {}
    for category, symbols in progress:
        progress.set_postfix_str(f"{category} ({len(symbols)})")
        block_prices[category], block_failures = download_symbol_block(
            symbols=symbols, config=config, end=end
        )
        for symbol, reason in block_failures.items():
            failures[symbol] = f"{category}: {reason}"
    prices: pd.DataFrame = pd.concat(
        objs=block_prices.values(), axis=1, sort=True
    ).sort_index(axis=0)
    metadata: pd.DataFrame = build_metadata(
        prices=prices, symbol_blocks=grouped_symbols
    )
    if int(metadata["rows"].sum()) == 0:
        raise RuntimeError("Yahoo Finance returned no price data for any configured symbol.")
    return prices, metadata, block_prices, failures


def main() -> None:
    """Download prices and write raw CSV files."""
    args: argparse.Namespace = parse_args()
    config: dict[str, Any] = load_yaml(path=args.config)
    prices, metadata, block_prices, failures = download_prices(config=config)

    prices_path: Path = Path(config.get("raw_prices_path", "data/raw/prices.parquet"))
    metadata_path: Path = Path(
        config.get("raw_metadata_path", "data/raw/symbol_metadata.parquet")
    )
    blocks_dir: Path = Path(config.get("raw_blocks_dir", "data/raw/blocks"))
    ensure_dir(path=prices_path.parent)
    ensure_dir(path=metadata_path.parent)
    ensure_dir(path=blocks_dir)
    prices = prices.rename_axis(index="date")
    prices.to_parquet(path=prices_path, index=True)
    metadata.to_parquet(path=metadata_path, index=False)
    for category, block in block_prices.items():
        block_path: Path = blocks_dir / f"{category}.parquet"
        block.rename_axis(index="date").to_parquet(path=block_path, index=True)

    print(f"Saved prices to {prices_path} with shape {prices.shape}.")
    print(f"Saved {len(block_prices)} price blocks to {blocks_dir}.")
    print(f"Saved metadata to {metadata_path} with {metadata.shape[0]} rows.")
    if failures:
        failed_symbols: str = ", ".join(sorted(failures))
        print(f"Symbols without price data after retry: {failed_symbols}.")
    log_raw_summaries(prices=prices, metadata=metadata, block_prices=block_prices)


if __name__ == "__main__":
    main()
