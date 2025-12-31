from datetime import date, timedelta
from pathlib import Path

import polars as pl

from src.data.utils import drop_null_columns, find_project_root


def read_us_indicators_weekly() -> pl.DataFrame:
    """Load and merge U.S. indicators into a single weekly DataFrame.

    Each CSV in `data/indicators/indicators/` must contain:
      - a DATE column
      - exactly one value column (the indicator)

    This function:
      1) Renames DATE -> Date
      2) Converts Date to pl.Date
      3) Generates a weekly index (Monday) from the global min/max across files
      4) For each indicator:
         - aligns it to the weekly index
         - treats 0.0 values as nulls (missing data)
         - forward-fills with the last previous non-null value
         - fills remaining nulls (no previous value) with 0.0

    Returns:
        Polars DataFrame with a `Date` column plus one column per indicator, at
        weekly frequency (Monday).
    """
    project_root: Path = find_project_root(start_path=Path(__file__).parent)
    indicators_dir: Path = project_root / "data" / "indicators" / "indicators"

    if not indicators_dir.is_dir():
        raise RuntimeError(f"Indicators directory not found: {indicators_dir}")

    csv_files: list[Path] = sorted(indicators_dir.glob(pattern="*.csv"))
    if not csv_files:
        raise RuntimeError(f"No CSV files found in {indicators_dir}")

    series_list: list[pl.DataFrame] = []
    global_min: date | None = None
    global_max: date | None = None

    for csv_path in csv_files:
        df: pl.DataFrame = pl.read_csv(
            source=csv_path,
            truncate_ragged_lines=True,
            ignore_errors=True,
        )

        if "DATE" not in df.columns:
            raise RuntimeError(f"Missing DATE column in {csv_path.name}")

        value_columns: list[str] = [col for col in df.columns if col != "DATE"]
        if len(value_columns) != 1:
            raise RuntimeError(
                f"Expected exactly one value column in {csv_path.name}, "
                f"found {value_columns}"
            )

        value_col: str = value_columns[0]
        value_expr: pl.Expr = pl.col(name=value_col).cast(
            dtype=pl.Float64,
            strict=False,
        )

        df = (
            df.rename(mapping={"DATE": "Date"})
            .with_columns(
                pl.col(name="Date")
                .cast(dtype=pl.Utf8)
                .str.to_date()
                .alias(name="Date"),
                pl.when(value_expr == 0.0)
                .then(statement=None)
                .otherwise(statement=value_expr)
                .alias(name=value_col),
            )
            .select(["Date", value_col])
            .drop_nulls(subset=["Date"])
            .unique(subset=["Date"], keep="last")
            .sort(by="Date")
        )

        if df.height == 0:
            continue

        min_date: date = df.select(pl.col(name="Date").min()).item()
        max_date: date = df.select(pl.col(name="Date").max()).item()

        global_min = min_date if global_min is None else min(global_min, min_date)
        global_max = max_date if global_max is None else max(global_max, max_date)

        series_list.append(df)

    if not series_list or global_min is None or global_max is None:
        raise RuntimeError(f"No usable indicator data found in {indicators_dir}")

    start_monday: date = global_min - timedelta(days=global_min.weekday())
    end_monday: date = global_max - timedelta(days=global_max.weekday())

    weekly_index: pl.DataFrame = pl.DataFrame(
        data={
            "Date": pl.date_range(
                start=start_monday,
                end=end_monday,
                interval="1w",
                closed="both",
                eager=True,
            )
        }
    )

    aligned_series: list[pl.DataFrame] = []
    for df in series_list:
        value_col = next(c for c in df.columns if c != "Date")

        aligned: pl.DataFrame = weekly_index.join(
            other=df, on="Date", how="left"
        ).with_columns(
            pl.col(name=value_col).fill_null(strategy="forward").fill_null(value=0.0)
        )
        aligned_series.append(aligned)

    us_indicators_weekly: pl.DataFrame = aligned_series[0]
    for df in aligned_series[1:]:
        value_col = next(c for c in df.columns if c != "Date")
        us_indicators_weekly = us_indicators_weekly.join(
            other=df.select(["Date", value_col]),
            on="Date",
            how="left",
        )

    return us_indicators_weekly


def read_sp500_stocks_daily() -> pl.DataFrame:
    """Load S&P 500 daily stock prices and enrich them with company sector data.

    The function:
    - Loads `data/sp500_stocks.csv`
    - Loads `data/sp500_companies.csv`
    - Joins both datasets on the `Symbol` column
    - Adds a `Sector` column to each daily stock observation

    Returns:
        A Polars DataFrame containing daily stock prices with sector information.
    """
    project_root: Path = find_project_root(start_path=Path(__file__).parent)

    stocks_path: Path = project_root / "data" / "sp500stocks" / "sp500_stocks.csv"
    companies_path: Path = project_root / "data" / "sp500stocks" / "sp500_companies.csv"

    if not stocks_path.is_file():
        raise RuntimeError(f"Missing file: {stocks_path}")

    if not companies_path.is_file():
        raise RuntimeError(f"Missing file: {companies_path}")

    stocks_df: pl.DataFrame = pl.read_csv(
        source=stocks_path, truncate_ragged_lines=True
    )
    companies_df: pl.DataFrame = pl.read_csv(
        source=companies_path, truncate_ragged_lines=True
    )

    required_stock_columns: set[str] = {
        "Date",
        "Symbol",
        "Adj Close",
        "Close",
        "High",
        "Low",
        "Open",
        "Volume",
    }

    required_company_columns: set[str] = {"Symbol", "Sector"}

    if not required_stock_columns.issubset(stocks_df.columns):
        raise RuntimeError(
            f"Unexpected schema in sp500_stocks.csv: {stocks_df.columns}"
        )

    if not required_company_columns.issubset(companies_df.columns):
        raise RuntimeError(
            f"Unexpected schema in sp500_companies.csv: {companies_df.columns}"
        )

    companies_sector_df: pl.DataFrame = companies_df.select(["Symbol", "Sector"])

    sp500_stocks_daily: pl.DataFrame = stocks_df.join(
        other=companies_sector_df,
        on="Symbol",
        how="left",
    )

    return sp500_stocks_daily


if __name__ == "__main__":
    us_indicators_weekly: pl.DataFrame = read_us_indicators_weekly()
    us_indicators_weekly: pl.DataFrame = drop_null_columns(df=us_indicators_weekly)
    sp500_stocks_daily: pl.DataFrame = read_sp500_stocks_daily()

    print("\nUS indicators (weekly):")
    print(us_indicators_weekly.describe())

    print("\nS&P 500 stocks (daily) with sector:")
    print(sp500_stocks_daily.describe())
