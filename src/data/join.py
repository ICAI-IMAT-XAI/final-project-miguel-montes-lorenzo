import polars as pl

from src.data.read import (
    drop_null_columns,
    read_sp500_stocks_daily,
    read_us_indicators_weekly,
)


def to_sp500_stocks_weekly(sp500_stocks_daily: pl.DataFrame) -> pl.DataFrame:
    """Aggregate S&P 500 daily OHLCV data to weekly frequency.

    Weekly aggregation rules per Symbol:
    - Open: first day of the week
    - Close, Adj Close: last day of the week
    - High: weekly max
    - Low: weekly min
    - Volume: weekly sum

    Weeks are anchored to Monday (start_by="monday"). The resulting Date is the
    week start date.

    Args:
        sp500_stocks_daily: Daily S&P 500 stock data with columns
            Date, Symbol, Adj Close, Close, High, Low, Open, Volume.

    Returns:
        Weekly-aggregated Polars DataFrame with the same columns.
    """
    required_columns: set[str] = {
        "Date",
        "Symbol",
        "Sector",
        "Adj Close",
        "Close",
        "High",
        "Low",
        "Open",
        "Volume",
    }
    if not required_columns.issubset(sp500_stocks_daily.columns):
        raise RuntimeError(
            f"Unexpected schema in sp500_stocks_daily: {sp500_stocks_daily.columns}"
        )

    weekly: pl.DataFrame = (
        sp500_stocks_daily.with_columns(
            pl.col(name="Date").str.to_date().alias(name="Date"),
            pl.col(name="Open").cast(dtype=pl.Float64, strict=False).alias(name="Open"),
            pl.col(name="High").cast(dtype=pl.Float64, strict=False).alias(name="High"),
            pl.col(name="Low").cast(dtype=pl.Float64, strict=False).alias(name="Low"),
            pl.col(name="Close")
            .cast(dtype=pl.Float64, strict=False)
            .alias(name="Close"),
            pl.col(name="Adj Close")
            .cast(dtype=pl.Float64, strict=False)
            .alias(name="Adj Close"),
            pl.col(name="Volume")
            .cast(dtype=pl.Int64, strict=False)
            .alias(name="Volume"),
        )
        .sort(by=["Symbol", "Date"])
        .group_by_dynamic(
            index_column="Date",
            every="1w",
            group_by=["Symbol"],
            start_by="monday",
        )
        .agg(
            pl.col(name="Sector").first().alias(name="Sector"),
            pl.col(name="Open").first().alias(name="Open"),
            pl.col(name="Close").last().alias(name="Close"),
            pl.col(name="Adj Close").last().alias(name="Adj Close"),
            pl.col(name="High").max().alias(name="High"),
            pl.col(name="Low").min().alias(name="Low"),
            pl.col(name="Volume").sum().alias(name="Volume"),
        )
        .sort(by=["Symbol", "Date"])
        .select([
            "Date",
            "Symbol",
            "Sector",
            "Adj Close",
            "Close",
            "High",
            "Low",
            "Open",
            "Volume",
        ])
    )

    return weekly


def compute_sp500_returns_weekly(
    sp500_stocks_weekly: pl.DataFrame,
    price_column_preference: str = "Adj Close",
) -> pl.DataFrame:
    """Compute weekly log-returns and aggregate them by sector and overall market.

    Steps:
    1) Compute per-stock weekly log-returns:
         r_{i,t} = log(P_{i,t} / P_{i,t-1})
       using a chosen price column (default: 'Adj Close'), with a fallback to
       'Close' when the preferred column is null.
    2) Aggregate returns by sector as the cross-sectional mean among stocks that
       have a valid return at time t (no fixed N).
    3) Aggregate returns across all stocks for an overall "SP500" return,
       computed the same way (mean among valid returns at t).

    The function is robust to missing prices/returns (common in this dataset):
    - Returns are only computed when both P_{t} and P_{t-1} are present and > 0.
    - Aggregations ignore null returns and track N per date.

    Expected columns in `sp500_stocks_weekly`:
      Date (date), Symbol (str), Sector (str),
      Adj Close (f64, optional), Close (f64, optional)

    Args:
        sp500_stocks_weekly: Weekly stock data (one row per Symbol and week).
        price_column_preference: Preferred price column for returns computation.
            Usually 'Adj Close'. If null, falls back to 'Close'.

    Returns:
        A DataFrame with columns:
          Date, Sector, Return, N
        plus an extra sector row with Sector == "SP500" for the overall index.
    """
    required_columns: set[str] = {"Date", "Symbol", "Sector", "Close", "Adj Close"}
    missing: set[str] = required_columns.difference(sp500_stocks_weekly.columns)
    if missing:
        raise RuntimeError(
            f"Missing required columns in sp500_stocks_weekly: {sorted(missing)}"
        )

    if price_column_preference not in {"Adj Close", "Close"}:
        raise RuntimeError(
            "price_column_preference must be either 'Adj Close' or 'Close'"
        )

    price_expr: pl.Expr
    if price_column_preference == "Adj Close":
        price_expr = pl.coalesce(exprs=[pl.col(name="Adj Close"), pl.col(name="Close")])
    else:
        price_expr = pl.coalesce(exprs=[pl.col(name="Close"), pl.col(name="Adj Close")])

    per_stock: pl.DataFrame = (
        sp500_stocks_weekly.select([
            pl.col(name="Date"),
            pl.col(name="Symbol"),
            pl.col(name="Sector"),
            price_expr.alias(name="P"),
        ])
        .sort(by=["Symbol", "Date"])
        .with_columns(
            pl.col(name="P")
            .shift(n=1)
            .over(partition_by="Symbol")
            .alias(name="P_prev"),
        )
        .with_columns(
            pl.when(
                pl.col(name="P").is_not_null()
                & pl.col(name="P_prev").is_not_null()
                & (pl.col(name="P") > 0.0)
                & (pl.col(name="P_prev") > 0.0)
            )
            .then(statement=(pl.col(name="P") / pl.col(name="P_prev")).log())
            .otherwise(statement=None)
            .alias(name="r")
        )
        .select(["Date", "Symbol", "Sector", "r"])
    )

    by_sector: pl.DataFrame = (
        per_stock.filter(pl.col(name="r").is_not_null())
        .group_by(["Date", "Sector"])
        .agg(
            pl.col(name="r").mean().alias(name="Return"),
            pl.len().alias(name="N"),
        )
        .sort(by=["Date", "Sector"])
    )

    overall: pl.DataFrame = (
        per_stock.filter(pl.col(name="r").is_not_null())
        .group_by(["Date"])
        .agg(
            pl.col(name="r").mean().alias(name="Return"),
            pl.len().alias(name="N"),
        )
        .with_columns(pl.lit(value="SP500").alias(name="Sector"))
        .select(["Date", "Sector", "Return", "N"])
        .sort(by=["Date", "Sector"])
    )

    sp500_returns_weekly: pl.DataFrame = (
        pl.concat(items=[by_sector, overall], how="vertical")
        .with_columns(
            pl.when(pl.col(name="Sector") == "SP500")
            .then(statement=0)
            .otherwise(statement=1)
            .alias(name="_sp500_first")
        )
        .sort(by=["Date", "_sp500_first", "Sector"])
        .drop("_sp500_first")
    )

    return sp500_returns_weekly


def transpose_sector(
    sp500_returns_weekly_long: pl.DataFrame,
    value_col: str = "Return",
    count_col: str | None = "N",
    include_counts: bool = False,
) -> pl.DataFrame:
    """Transpose sector data from long to wide format.

    Expected input format (long):
        - Date
        - Sector
        - <value_col>  (default: "Return")
        - <count_col>  (default: "N", optional)

    Output format (wide):
        - Date
        - one column per Sector containing <value_col>
        - optionally, one column per Sector containing <count_col>, prefixed with "N_"

    Args:
        sp500_returns_weekly_long: Long-format DataFrame with Date/Sector rows.
        value_col: Name of the value column to pivot (e.g., "Return").
        count_col: Name of the count column to pivot (e.g., "N"). If None, counts
            are ignored even if include_counts=True.
        include_counts: Whether to also pivot the count column.

    Returns:
        Wide-format DataFrame with sectors as columns.

    Raises:
        RuntimeError: If required columns are missing.
    """
    required: set[str] = {"Date", "Sector", value_col}
    if include_counts and count_col is not None:
        required.add(count_col)

    missing: set[str] = required.difference(sp500_returns_weekly_long.columns)
    if missing:
        raise RuntimeError(
            "Missing required columns for transpose_sector. "
            f"Missing: {sorted(missing)}. "
            f"Got columns: {sp500_returns_weekly_long.columns}"
        )

    returns_wide: pl.DataFrame = sp500_returns_weekly_long.pivot(
        index="Date",
        on="Sector",
        values=value_col,
        aggregate_function="first",
    ).sort(by="Date")

    if not include_counts or count_col is None:
        return returns_wide

    counts_wide: pl.DataFrame = (
        sp500_returns_weekly_long.select(["Date", "Sector", count_col])
        .pivot(
            index="Date",
            on="Sector",
            values=count_col,
            aggregate_function="first",
        )
        .sort("Date")
    )

    non_date_cols: list[str] = [c for c in counts_wide.columns if c != "Date"]
    counts_wide = counts_wide.rename(mapping={c: f"N_{c}" for c in non_date_cols})

    return returns_wide.join(other=counts_wide, on="Date", how="left")


def add_indicators_to_sp500_returns_weekly(
    sp500_returns_weekly: pl.DataFrame,
    us_indicators_weekly: pl.DataFrame,
) -> pl.DataFrame:
    """Join weekly US indicators into the S&P 500 sector returns DataFrame.

    Expected schema:
      - sp500_returns_weekly: must contain a `Date` column (pl.Date recommended).
      - us_indicators_weekly: must contain a `DATE` column (string or date) and
        one or more indicator columns.

    The function:
      1) Renames `DATE` -> `Date` in indicators,
      2) Casts indicator Date to pl.Date,
      3) Left-joins indicators onto sp500 returns by `Date`.

    Args:
        sp500_returns_weekly: Wide weekly returns DataFrame with a `Date` column.
        us_indicators_weekly: Weekly indicators DataFrame with a `DATE` column.

    Returns:
        A DataFrame containing all columns from sp500_returns_weekly plus all
        indicator columns, joined on Date.
    """
    if "Date" not in sp500_returns_weekly.columns:
        raise RuntimeError(
            f"sp500_returns_weekly missing 'Date'. Got: {sp500_returns_weekly.columns}"
        )

    if "Date" not in us_indicators_weekly.columns:
        raise RuntimeError(
            f"us_indicators_weekly missing 'Date'. Got: {us_indicators_weekly.columns}"
        )

    indicators: pl.DataFrame = us_indicators_weekly.with_columns(
        pl.col(name="Date").cast(dtype=pl.Utf8).str.to_date().alias(name="Date"),
    )

    indicators: pl.DataFrame = us_indicators_weekly.with_columns(
        pl.col(name="Date").cast(dtype=pl.Utf8).str.to_date().alias(name="Date"),
    )

    returns: pl.DataFrame = sp500_returns_weekly.with_columns(
        pl.col(name="Date").cast(dtype=pl.Date).alias(name="Date"),
    )

    sp500_returns_with_indicators_weekly: pl.DataFrame = returns.join(
        other=indicators,
        on="Date",
        how="left",
    ).sort(by="Date")

    return sp500_returns_with_indicators_weekly


if __name__ == "__main__":
    us_indicators_weekly: pl.DataFrame = read_us_indicators_weekly()
    us_indicators_weekly: pl.DataFrame = drop_null_columns(df=us_indicators_weekly)
    sp500_stocks_daily: pl.DataFrame = read_sp500_stocks_daily()

    sp500_stocks_weekly: pl.DataFrame = to_sp500_stocks_weekly(
        sp500_stocks_daily=sp500_stocks_daily
    )

    sp500_returns_weekly: pl.DataFrame = compute_sp500_returns_weekly(
        sp500_stocks_weekly=sp500_stocks_weekly
    )

    sp500_returns_weekly = transpose_sector(
        sp500_returns_weekly_long=sp500_returns_weekly
    )

    sp500_returns_with_indicators_weekly: pl.DataFrame = (
        add_indicators_to_sp500_returns_weekly(
            sp500_returns_weekly=sp500_returns_weekly,
            us_indicators_weekly=us_indicators_weekly,
        )
    )

    print(sp500_returns_with_indicators_weekly.head(n=10))
