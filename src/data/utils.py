from pathlib import Path

import polars as pl


def find_project_root(start_path: Path) -> Path:
    """Find the nearest parent directory containing a `src` folder.

    Args:
        start_path: Path from which to start searching upwards.

    Returns:
        Absolute path to the project root directory.

    Raises:
        RuntimeError: If no directory containing `src` is found.
    """
    current: Path = start_path.resolve()

    for parent in [current, *current.parents]:
        if (parent / "src").is_dir():
            return parent

    raise RuntimeError("Project root not found (no 'src' directory detected).")


def drop_null_columns(df: pl.DataFrame) -> pl.DataFrame:
    """Elimina columnas que son completamente nulas o completamente cero.

    Args:
        df: DataFrame de Polars a limpiar.

    Returns:
        DataFrame sin columnas sin información real.
    """
    valid_columns: list[str] = []

    for col in df.columns:
        series: pl.Series = df[col]

        if series.null_count() == df.height:
            continue

        if series.drop_nulls().n_unique() == 1 and series.drop_nulls()[0] == 0.0:
            continue

        valid_columns.append(col)

    return df.select(valid_columns)


def forward_fill_features(
    df: pl.DataFrame,
    date_col: str = "Date",
) -> pl.DataFrame:
    """Forward-fill all feature columns over time, leaving initial nulls as 0.

    The DataFrame is first sorted by date. For every column except the date
    column:
      - forward-fill null values
      - replace remaining nulls (at the start) with 0.0

    Args:
        df: Input DataFrame.
        date_col: Name of the date column.

    Returns:
        A DataFrame with no nulls in non-date columns.
    """
    if date_col not in df.columns:
        raise RuntimeError(f"Expected '{date_col}' column. Got: {df.columns}")

    feature_cols: list[str] = [c for c in df.columns if c != date_col]

    return df.sort(date_col).with_columns([
        pl.col(name=c).fill_null(strategy="forward").fill_null(0.0)
        for c in feature_cols
    ])
