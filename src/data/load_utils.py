import numpy as np
import numpy.typing as npt
import polars as pl
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from src.data.integrate import load_integrated_dataset

type TorchDataLoader = DataLoader[tuple[Tensor, ...]]
type GenericDataLoader = TorchDataLoader  # |


class DataModule:
    def __init__(
        self,
        dataframe: pl.DataFrame,
        fields: list[str],
        normalize: bool = True,
    ) -> None:
        """Hold a numeric feature matrix with optional normalization.

        Notes:
            This version is fully date-agnostic. The input dataframe is expected
            to contain only numeric feature columns, and every `field` must be
            present. All selected fields must share the same float dtype.

        Args:
            dataframe: Input Polars DataFrame containing only feature columns.
            fields: Feature column names (case-insensitive match against dataframe).
            normalize: Whether to store normalized values in memory.

        Raises:
            AssertionError: If inputs are invalid or the dataframe violates constraints.
        """
        assert isinstance(dataframe, pl.DataFrame)
        assert isinstance(fields, list)
        assert all(isinstance(f, str) for f in fields)
        assert isinstance(normalize, bool)

        self._fields: list[str] = fields
        self._normalize: bool = normalize

        self._check_dataframe(dataframe=dataframe, fields=fields)

        processed: tuple[Tensor, Tensor, Tensor]
        processed = self._process_dataframe(dataframe=dataframe, fields=fields)
        self._data: Tensor = processed[0]
        self._mean: Tensor = processed[1]
        self._variance: Tensor = processed[2]

        return None

    @staticmethod
    def _find_col_case_insensitive(df: pl.DataFrame, name: str) -> str | None:
        """Find a column name in a DataFrame ignoring case.

        Args:
            df: Polars DataFrame to search in.
            name: Target column name.

        Returns:
            The exact column name in df if found, otherwise None.
        """
        assert isinstance(df, pl.DataFrame)
        assert isinstance(name, str)

        target: str = name.casefold()
        for col in df.columns:
            if col.casefold() == target:
                return col
        return None

    @staticmethod
    def _is_float_dtype(dtype: pl.DataType) -> bool:
        """Return True if dtype is a Polars float dtype."""
        assert isinstance(dtype, pl.DataType)
        return dtype in {pl.Float16, pl.Float32, pl.Float64}

    def _check_dataframe(self, dataframe: pl.DataFrame, fields: list[str]) -> None:
        """Validate that the dataframe contains all fields and is purely numeric.

        Constraints enforced:
          - Every `field` exists in `dataframe` (case-insensitive).
          - No null values in the selected fields.
          - All selected fields share the same float dtype (Float16/32/64).

        Args:
            dataframe: Input Polars DataFrame.
            fields: Feature column names to use.

        Raises:
            AssertionError: If any constraint is violated.
        """
        assert isinstance(dataframe, pl.DataFrame)
        assert isinstance(fields, list)
        assert all(isinstance(f, str) for f in fields)
        assert len(fields) >= 1

        # 1) Check that all fields exist (case-insensitive)
        df_cols_cf: set[str] = {c.casefold() for c in dataframe.columns}
        assert all(f.casefold() in df_cols_cf for f in fields)

        # Map fields to exact column names
        field_cols: list[str] = []
        for f in fields:
            col_name: str | None = self._find_col_case_insensitive(df=dataframe, name=f)
            assert col_name is not None
            field_cols.append(col_name)

        # 2) No nulls in the selected fields
        nulls_total: int = (
            dataframe.select([
                pl.col(name=c).null_count().alias(name=c) for c in field_cols
            ])
            .sum_horizontal()
            .item()
        )
        assert nulls_total == 0, f"{nulls_total} null values found in the dataframe"

        # 3) All selected fields must share the same float dtype
        dtypes: list[pl.DataType] = [dataframe.schema[c] for c in field_cols]
        assert all(self._is_float_dtype(dtype=dt) for dt in dtypes), (
            f"All fields must be float dtypes, got: {dtypes!r}"
        )
        unique_dtypes: set[pl.DataType] = set(dtypes)
        assert len(unique_dtypes) == 1, f"Fields must share dtype, got: {dtypes!r}"

        return None

    def _process_dataframe(
        self, dataframe: pl.DataFrame, fields: list[str]
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Extract a feature matrix tensor and compute mean/variance.

        Args:
            dataframe: Input Polars DataFrame.
            fields: Feature column names to use (case-insensitive).

        Returns:
            A tuple (x, mean, variance_safe), where:
              - x is (N, F) float32 tensor (normalized if enabled)
              - mean is (F,) float32 tensor
              - variance_safe is (F,) float32 tensor with eps floor

        Raises:
            AssertionError: If inputs are invalid.
        """
        assert isinstance(dataframe, pl.DataFrame)
        assert isinstance(fields, list)
        assert all(isinstance(f, str) for f in fields)

        # Map fields to exact column names
        field_cols: list[str] = []
        for f in fields:
            col_name: str | None = self._find_col_case_insensitive(df=dataframe, name=f)
            assert col_name is not None
            field_cols.append(col_name)

        # Keep only the selected columns, cast to Float32 for torch
        df_x: pl.DataFrame = dataframe.select([
            pl.col(name=c).cast(dtype=pl.Float32) for c in field_cols
        ])
        assert df_x.height == dataframe.height
        assert df_x.width == len(fields)

        x_np: npt.NDArray[np.float32] = df_x.to_numpy().astype(
            dtype=np.float32, copy=False
        )
        x: Tensor = torch.as_tensor(data=x_np, dtype=torch.float32)
        assert x.ndim == 2
        assert x.shape[0] == dataframe.height
        assert x.shape[1] == len(fields)

        mean: Tensor = x.mean(dim=0)
        variance: Tensor = x.var(dim=0, unbiased=False)

        assert mean.ndim == 1
        assert variance.ndim == 1
        assert mean.shape[0] == len(fields)
        assert variance.shape[0] == len(fields)

        eps: float = 1e-12
        variance_safe: Tensor = torch.where(
            condition=variance > eps,
            input=variance,
            other=torch.full_like(input=variance, fill_value=eps),
        )

        if self._normalize:
            x = (x - mean) / torch.sqrt(input=variance_safe)

        return x, mean, variance_safe

    @property
    def fields(self) -> list[str]:
        """Return the feature names used by the module."""
        return self._fields

    def extract_dataframe(
        self,
        inferior_percentile: float = 0.0,
        superior_percentile: float = 1.0,
        normalized: bool = True,
    ) -> pl.DataFrame:
        """Generate a Polars DataFrame with the selected fields.

        Args:
            inferior_percentile: Lower percentile (in [0, 1]) of rows to keep.
            superior_percentile: Upper percentile (in [0, 1]) of rows to keep.
            normalized: Whether to return normalized data.

        Returns:
            Polars DataFrame containing the selected fields.

        Raises:
            AssertionError: If percentiles are invalid or slice is empty.
        """
        assert isinstance(inferior_percentile, float)
        assert isinstance(superior_percentile, float)
        assert 0.0 <= inferior_percentile <= 1.0
        assert 0.0 <= superior_percentile <= 1.0
        assert inferior_percentile < superior_percentile
        assert isinstance(normalized, bool)

        x: Tensor = (
            self._data
            if normalized
            else self.denormalize_prediction(prediction=self._data)
        )

        assert x.ndim == 2
        assert x.shape[1] == len(self._fields)

        n_rows: int = int(x.shape[0])
        start: int = int(np.floor(inferior_percentile * n_rows))
        end: int = int(np.ceil(superior_percentile * n_rows))
        start = max(0, min(start, n_rows))
        end = max(0, min(end, n_rows))
        assert start < end, "Selected percentile range produces an empty slice"

        x_slice: Tensor = x[start:end, :]
        data_np: npt.NDArray[np.float32] = x_slice.detach().cpu().numpy()

        columns: dict[str, pl.Series] = {}
        for i, field in enumerate(iterable=self._fields):
            columns[field] = pl.Series(name=field, values=data_np[:, i])

        df: pl.DataFrame = pl.DataFrame(data=columns)

        assert df.width == len(self._fields)
        assert df.height == (end - start)

        return df

    def denormalize_prediction(self, prediction: Tensor) -> Tensor:
        """Denormalize a prediction tensor using stored mean/variance.

        Args:
            prediction: A (F,) or (N, F) tensor.

        Returns:
            A tensor with the same shape as `prediction`, denormalized if enabled.

        Raises:
            AssertionError: If shapes are invalid.
        """
        assert isinstance(prediction, Tensor)
        assert prediction.ndim in (1, 2)

        if prediction.ndim == 1:
            assert prediction.shape[0] == len(self._fields)
        if prediction.ndim == 2:
            assert prediction.shape[1] == len(self._fields)

        if not self._normalize:
            return prediction

        mean: Tensor = self._mean
        variance: Tensor = self._variance

        assert mean.ndim == 1 and variance.ndim == 1
        assert mean.shape[0] == len(self._fields)
        assert variance.shape[0] == len(self._fields)

        std: Tensor = torch.sqrt(input=variance)

        if prediction.ndim == 1:
            out: Tensor = prediction * std + mean
        else:
            out = prediction * std.unsqueeze(dim=0) + mean.unsqueeze(dim=0)

        assert out.shape == prediction.shape
        return out

    def denormalize_dataframe(self, dataframe: pl.DataFrame) -> pl.DataFrame:
        """Denormalize the given DataFrame using this module's stored mean/variance.

        This method only denormalizes columns present in `dataframe` that correspond
        (case-insensitively) to `self._fields`. Columns not in `self._fields` are
        not allowed (except an optional `date` column, case-insensitive), and will
        raise an assertion error.

        Args:
            dataframe: Polars DataFrame containing a subset of the module fields
                (optionally including a date column).

        Returns:
            A new Polars DataFrame with the same columns as `dataframe`, but with
            the feature columns denormalized if normalization is enabled.
        """
        assert isinstance(dataframe, pl.DataFrame)

        field_to_idx: dict[str, int] = {
            f.casefold(): i for i, f in enumerate(iterable=self._fields)
        }
        assert len(field_to_idx) == len(self._fields)

        assert all(c.casefold() in set(field_to_idx.keys()) for c in dataframe.columns)

        if not self._normalize:
            return dataframe

        mean: Tensor = self._mean
        variance: Tensor = self._variance
        assert mean.ndim == 1 and variance.ndim == 1
        assert mean.shape[0] == len(self._fields)
        assert variance.shape[0] == len(self._fields)

        std: Tensor = torch.sqrt(input=variance)
        mean_np: npt.NDArray[np.float32] = (
            mean.detach().cpu().numpy().astype(dtype=np.float32)
        )
        std_np: npt.NDArray[np.float32] = (
            std.detach().cpu().numpy().astype(dtype=np.float32)
        )

        exprs: list[pl.Expr] = []
        for col in dataframe.columns:
            idx: int = field_to_idx[col.casefold()]
            exprs.append(
                (
                    pl.col(name=col) * pl.lit(value=float(std_np[idx]))
                    + pl.lit(value=float(mean_np[idx]))
                ).alias(name=col)
            )

        out: pl.DataFrame = dataframe.with_columns(*exprs)
        assert out.columns == dataframe.columns
        assert out.height == dataframe.height

        return out


class _SlidingWindowDataset(Dataset[tuple[Tensor, Tensor]]):
    """Dataset of (x, y) sliding windows built from a dense time-series tensor.

    Each sample is:
      - x: (L, C) window ending at time t-1
      - y: (H, K) horizon starting at time t, with columns selected by output_mask

    Where:
      - C is number of dataframe columns
      - K is number of True entries in output_mask
    """

    def __init__(
        self,
        *,
        data: Tensor,
        output_mask: Tensor,
        lookback: int,
        horizon: int,
        slide: int,
    ) -> None:
        """Initialize the dataset.

        Args:
            data: Dense time-series tensor of shape (T, C).
            output_mask: Boolean tensor of shape (C,) selecting output columns.
            lookback: Lookback window length L.
            horizon: Prediction horizon length H.
            slide: Step between consecutive samples (in time steps).
        """
        super().__init__()
        self._data: Tensor = data
        self._output_mask: Tensor = output_mask
        self._lookback: int = lookback
        self._horizon: int = horizon
        self._slide: int = slide

        assert isinstance(self._slide, int)
        assert self._slide > 0

        t_total: int = int(self._data.shape[0])
        max_start: int = t_total - self._lookback - self._horizon
        self._n_samples: int = (max_start // self._slide) + 1
        assert self._n_samples > 0, (
            "Not enough rows to build any sample: need at least "
            f"lookback + horizon = {self._lookback + self._horizon} rows."
        )

        # Precompute the integer indices of selected output columns.
        self._out_idx: Tensor = torch.nonzero(
            input=self._output_mask, as_tuple=False
        ).flatten()
        assert self._out_idx.numel() > 0

    def __len__(self) -> int:
        """Return the number of available sliding-window samples."""
        return self._n_samples

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        """Get one (x, y) sample pair.

        Args:
            index: Sample index in [0, len(self)).

        Returns:
            Tuple (x, y) where:
              - x has shape (L, C)
              - y has shape (H, K)
        """
        assert isinstance(index, int)
        assert 0 <= index < self._n_samples

        start_x: int = index * self._slide
        end_x: int = start_x + self._lookback
        start_y: int = end_x
        end_y: int = end_x + self._horizon

        x: Tensor = self._data[start_x:end_x, :]
        y_full: Tensor = self._data[start_y:end_y, :]
        y: Tensor = y_full.index_select(dim=1, index=self._out_idx)

        # x: (L, C), y: (H, K)
        return x, y


def generate_torch_dataloader(
    dataframe: pl.DataFrame,
    output_mask: Tensor,
    batch: int,
    lookback: int,
    horizon: int,
    slide: int = 1,
    drop_last: bool = True,
    shuffle: bool = True,
) -> TorchDataLoader:
    """Create a single PyTorch DataLoader returning (x, y) sliding-window samples.

    The DataLoader yields:
      - x: Tensor of shape (B, L, C), where C == len(dataframe.columns)
      - y: Tensor of shape (B, H, K), where K == int(output_mask.sum())

    Args:
        dataframe: Input time-series as a Polars DataFrame of shape (T, C).
        output_mask: Boolean torch tensor selecting which columns are predicted.
        batch: Batch size B.
        lookback: Lookback window length L.
        horizon: Prediction horizon length H.
        slide: Step between consecutive samples (in time steps).
        drop_last: Whether to drop the last incomplete batch.
        shuffle: Whether to shuffle samples each epoch.

    Returns:
        A DataLoader yielding batches of (x, y).

    Notes:
        output_mask denotes which values (columns) of the input tensor are meant
        to be in the predicted output vector.
    """
    assert isinstance(batch, int)
    assert batch > 0
    assert isinstance(lookback, int)
    assert lookback > 0
    assert isinstance(horizon, int)
    assert horizon > 0
    assert isinstance(slide, int)
    assert slide > 0
    assert isinstance(drop_last, bool)
    assert isinstance(shuffle, bool)

    assert isinstance(dataframe, pl.DataFrame)
    assert dataframe.height > 0
    assert dataframe.width > 0

    # Check that all column dtypes of the dataframe are floats.
    float_dtypes: set[pl.DataType] = {pl.Float32, pl.Float64}
    for name in dataframe.columns:
        dtype: pl.DataType = dataframe.schema[name]
        assert dtype in float_dtypes, (
            f"Column '{name}' must be float32/float64, got {dtype}."
        )

    # Check that output mask is a boolean torch tensor.
    assert isinstance(output_mask, Tensor)
    assert output_mask.dtype == torch.bool, "output_mask must be torch.bool."

    # Check that output mask is a 1-d tensor with same length as dataframe columns.
    assert output_mask.ndim == 1, "output_mask must be 1-D."
    n_cols: int = int(dataframe.width)
    assert int(output_mask.numel()) == n_cols, (
        "output_mask length must match number of dataframe columns: "
        f"{int(output_mask.numel())} vs {n_cols}."
    )

    # Check that output mask has at least 1 True value.
    n_true: int = int(output_mask.sum().item())
    assert n_true >= 1, "output_mask must contain at least one True entry."

    # Convert dataframe to a dense torch tensor of shape (T, C).
    # to_numpy() returns a 2D array with columns in dataframe order.
    data_np: npt.NDArray[np.float32] = dataframe.to_numpy()
    data: Tensor = torch.tensor(data=data_np, dtype=torch.float32)
    assert data.ndim == 2
    assert int(data.shape[1]) == n_cols

    dataset: Dataset[tuple[Tensor, Tensor]] = _SlidingWindowDataset(
        data=data,
        output_mask=output_mask,
        lookback=lookback,
        horizon=horizon,
        slide=slide,
    )

    loader: TorchDataLoader = DataLoader(
        dataset=dataset,
        batch_size=batch,
        shuffle=shuffle,
        drop_last=drop_last,
    )
    return loader


def aggregate_lags(
    dataframe: pl.DataFrame,
    fields: list[str],
    lags: list[int],
    fill_method: str | None = None,
) -> pl.DataFrame:
    """Append lagged versions of specified columns to the end of a DataFrame.

    For each `field` in `fields` and each `lag` in `lags`, a new column named
    `<field>_lag<lag>` is created as `field` shifted down by `lag` rows.

    Args:
        dataframe: Input Polars DataFrame.
        fields: Column names to lag (case-insensitive match against dataframe columns).
        lags: Positive integer lags (number of rows to shift).
        fill_method: Strategy to handle rows with nulls introduced by lagging.
            - "cut": drop rows containing any nulls
            - "zero": fill nulls with 0.0
            - "avg": fill nulls with column mean
            - "first": fill nulls with the first non-null value of the column
            - None: leave nulls as-is

    Returns:
        A new Polars DataFrame with the lag columns appended (and optionally filled).
    """
    assert isinstance(fields, list)
    assert all(isinstance(f, str) for f in fields)
    assert isinstance(lags, list)
    assert all(isinstance(l, int) for l in lags)
    assert all(l > 0 for l in lags)
    assert fill_method in ("cut", "zero", "avg", "first", None)

    columns_lower: dict[str, str] = {c.lower(): c for c in dataframe.columns}
    assert all(f.lower() in columns_lower for f in fields)

    normalized_fields: list[str] = [columns_lower[f.lower()] for f in fields]

    lag_exprs: list[pl.Expr] = [
        pl.col(name=field).shift(n=lag).alias(name=f"{field}_lag{lag}")
        for field in normalized_fields
        for lag in lags
    ]

    result: pl.DataFrame = dataframe.with_columns(*lag_exprs)

    if fill_method is None:
        return result

    if fill_method == "cut":
        return result.drop_nulls()

    if fill_method == "zero":
        return result.fill_null(value=0.0)

    if fill_method == "avg":
        fill_exprs: list[pl.Expr] = []
        for col in result.columns:
            mean_value: float | None = result.select(pl.col(name=col).mean()).item()
            if mean_value is not None:
                fill_exprs.append(
                    pl.col(name=col).fill_null(value=float(mean_value)).alias(col)
                )
        return result.with_columns(*fill_exprs)

    if fill_method == "first":
        fill_exprs = []
        for col in result.columns:
            first_value: float | None = result.select(
                pl.col(name=col).drop_nulls().first()
            ).item()
            if first_value is not None:
                fill_exprs.append(
                    pl.col(name=col).fill_null(value=float(first_value)).alias(col)
                )
        return result.with_columns(*fill_exprs)

    # unreachable due to assertion
    return result


def load_datamodule_2010to2024() -> DataModule:

    sp500_returns_with_indicators_weekly: pl.DataFrame = load_integrated_dataset()
    all_fields: list[str] = sp500_returns_with_indicators_weekly.columns

    metrics: list[str] = ["SP500"]
    sectors: list[str] = [
        "Basic Materials",
        "Communication Services",
        "Consumer Cyclical",
        "Consumer Defensive",
        "Energy",
        "Financial Services",
        "Healthcare",
        "Industrials",
        "Real Estate",
        "Technology",
        "Utilities",
    ]
    features: list[str] = [
        f
        for f in all_fields
        if f not in metrics and f not in sectors and f.lower() != "date"
    ]

    fields: list[str] = [*metrics, *sectors, *features]
    data_module: DataModule = DataModule(
        dataframe=sp500_returns_with_indicators_weekly,
        fields=fields,
        normalize=True,
    )

    return data_module


if __name__ == "__main__":
    data_module: DataModule = load_datamodule_2010to2024()
    dataframe: pl.DataFrame = data_module.extract_dataframe(
        inferior_percentile=0.1, superior_percentile=0.9
    )

    print("\nDATAFRAME")
    print(dataframe.head(n=5))

    original_dataframe: pl.DataFrame = data_module.extract_dataframe(
        inferior_percentile=0.1, superior_percentile=0.9, normalized=False
    )
    denormalized_dataframe: pl.DataFrame = data_module.denormalize_dataframe(
        dataframe=dataframe
    )

    print("\nDENORMALIZED DATAFRAME")
    print("original dataframe:")
    print(original_dataframe.head(n=5))
    print("denormalized dataframe:")
    print(denormalized_dataframe.head(n=5))

    lagged_fields: list[str] = ["SP500"]
    lagged_dataframe: pl.DataFrame = aggregate_lags(
        dataframe=dataframe, fields=lagged_fields, lags=[1, 2], fill_method="cut"
    )

    print("\nLAGGED DATAFRAME")
    print(lagged_dataframe.head(n=5))

    # test DataLoader generation:
    output_mask: Tensor = torch.tensor(
        data=[*(True for _ in range(12)), *(False for _ in range(19))]
    )
    dataloader: TorchDataLoader = generate_torch_dataloader(
        dataframe=dataframe,
        output_mask=output_mask,
        batch=32,
        lookback=10,
        horizon=1,
        slide=1,
        drop_last=True,
        shuffle=True,
    )

    print("\nDATALOADER:")
    print("dataframe shape:", dataframe.shape)
    print("dataloader batches:", len(dataloader))
    print("dataloader batch size", dataloader.batch_size)
