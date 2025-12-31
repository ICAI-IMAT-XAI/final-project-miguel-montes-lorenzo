import numpy as np
import numpy.typing as npt
import polars as pl
import torch
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset

from src.data.join import (
    add_indicators_to_sp500_returns_weekly,
    compute_sp500_returns_weekly,
    to_sp500_stocks_weekly,
    transpose_sector,
)
from src.data.read import (
    drop_null_columns,
    read_sp500_stocks_daily,
    read_us_indicators_weekly,
)
from src.data.utils import forward_fill_features

type TorchDataLoader = DataLoader[tuple[Tensor, ...]]


def load_raw_dataset() -> pl.DataFrame:

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

    sp500_returns_with_indicators_weekly = forward_fill_features(
        df=sp500_returns_with_indicators_weekly,
        date_col="Date",
    )

    return sp500_returns_with_indicators_weekly


class DataModule:
    def __init__(
        self,
        dataframe: pl.DataFrame,
        fields: list[str],
        normalize: bool = True,
        frequency_regular: bool = True,
    ) -> None:
        assert isinstance(dataframe, pl.DataFrame)
        assert isinstance(fields, list)
        assert all(isinstance(f, str) for f in fields)
        assert isinstance(normalize, bool)
        assert isinstance(frequency_regular, bool)

        self._fields: list[str] = fields
        self._normalize: bool = normalize
        self._freqency: int | None = self._check_dataframe(
            dataframe=dataframe, fields=fields, frequency_regular=frequency_regular
        )

        processed_dataframe: tuple[pl.Series, Tensor, Tensor, Tensor]
        processed_dataframe = self._process_dataframe(
            dataframe=dataframe, fields=fields
        )
        self._dates: pl.Series = processed_dataframe[0]
        self._data: Tensor = processed_dataframe[1]
        self._mean: Tensor = processed_dataframe[2]
        self._variance: Tensor = processed_dataframe[3]

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

    def _check_dataframe(
        self, dataframe: pl.DataFrame, fields: list[str], frequency_regular: bool = True
    ) -> int | None:
        assert isinstance(dataframe, pl.DataFrame)
        assert isinstance(fields, list)
        assert all(isinstance(f, str) for f in fields)
        assert isinstance(frequency_regular, bool)

        # 0. comprobar que "date" no está en fields (case insensitive)
        fields_cf: list[str] = [f.casefold() for f in fields]
        assert "date" not in fields_cf, "date column missing in dataframe"

        # 2. comprobar que df tiene un campo llamado "date" (case insensitive)
        date_col: str | None = self._find_col_case_insensitive(
            df=dataframe, name="date"
        )
        assert date_col is not None

        # 1. comprobar que todos los fields están en df (case insensitive)
        df_cols_cf: set[str] = {c.casefold() for c in dataframe.columns}
        assert all(f.casefold() in df_cols_cf for f in fields)

        # 3. comprobar que no hay valores nulos en el dataframe
        # (en todas las columnas relevantes: date + fields)
        cols_needed: list[str] = [date_col] + [
            self._find_col_case_insensitive(df=dataframe, name=f) for f in fields
        ]  # type: ignore
        assert all(c is not None for c in cols_needed), "found null column"
        cols_needed2: list[str]
        cols_needed2 = [c for c in cols_needed if c is not None]  # type: ignore

        nulls_total: int = (
            dataframe.select([
                pl.col(name=c).null_count().alias(name=c) for c in cols_needed2
            ])
            .sum_horizontal()
            .item()
        )
        assert nulls_total == 0, f"{nulls_total} null values found in the dataframe"

        # comprobar tipos/orden de fechas
        df_sorted: pl.DataFrame = dataframe.sort(by=date_col)
        date_s: pl.Series = df_sorted.get_column(name=date_col)
        assert date_s.len() == df_sorted.height
        assert date_s.is_sorted()

        # aceptar pl.Date o pl.Datetime;
        # si es Datetime, convertir a Date para el diff en días
        if date_s.dtype == pl.Datetime:
            date_s = date_s.dt.date()
        assert date_s.dtype == pl.Date

        # 4. comprobar frecuencia regular (si procede)
        if not frequency_regular:
            return None

        assert df_sorted.height >= 2

        diffs: pl.Series = date_s.diff().drop_nulls()
        # diffs es Duration (en ns); convertimos a días enteros
        diffs_days: pl.Series = diffs.dt.total_days()
        assert diffs_days.len() == df_sorted.height - 1

        unique_diffs: list[int] = diffs_days.unique().to_list()
        assert len(unique_diffs) == 1

        freq_days: int = int(unique_diffs[0])
        assert freq_days > 0
        return freq_days

    def _process_dataframe(
        self, dataframe: pl.DataFrame, fields: list[str]
    ) -> tuple[pl.Series, Tensor, Tensor, Tensor]:
        assert isinstance(dataframe, pl.DataFrame)
        assert isinstance(fields, list)
        assert all(isinstance(f, str) for f in fields)

        date_col: str | None = self._find_col_case_insensitive(
            df=dataframe, name="date"
        )
        assert date_col is not None

        # map fields a nombres exactos del df (respetando mayúsculas/minúsculas)
        field_cols: list[str] = []
        for f in fields:
            col_name: str | None = self._find_col_case_insensitive(df=dataframe, name=f)
            assert col_name is not None
            field_cols.append(col_name)

        df_sorted: pl.DataFrame = dataframe.sort(by=date_col)

        dates: pl.Series = df_sorted.get_column(name=date_col)
        if dates.dtype == pl.Datetime:
            dates = dates.dt.date()
        assert dates.dtype == pl.Date
        assert dates.is_sorted()

        # asegurar float32/float64 para tensor
        df_x: pl.DataFrame = df_sorted.select([
            pl.col(name=c).cast(dtype=pl.Float32) for c in field_cols
        ])
        assert df_x.height == df_sorted.height
        assert df_x.width == len(fields)

        x_np = df_x.to_numpy()
        x: Tensor = torch.as_tensor(data=x_np, dtype=torch.float32)
        assert x.ndim == 2
        assert x.shape[0] == df_sorted.height
        assert x.shape[1] == len(fields)

        # medias y varianzas por feature
        mean: Tensor = x.mean(dim=0)
        variance: Tensor = x.var(dim=0, unbiased=False)

        assert mean.ndim == 1
        assert variance.ndim == 1
        assert mean.shape[0] == len(fields)
        assert variance.shape[0] == len(fields)

        # evitar divisiones por 0: var == 0 implica feature constante
        eps: float = 1e-12
        variance_safe: Tensor = torch.where(
            condition=variance > eps,
            input=variance,
            other=torch.full_like(input=variance, fill_value=eps),
        )

        if self._normalize:
            x = (x - mean) / torch.sqrt(input=variance_safe)

        return dates, x, mean, variance_safe

    @property
    def fields(self) -> list[str]:
        return self._fields

    @property
    def dates(self) -> pl.Series:
        return self._dates

    def spit_dates(self, split_proportion: float) -> tuple[pl.Series, pl.Series]:
        assert isinstance(split_proportion, float)
        assert 0.0 < split_proportion < 1.0

        n: int = int(self._dates.len())
        assert n == self._data.shape[0]
        assert n >= 2

        n_train: int = round(number=(n * split_proportion))
        # asegurar al menos 1 en cada split
        n_train = max(1, min(n_train, n - 1))
        n_test: int = n - n_train
        assert n_test >= 1

        train_dates: pl.Series = self._dates.head(n=n_train)
        test_dates: pl.Series = self._dates.tail(n=n_test)

        assert train_dates.len() == n_train
        assert test_dates.len() == n_test
        assert train_dates.max() < test_dates.min()

        return train_dates, test_dates

    def generate_torch_dataloaders(
        self,
        batch: int,
        drop_last: bool = True,
        split_proportion: float = 0.8,
        shuffle_train: bool = True,
    ) -> tuple[TorchDataLoader, TorchDataLoader]:
        assert isinstance(batch, int)
        assert batch > 0
        assert isinstance(drop_last, bool)
        assert isinstance(split_proportion, float)
        assert 0.0 < split_proportion < 1.0
        assert isinstance(shuffle_train, bool)

        n: int = int(self._data.shape[0])
        assert n == self._dates.len()
        assert self._data.ndim == 2
        assert self._data.shape[1] == len(self._fields)

        n_train: int = round(number=n * split_proportion)
        n_train = max(1, min(n_train, n - 1))
        n_test: int = n - n_train
        assert n_test >= 1

        x_train: Tensor = self._data[:n_train]
        x_test: Tensor = self._data[n_train:]

        assert x_train.shape[0] == n_train
        assert x_test.shape[0] == n_test
        assert x_train.shape[1] == len(self._fields)
        assert x_test.shape[1] == len(self._fields)

        train_ds: TensorDataset = TensorDataset(x_train)
        test_ds: TensorDataset = TensorDataset(x_test)

        train_loader: TorchDataLoader = DataLoader(
            dataset=train_ds,
            batch_size=batch,
            shuffle=shuffle_train,
            drop_last=drop_last,
        )
        test_loader: TorchDataLoader = DataLoader(
            dataset=test_ds,
            batch_size=batch,
            shuffle=False,
            drop_last=drop_last,
        )

        return train_loader, test_loader

    def generate_dataframe(
        self, keep_date: bool = True, normalized: bool = True
    ) -> pl.DataFrame:
        """Generate a Polars DataFrame with the selected fields.

        Args:
            keep_date: Whether to include the date column.
            normalized: Whether to return normalized data.

        Returns:
            Polars DataFrame containing the selected fields (and optionally date).
        """
        assert isinstance(keep_date, bool)
        assert isinstance(normalized, bool)

        x: Tensor
        x = self._data if normalized else self.unnormalize_series(prediction=self._data)

        assert x.ndim == 2
        assert x.shape[0] == self._dates.len()
        assert x.shape[1] == len(self._fields)

        data_np: npt.NDArray[np.float32] = x.detach().cpu().numpy()

        columns: dict[str, pl.Series] = {}
        for i, field in enumerate(iterable=self._fields):
            columns[field] = pl.Series(
                name=field,
                values=data_np[:, i],
            )

        if keep_date:
            columns = {"date": self._dates, **columns}

        df: pl.DataFrame = pl.DataFrame(data=columns)

        expected_cols: int = len(self._fields) + (1 if keep_date else 0)
        assert df.width == expected_cols
        assert df.height == self._dates.len()

        return df

    def unnormalize_series(self, prediction: Tensor) -> Tensor:
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


def load_datamodule_complete() -> DataModule:

    sp500_returns_with_indicators_weekly: pl.DataFrame = load_raw_dataset()
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
        frequency_regular=True,
    )

    return data_module


if __name__ == "__main__":
    data_module: DataModule = load_datamodule_complete()
    dataframe: pl.DataFrame = data_module.generate_dataframe()
    print(dataframe.head(n=10))
