# import polars as pl

# type DateTimeFormat = pl.Date | pl.Datetime | pl.Time | pl.Duration


# def split_dataframe(
#     dataframe: pl.DataFrame, columns: list[str]
# ) -> tuple[pl.DataFrame | pl.Series, pl.DataFrame | pl.Series]:
#     pass

#     # TODO
#     # check that all the strings in columns are in the dataframe columns
#     # return 2 dataframes / series:
#     #   one containing the specified columns
#     #   one containing the remaining ones


# def join_dataframes(dataframes: list[pl.DataFrame | pl.Series]) -> pl.DataFrame:
#     pass

#     # TODO
#     # assert they have the same length (vertically)
#     # join the dataframes / series (in the specified order)


# def _detect_datetime_format(datetime: str) -> str:
#     pass

#     # TODO
#     # receives a string date / datetime / time value
#     # returns a string describing the format
#     # this function should support a great variety of formats


# def to_date(datetime: str) -> pl.Date:
#     pass
#     # TODO
#     # returns the datetime described in the string in date format
#     # use the _detect_datetime_format
#     # if the datetime passed contains time info just ignore it
#     # focus just in date info


# def to_datetime(
#     datetime: str, time_unit: str | None = None, timezone: str | None = None
# ) -> pl.Datetime:
#     pass
#     # TODO
#     # returns the datetime described in the string in datetime format
#     # use the _detect_datetime_format


# def to_time(datetime: str) -> pl.Time:
#     pass
#     # TODO
#     # returns the datetime described in the string in time format
#     # use the _detect_datetime_format
#     # if the datetime passed contains date info just ignore it
#     # focus just in time info


# class DateTimeModule:
#     def __init__(self, dates: pl.Series) -> None:

#         self._dates: pl.Series = dates
#         self._format: type[DateTimeFormat] = self._detect_date_format(dates=dates)
#         self._check_sorted(dates=dates)
#         self._differences: dict[pl.Duration, int] = self._count_differences(dates=dates)
#         self._regular: bool = len(self._differences) == 1
#         self._frequency: pl.Duration | None
#         self._frequency = self._differences.values()[0] if self._regular else None

#         return None

#     def _detect_date_format(
#         self,
#         dates: pl.Series,
#     ) -> type[DateTimeFormat]:

#         pass

#         # TODO
#         # assert dates is a pl.Series
#         # assert dates values are one of this types:
#         # pl.Date, pl.Datetime, pl.Datetime[ms|us|ns], pl.Datetime[ms|us|ns, tz], pl.Time, pl.Duration
#         # return de type

#     def _check_sorted(self, dates: pl.Series) -> None:

#         pass

#         # TODO
#         # assert that the series is sorted

#     def _count_differences(self, dates: pl.Series) -> dict[pl.Duration, int]:

#         pass

#         # TODO
#         # compute the differences in time of the values in the series
#         # count the appearances of each difference
#         # return the count

#     @property
#     def first(self) -> None:
#         pass
#         # TODO: return the first date time of the series (earliest)

#     @property
#     def last(self) -> None:
#         pass
#         # TODO: return the last date time of the series (latest)

#     @property
#     def leaps(self) -> dict[pl.Duration, int]:
#         return self._differences

#     def next_datetime(self, datetime: DateTimeFormat) -> None:
#         pass

#         # TODO
#         # assert the series is regular
#         # assert the datetime passed is in the same format as the series
#         # return the next value to datetime following the pattern of the series
#         #   (maintain frequency to values in the series) (not including the passed datetime)

#     def generate_compatible_series(
#         self, starting_datetime: DateTimeFormat, len: int
#     ) -> None:
#         pass

#         # TODO
#         # assert the series is regular
#         # assert the datetime passed is in the same format as the series
#         # return the the series:
#         # - starting in the first compatible value from starting_datetime (included)
#         # - with the specified length

#     def extract_dates(
#         self,
#         inferior_percentile: float = 0.0,
#         superior_percentile: float = 1.0,
#     ) -> pl.Series:

#         pass

#         # equivalent to this other function that I developped for another class
#         # just return a copy of the dataframe cropped in the specified percentiles

#         def extract_dates(
#             self,
#             inferior_percentile: float = 0.0,
#             superior_percentile: float = 1.0,
#         ) -> pl.Series:
#             """Return the subset of dates whose indices fall between two percentiles.

#             Args:
#                 inferior_percentile: Lower bound percentile in [0.0, 1.0].
#                 superior_percentile: Upper bound percentile in [0.0, 1.0].

#             Returns:
#                 A Polars Series containing the selected dates, in the original order.

#             Raises:
#                 AssertionError: If percentiles are invalid or if the resulting slice is empty.
#             """
#             assert isinstance(inferior_percentile, float)
#             assert isinstance(superior_percentile, float)
#             assert 0.0 <= inferior_percentile <= 1.0
#             assert 0.0 <= superior_percentile <= 1.0
#             assert inferior_percentile <= superior_percentile

#             n: int = int(self._dates.len())
#             assert n == self._data.shape[0]
#             assert n >= 1

#             start_idx: int = round(number=(n * inferior_percentile))
#             end_idx: int = round(number=(n * superior_percentile))

#             start_idx = max(0, min(start_idx, n))
#             end_idx = max(0, min(end_idx, n))

#             assert 0 <= start_idx <= end_idx <= n

#             selected_dates: pl.Series = self._dates.slice(
#                 offset=start_idx, length=end_idx - start_idx
#             )
#             assert selected_dates.len() == (end_idx - start_idx)

#             # If the user asked for a strict interior interval, this ensures it's not empty.
#             assert selected_dates.len() >= 1

#             return selected_dates


from __future__ import annotations

import contextlib
from collections import Counter
from dataclasses import dataclass
from datetime import date as PyDate
from datetime import datetime as PyDateTime
from datetime import time as PyTime
from datetime import timedelta
from zoneinfo import ZoneInfo

import polars as pl

type DateTimeFormat = pl.Date | pl.Datetime | pl.Time | pl.Duration


_PY_SCALAR = PyDate | PyDateTime | PyTime | timedelta


def split_dataframe(
    dataframe: pl.DataFrame, columns: list[str]
) -> tuple[pl.DataFrame | pl.Series, pl.DataFrame | pl.Series]:
    """Split a DataFrame into two parts: selected columns and the remaining columns.

    If `columns` contains exactly one column name, the first return value is a
     `pl.Series` (that column). Otherwise it is a `pl.DataFrame`.

     Args:
         dataframe: Input Polars DataFrame.
         columns: Column names to extract.

     Returns:
         A pair (selected, remaining) where each element is either a DataFrame or
         a Series, depending on the number of selected columns.

     Raises:
         AssertionError: If inputs are invalid or requested columns are missing.
    """
    assert isinstance(dataframe, pl.DataFrame)
    assert isinstance(columns, list)
    assert all(isinstance(c, str) for c in columns)

    df_cols: list[str] = dataframe.columns
    missing: list[str] = [c for c in columns if c not in df_cols]
    assert len(missing) == 0, f"Missing columns: {missing!r}"

    selected_cols: list[str] = columns
    remaining_cols: list[str] = [c for c in df_cols if c not in set(columns)]

    if len(selected_cols) == 1:
        selected: pl.Series = dataframe.get_column(name=selected_cols[0])
        remaining: pl.DataFrame = dataframe.select(remaining_cols)
        return selected, remaining

    selected_df: pl.DataFrame = dataframe.select(selected_cols)
    remaining_df: pl.DataFrame = dataframe.select(remaining_cols)
    return selected_df, remaining_df


def join_dataframes(dataframes: list[pl.DataFrame | pl.Series]) -> pl.DataFrame:
    """Horizontally concatenate DataFrames/Series preserving the given order.

    Args:
        dataframes: List of Polars DataFrames or Series.

    Returns:
        A single DataFrame with all columns appended in the specified order.

    Raises:
        AssertionError: If input is empty or lengths are inconsistent.
    """
    assert isinstance(dataframes, list)
    assert len(dataframes) >= 1
    assert all(isinstance(d, (pl.DataFrame, pl.Series)) for d in dataframes)

    lengths: list[int] = []
    for d in dataframes:
        if isinstance(d, pl.DataFrame):
            lengths.append(d.height)
        else:
            lengths.append(int(d.len()))
    assert len(set(lengths)) == 1, f"Inconsistent lengths: {lengths!r}"

    dfs: list[pl.DataFrame] = []
    for d in dataframes:
        if isinstance(d, pl.DataFrame):
            dfs.append(d)
        else:
            dfs.append(d.to_frame())
    out: pl.DataFrame = pl.concat(dfs, how="horizontal")
    assert out.height == lengths[0]
    return out


def _strip_tz_colon(tz: str) -> str:
    """Convert '+HH:MM' or '-HH:MM' into '+HHMM' or '-HHMM'."""
    assert isinstance(tz, str)
    if len(tz) == 6 and (tz[0] in {"+", "-"}) and (tz[3] == ":"):
        return f"{tz[0]}{tz[1:3]}{tz[4:6]}"
    return tz


def _normalize_datetime_string(s: str) -> str:
    """Normalize common datetime string variants to improve parsability."""
    assert isinstance(s, str)
    s2: str = s.strip()

    # Replace 'Z' with explicit UTC offset for %z parsing.
    if s2.endswith("Z") or s2.endswith("z"):
        s2 = s2[:-1] + "+0000"

    # If timezone is like +01:00, drop the colon for %z.
    if len(s2) >= 6:
        tail: str = s2[-6:]
        if (tail[0] in {"+", "-"}) and (tail[3] == ":"):
            s2 = s2[:-6] + _strip_tz_colon(tz=tail)

    return s2


def _detect_datetime_format(datetime: str) -> str:
    """Detect a strptime-compatible format string for a datetime-like input.

    This function tries a wide set of common formats:
      - Dates: YYYY-MM-DD, YYYY/MM/DD, DD-MM-YYYY, DD/MM/YYYY, YYYYMMDD
      - Times: HH:MM, HH:MM:SS, optional .ffffff
      - Datetimes: date + ('T' or ' ') + time, optional timezone (Z, ±HHMM, ±HH:MM)

    Args:
        datetime: Input string representing a date/time/datetime.

    Returns:
        A `datetime.strptime` format string.

    Raises:
        AssertionError: If input is not a string or no format matches.
    """
    assert isinstance(datetime, str)
    s: str = _normalize_datetime_string(s=datetime)

    date_fmts: list[str] = [
        "%Y-%m-%d",
        "%Y/%m/%d",
        "%d-%m-%Y",
        "%d/%m/%Y",
        "%Y%m%d",
    ]
    time_fmts: list[str] = [
        "%H:%M",
        "%H:%M:%S",
        "%H:%M:%S.%f",
    ]
    seps: list[str] = ["T", " "]
    tz_suffixes: list[str] = ["", " %z", "%z"]

    candidates: list[str] = []

    # Pure date
    candidates.extend(date_fmts)

    # Pure time
    candidates.extend(time_fmts)

    # Datetime: date + sep + time (+ tz)
    for df in date_fmts:
        for sep in seps:
            for tf in time_fmts:
                base: str = f"{df}{sep}{tf}"
                for tz in tz_suffixes:
                    candidates.append(base + tz)

    for fmt in candidates:
        try:
            _ = PyDateTime.strptime(s, fmt)
            return fmt
        except ValueError:
            continue

    # Special case: allow missing separators in some ISO variants (e.g., 2024-01-01T120000)
    iso_compact: list[tuple[str, str]] = [
        ("%Y-%m-%dT%H%M%S", "%Y-%m-%dT%H%M%S"),
        ("%Y-%m-%d %H%M%S", "%Y-%m-%d %H%M%S"),
        ("%Y%m%dT%H%M%S", "%Y%m%dT%H%M%S"),
    ]
    for raw_fmt, fmt in iso_compact:
        try:
            _ = PyDateTime.strptime(s, fmt)
            return raw_fmt
        except ValueError:
            continue

    assert False, f"Could not detect datetime format for: {datetime!r}"
    raise AssertionError  # unreachable


def _parse_datetime(
    datetime: str, time_unit: str | None = None, timezone: str | None = None
) -> PyDateTime:
    """Parse a datetime string into a Python datetime with optional timezone handling."""
    assert isinstance(datetime, str)
    assert (time_unit is None) or (time_unit in {"ms", "us", "ns"})
    assert (timezone is None) or isinstance(timezone, str)

    s: str = _normalize_datetime_string(s=datetime)
    fmt: str = _detect_datetime_format(datetime=s)

    dt: PyDateTime
    dt = PyDateTime.strptime(s, fmt)

    # If parsed contains tzinfo, keep it unless overridden.
    if timezone is not None:
        tz: ZoneInfo = ZoneInfo(timezone)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=tz)
        else:
            dt = dt.astimezone(tz)

    # time_unit affects Polars dtype when used in expressions; for scalar Python
    # datetime it does nothing. Kept for API consistency.
    _ = time_unit
    return dt


def to_date(datetime: str) -> pl.Date:
    """Parse a string and return only the date component.

    If the input contains time information, it is ignored.

    Args:
        datetime: Input string with date/datetime content.

    Returns:
        A Python `datetime.date` scalar (the scalar type Polars uses for pl.Date).

    Raises:
        AssertionError: If parsing fails.
    """
    assert isinstance(datetime, str)

    s: str = _normalize_datetime_string(s=datetime)
    fmt: str = _detect_datetime_format(datetime=s)

    # Try parse as datetime first; if it fails, try date-only formats.
    try:
        dt: PyDateTime = PyDateTime.strptime(s, fmt)
        return dt.date()  # type: ignore[return-value]
    except ValueError:
        # If detected fmt is a pure date, parse it as date.
        for df in ["%Y-%m-%d", "%Y/%m/%d", "%d-%m-%Y", "%d/%m/%Y", "%Y%m%d"]:
            try:
                d: PyDate = PyDateTime.strptime(s, df).date()
                return d  # type: ignore[return-value]
            except ValueError:
                continue

    assert False, f"Could not parse date from: {datetime!r}"
    raise AssertionError  # unreachable


def to_datetime(
    datetime: str, time_unit: str | None = None, timezone: str | None = None
) -> pl.Datetime:
    """Parse a string and return a datetime scalar.

    Args:
        datetime: Input string with date+time content.
        time_unit: One of {'ms','us','ns'}; kept for API consistency.
        timezone: IANA timezone name (e.g. 'Europe/Madrid') to attach/convert.

    Returns:
        A Python `datetime.datetime` scalar (the scalar type Polars uses for pl.Datetime).

    Raises:
        AssertionError: If parsing fails.
    """
    dt: PyDateTime = _parse_datetime(
        datetime=datetime, time_unit=time_unit, timezone=timezone
    )
    return dt  # type: ignore[return-value]


def to_time(datetime: str) -> pl.Time:
    """Parse a string and return only the time component.

    If the input contains date information, it is ignored.

    Args:
        datetime: Input string with time/datetime content.

    Returns:
        A Python `datetime.time` scalar (the scalar type Polars uses for pl.Time).

    Raises:
        AssertionError: If parsing fails.
    """
    assert isinstance(datetime, str)

    s: str = _normalize_datetime_string(s=datetime)
    fmt: str = _detect_datetime_format(datetime=s)

    # If it's a pure time, parse directly.
    for tf in ["%H:%M", "%H:%M:%S", "%H:%M:%S.%f"]:
        try:
            t: PyTime = PyDateTime.strptime(s, tf).time()
            return t  # type: ignore[return-value]
        except ValueError:
            continue

    # Otherwise parse as datetime and extract time.
    try:
        dt: PyDateTime = PyDateTime.strptime(s, fmt)
        return dt.time()  # type: ignore[return-value]
    except ValueError:
        pass

    raise AssertionError(f"Could not parse time from: {datetime!r}")


def _is_supported_datetime_dtype(dtype: pl.DataType) -> bool:
    """Return True if dtype is one of the supported temporal dtypes."""
    assert isinstance(dtype, pl.DataType)

    if dtype == pl.Date:
        return True
    if dtype == pl.Time:
        return True
    if dtype == pl.Duration:
        return True
    return bool(isinstance(dtype, pl.Datetime))


@dataclass(frozen=True)
class _TemporalSpec:
    """Internal representation for temporal arithmetic in microseconds."""

    freq_us: int
    origin_us: int


def _scalar_to_us(value: _PY_SCALAR) -> int:
    """Convert a temporal Python scalar to an integer number of microseconds."""
    assert isinstance(value, (PyDate, PyDateTime, PyTime, timedelta))

    if isinstance(value, PyDateTime):
        # Use POSIX epoch when naive: treat as "as-is" (no tz conversion).
        epoch: PyDateTime = PyDateTime(year=1970, month=1, day=1, tzinfo=value.tzinfo)
        delta: timedelta = value - epoch
        return int(delta.total_seconds() * 1_000_000)

    if isinstance(value, PyDate):
        # Days since epoch at midnight.
        epoch_date: PyDate = PyDate(year=1970, month=1, day=1)
        delta_days: int = (value - epoch_date).days
        return delta_days * 24 * 3600 * 1_000_000

    if isinstance(value, PyTime):
        return (
            value.hour * 3600 + value.minute * 60 + value.second
        ) * 1_000_000 + value.microsecond

    # timedelta
    return int(value.total_seconds() * 1_000_000)


def _us_to_scalar(value_us: int, template: _PY_SCALAR) -> _PY_SCALAR:
    """Convert microseconds back to the same Python scalar type as `template`."""
    assert isinstance(value_us, int)
    assert isinstance(template, (PyDate, PyDateTime, PyTime, timedelta))

    if isinstance(template, PyDateTime):
        epoch: PyDateTime = PyDateTime(
            year=1970, month=1, day=1, tzinfo=template.tzinfo
        )
        return epoch + timedelta(microseconds=value_us)

    if isinstance(template, PyDate):
        epoch_date: PyDate = PyDate(year=1970, month=1, day=1)
        days: int = value_us // (24 * 3600 * 1_000_000)
        return epoch_date + timedelta(days=days)

    if isinstance(template, PyTime):
        day_us: int = 24 * 3600 * 1_000_000
        v: int = value_us % day_us
        hour: int = v // (3600 * 1_000_000)
        v = v - hour * 3600 * 1_000_000
        minute: int = v // (60 * 1_000_000)
        v = v - minute * 60 * 1_000_000
        second: int = v // 1_000_000
        microsecond: int = v - second * 1_000_000
        return PyTime(hour=hour, minute=minute, second=second, microsecond=microsecond)

    return timedelta(microseconds=value_us)


class DateTimeModule:
    def __init__(self, dates: pl.Series) -> None:
        """Analyze a temporal Series: ordering, step regularity, and frequency.

        Args:
            dates: A Polars Series containing temporal values.

        Raises:
            AssertionError: If the series is not temporal or not sorted.
        """
        assert isinstance(dates, pl.Series)
        assert int(dates.len()) >= 1

        self._dates: pl.Series = dates
        self._dtype: pl.DataType = self._detect_date_format(dates=dates)
        self._check_sorted(dates=dates)
        self._differences_us: dict[int, int] = self._count_differences(dates=dates)
        self._regular: bool = len(self._differences_us) == 1
        self._frequency_us: int | None = (
            next(iter(self._differences_us.keys())) if self._regular else None
        )

    def _detect_date_format(self, dates: pl.Series) -> pl.DataType:
        """Detect and validate the Polars dtype of the series.

        Args:
            dates: Series to inspect.

        Returns:
            The series dtype.

        Raises:
            AssertionError: If dtype is not a supported temporal dtype.
        """
        assert isinstance(dates, pl.Series)
        dtype: pl.DataType = dates.dtype
        assert _is_supported_datetime_dtype(dtype=dtype), (
            f"Unsupported dtype: {dtype!r}"
        )
        return dtype

    def _check_sorted(self, dates: pl.Series) -> None:
        """Assert that the series is sorted in non-decreasing order.

        Args:
            dates: Series to validate.

        Raises:
            AssertionError: If the series is not sorted.
        """
        assert isinstance(dates, pl.Series)
        vals: list[object] = dates.to_list()
        assert all(v is not None for v in vals), "Series contains nulls."
        py_vals: list[_PY_SCALAR] = [v for v in vals if v is not None]  # type: ignore

        us_vals: list[int] = [_scalar_to_us(value=v) for v in py_vals]
        assert us_vals == sorted(us_vals), "Series is not sorted."

    def _count_differences(self, dates: pl.Series) -> dict[int, int]:
        """Count consecutive time differences (in microseconds) across the series.

        Args:
            dates: Series of temporal values.

        Returns:
            A dict mapping delta_us -> count.

        Raises:
            AssertionError: If series length < 2 or contains nulls.
        """
        assert isinstance(dates, pl.Series)
        n: int = int(dates.len())
        assert n >= 1

        if n == 1:
            return {}

        vals: list[object] = dates.to_list()
        assert all(v is not None for v in vals), "Series contains nulls."
        py_vals: list[_PY_SCALAR] = [v for v in vals if v is not None]  # type: ignore
        assert len(py_vals) == n

        us_vals: list[int] = [_scalar_to_us(value=v) for v in py_vals]
        deltas: list[int] = [us_vals[i + 1] - us_vals[i] for i in range(n - 1)]
        assert all(d > 0 for d in deltas), "Series must be strictly increasing."

        counts: Counter[int] = Counter(deltas)
        return dict(counts)

    @property
    def first(self) -> _PY_SCALAR:
        """Return the earliest value in the series."""
        vals: list[object] = self._dates.to_list()
        assert len(vals) >= 1
        first_val: object = vals[0]
        assert first_val is not None
        return first_val  # type: ignore[return-value]

    @property
    def last(self) -> _PY_SCALAR:
        """Return the latest value in the series."""
        vals: list[object] = self._dates.to_list()
        assert len(vals) >= 1
        last_val: object = vals[-1]
        assert last_val is not None
        return last_val  # type: ignore[return-value]

    @property
    def leaps(self) -> dict[int, int]:
        """Return the histogram of consecutive deltas (microseconds)."""
        return dict(self._differences_us)

    @property
    def is_regular(self) -> bool:
        """Return True if there is exactly one consecutive delta in the series."""
        return self._regular

    @property
    def frequency_us(self) -> int | None:
        """Return the regular frequency in microseconds, if regular."""
        return self._frequency_us

    def next_datetime(self, datetime: _PY_SCALAR) -> _PY_SCALAR:
        """Return the next value after `datetime` according to the series frequency.

        Args:
            datetime: A Python scalar compatible with the series (date/datetime/time/timedelta).

        Returns:
            The next temporal value (datetime + frequency).

        Raises:
            AssertionError: If series is not regular or input type is incompatible.
        """
        assert self._regular, "Series is not regular."
        assert self._frequency_us is not None
        assert isinstance(datetime, type(self.first)), "Incompatible scalar type."

        base_us: int = _scalar_to_us(value=datetime)
        nxt_us: int = base_us + self._frequency_us
        return _us_to_scalar(value_us=nxt_us, template=datetime)

    def generate_compatible_series(
        self, starting_datetime: _PY_SCALAR, len: int
    ) -> pl.Series:
        """Generate a regular series aligned with this module's frequency.

        The returned series:
          - starts at the first value >= `starting_datetime` that is aligned to the
            grid defined by `self.first` and the series frequency
          - has the specified length

        Args:
            starting_datetime: Starting point to align forward.
            len: Number of elements to generate.

        Returns:
            A Polars Series containing generated temporal values.

        Raises:
            AssertionError: If series is not regular or inputs are invalid.
        """
        assert self._regular, "Series is not regular."
        assert self._frequency_us is not None
        assert isinstance(len, int)
        assert len >= 1
        assert isinstance(starting_datetime, type(self.first)), "Incompatible scalar."

        origin: _PY_SCALAR = self.first
        origin_us: int = _scalar_to_us(value=origin)
        start_us: int = _scalar_to_us(value=starting_datetime)
        freq_us: int = self._frequency_us

        if start_us <= origin_us:
            aligned_us: int = origin_us
        else:
            offset_us: int = start_us - origin_us
            remainder: int = offset_us % freq_us
            aligned_us = (
                start_us if remainder == 0 else (start_us + (freq_us - remainder))
            )

        values: list[_PY_SCALAR] = [
            _us_to_scalar(value_us=aligned_us + i * freq_us, template=origin)
            for i in range(len)
        ]
        out: pl.Series = pl.Series(values=values)
        assert int(out.len()) == len
        return out

    def extract_dates(
        self,
        inferior_percentile: float = 0.0,
        superior_percentile: float = 1.0,
    ) -> pl.Series:
        """Return the subset of dates whose indices fall between two percentiles.

        Args:
            inferior_percentile: Lower bound percentile in [0.0, 1.0].
            superior_percentile: Upper bound percentile in [0.0, 1.0].

        Returns:
            A Polars Series containing the selected dates, in the original order.

        Raises:
            AssertionError: If percentiles are invalid or if the resulting slice is empty.
        """
        assert isinstance(inferior_percentile, float)
        assert isinstance(superior_percentile, float)
        assert 0.0 <= inferior_percentile <= 1.0
        assert 0.0 <= superior_percentile <= 1.0
        assert inferior_percentile <= superior_percentile

        n: int = int(self._dates.len())
        assert n >= 1

        start_idx: int = round(number=(n * inferior_percentile))
        end_idx: int = round(number=(n * superior_percentile))

        start_idx = max(0, min(start_idx, n))
        end_idx = max(0, min(end_idx, n))

        assert 0 <= start_idx <= end_idx <= n

        selected_dates: pl.Series = self._dates.slice(
            offset=start_idx, length=end_idx - start_idx
        )
        assert int(selected_dates.len()) == (end_idx - start_idx)
        assert int(selected_dates.len()) >= 1

        return selected_dates


if __name__ == "__main__":
    # from src.data.integrate import load_integrated_dataset

    # dataframe: pl.DataFrame = load_integrated_dataset()
    # print(dataframe.columns)

    from datetime import date as PyDate

    from src.data.integrate import load_integrated_dataset

    dataframe: pl.DataFrame = load_integrated_dataset()
    print("COLUMNS:", dataframe.columns)
    print("\nHEAD:")
    print(dataframe.head(n=3))

    # --- split/join smoke test ---
    selected_cols: list[str] = ["Date", "SP500"]
    left: pl.DataFrame | pl.Series
    right: pl.DataFrame | pl.Series
    left, right = split_dataframe(dataframe=dataframe, columns=selected_cols)

    print("\nSPLIT:")
    print("left type:", type(left), "right type:", type(right))
    if isinstance(left, pl.DataFrame):
        print("left cols:", left.columns)
    else:
        print("left name:", left.name)
    if isinstance(right, pl.DataFrame):
        print("right cols (first 5):", right.columns[:5])

    joined: pl.DataFrame = join_dataframes(dataframes=[left, right])
    print("\nJOIN:")
    print("joined shape:", joined.shape)
    assert joined.shape == dataframe.shape
    assert joined.columns == dataframe.columns

    # --- datetime parsers smoke test ---
    print("\nPARSERS:")
    samples: list[str] = [
        "2011-07-11",
        "2011/07/11",
        "11-07-2011",
        "2011-07-11T13:45:10",
        "2011-07-11 13:45:10.123456",
        "13:45",
        "13:45:10",
        "13:45:10.123456",
        "2011-07-11T13:45:10Z",
        "2011-07-11T13:45:10+01:00",
    ]
    for s in samples:
        fmt: str = _detect_datetime_format(datetime=s)
        print(f"  {s!r} -> {fmt!r}")
        # These should not raise:
        with contextlib.suppress(AssertionError):
            to_date(datetime=s)
        with contextlib.suppress(AssertionError):
            to_time(datetime=s)
        with contextlib.suppress(AssertionError):
            to_datetime(datetime=s, time_unit="us", timezone="Europe/Madrid")

    # --- DateTimeModule smoke test (weekly 'Date' column) ---
    assert "Date" in dataframe.columns
    dates: pl.Series = dataframe.get_column(name="Date")
    dtm: DateTimeModule = DateTimeModule(dates=dates)

    print("\nDATETIME MODULE:")
    print("dtype:", dates.dtype)
    print("first:", dtm.first)
    print("last:", dtm.last)
    print("is_regular:", dtm.is_regular)
    print("frequency_us:", dtm.frequency_us)
    print("leaps (top):", dict(sorted(dtm.leaps.items())[:3]))

    # next_datetime / generate_compatible_series only meaningful when regular
    if dtm.is_regular:
        first_val: _PY_SCALAR = dtm.first
        nxt: _PY_SCALAR = dtm.next_datetime(datetime=first_val)
        print("next(first):", nxt)

        start: _PY_SCALAR = first_val
        if isinstance(start, PyDate):
            start = PyDate(year=start.year, month=start.month, day=start.day + 3)

        gen: pl.Series = dtm.generate_compatible_series(starting_datetime=start, len=5)
        print("generated series head:", gen)

    # --- extract_dates test ---
    extracted: pl.Series = dtm.extract_dates(
        inferior_percentile=0.1, superior_percentile=0.12
    )
    print("\nEXTRACTED DATES:")
    print(extracted)
