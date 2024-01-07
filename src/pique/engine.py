"""Underlying engine for handling/filtering data"""

from functools import reduce
from logging import getLogger
from pathlib import Path
from typing import Callable, OrderedDict

import polars as pl

Reader = Callable[[Path], pl.LazyFrame]


_DEFAULT_CACHED_ROWS = 300

log = getLogger(__name__)


class CachedData:
    lazy_frame: pl.LazyFrame
    num_cached_rows: int
    cached_row_start: int
    data: pl.DataFrame

    def __init__(
        self, lazy_frame: pl.LazyFrame, num_cached_rows: int, cached_row_start: int = 0
    ):
        self.lazy_frame = lazy_frame
        self.num_cached_rows = num_cached_rows
        self.cached_row_start = cached_row_start
        self.update_cache()

    @property
    def cached_row_end(self) -> int:
        return self.cached_row_start + self.num_cached_rows

    def is_row_in_cache(self, row_idx: int) -> bool:
        return (row_idx >= self.cached_row_start) and (row_idx <= self.cached_row_end)

    def is_slice_in_cache(self, offset: int, length: int) -> bool:
        start_in_cache = self.is_row_in_cache(offset)

        end_in_cache = self.is_row_in_cache(offset + length)
        return start_in_cache and end_in_cache

    def cache_start_for_row(self, row: int) -> int:
        """Calculates the best cache_start to center the cache around a row"""
        return max(row - (self.num_cached_rows // 2), 0)

    def update_cache(
        self, cached_row_start: int | None = None, num_cached_rows: int | None = None
    ) -> None:
        self.cached_row_start = cached_row_start or self.cached_row_start
        self.num_cached_rows = num_cached_rows or self.num_cached_rows

        log.info(f"UPDATE CACHE: {self.cached_row_start=}, {self.num_cached_rows=}")
        self.data = self.lazy_frame.slice(
            offset=self.cached_row_start, length=self.num_cached_rows
        ).collect()

    def cache_relative_offset(self, offset: int) -> int:
        """Calculates a cache-relative offset from a frame-relative offset"""
        return offset - self.cached_row_start

    def view_slice(self, offset: int, length: int) -> pl.DataFrame:
        log.info(f"VIEW_SLICE {offset=}, {length=}")
        if not self.is_slice_in_cache(offset=offset, length=length):
            log.warn(
                f"CACHE MISS: {offset=}, {length=}, {self.cached_row_start=}, {self.num_cached_rows=}"
            )
            # cache miss - ensure we cache at least the lenght of the requested slice
            self.num_cached_rows = max(self.num_cached_rows, length * 2)
            self.update_cache(
                cached_row_start=self.cache_start_for_row(offset),
            )

        log.info(
            f"CACHE HIT: {offset=}, {length=}, {self.cached_row_start=}, {self.num_cached_rows=}"
        )
        log.info(f"RELATIVE_OFFSET: {self.cache_relative_offset(offset)=}")
        return self.data.slice(offset=self.cache_relative_offset(offset), length=length)


class Engine:
    filepath: Path
    reader: Reader
    num_cached_rows: int
    frame: pl.LazyFrame
    cache: CachedData
    schema: OrderedDict
    columns: list
    row_count: int

    def __init__(
        self,
        filepath: Path | str,
        reader: Reader | None = None,
        num_cached_rows: int = _DEFAULT_CACHED_ROWS,
    ):
        if isinstance(filepath, str):
            filepath = Path(filepath)

        if reader is None:
            reader = auto_reader

        frame = reader(filepath)

        self.filepath = filepath
        self.reader = reader
        self.frame = frame
        self.schema = frame.schema
        self.columns = frame.columns
        self.dtypes = frame.dtypes
        self.row_count = lazy_row_count(frame)
        self.num_cached_rows = min(num_cached_rows, self.row_count)
        self.cache = CachedData(lazy_frame=frame, num_cached_rows=self.num_cached_rows)

    def view_slice(self, offset: int, length: int) -> pl.DataFrame:
        return self.cache.view_slice(offset, length)

    def column_stats(self) -> pl.DataFrame:
        """per-column statistics (number of unique values, nulls, etc)"""

        lf = self.frame

        def _transpose_result(
            frame: pl.LazyFrame | pl.DataFrame, result_name: str
        ) -> pl.DataFrame:
            if isinstance(frame, pl.LazyFrame):
                frame = frame.collect()

            return frame.transpose(
                include_header=True,
                header_name="column_name",
                column_names=[result_name],
            )

        dtypes = pl.DataFrame(
            {str(name): str(dtype) for name, dtype in lf.schema.items()}
        ).pipe(_transpose_result, result_name="dtype")

        n_unique = lf.select(pl.all().n_unique()).pipe(
            _transpose_result, result_name="n_unique"
        )
        null_count = lf.select(pl.all().null_count()).pipe(
            _transpose_result, result_name="null_count"
        )
        min_value = lf.select(pl.all().min()).pipe(
            _transpose_result, result_name="min_value"
        )
        max_value = lf.select(pl.all().max()).pipe(
            _transpose_result, result_name="max_value"
        )

        result = reduce(
            lambda x, y: x.join(
                y, on="column_name", how="outer_coalesce", validate="1:1"
            ),
            [dtypes, n_unique, null_count, min_value, max_value],
        )

        return result


def lazy_read_csv(path: Path) -> pl.LazyFrame:
    return pl.scan_csv(path, try_parse_dates=True)


def lazy_read_parquet(path: Path) -> pl.LazyFrame:
    return pl.scan_parquet(path)


def auto_reader(path: Path) -> pl.LazyFrame:
    READERS: dict[str, Reader] = {
        ".csv": lazy_read_csv,
        ".pqt": lazy_read_parquet,
        ".parquet": lazy_read_parquet,
    }

    extension = path.suffix.lower()

    try:
        return READERS[extension](path)
    except KeyError:
        raise ValueError(f"The file extension {extension} is not supported")


def lazy_row_count(lf: pl.LazyFrame) -> int:
    """Get the row count of a LazyFrame"""
    return lf.select(pl.count()).collect().item()
