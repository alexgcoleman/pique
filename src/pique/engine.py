"""Underlying engine for handling/filtering data"""

from pathlib import Path
from typing import Callable, OrderedDict

import polars as pl

Reader = Callable[[Path], pl.LazyFrame]


_DEFAULT_CACHED_ROWS = 2000


class Engine:
    filepath: Path
    reader: Reader
    num_cached_rows: int
    frame: pl.LazyFrame
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

    def view_slice(self, offset: int, length: int | None = None) -> pl.DataFrame:
        return self.frame.slice(offset, length).collect()


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
