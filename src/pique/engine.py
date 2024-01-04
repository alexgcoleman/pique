"""Underlying engine for handling/filtering data"""

from pathlib import Path
from typing import Callable

import polars as pl

Reader = Callable[[Path], pl.LazyFrame]


def lazy_read_csv(path: Path) -> pl.LazyFrame:
    return pl.scan_csv(path, try_parse_dates=True)


def lazy_read_parquet(path: Path) -> pl.LazyFrame:
    return pl.scan_parquet(path)


def reader(path: Path) -> pl.LazyFrame:
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
