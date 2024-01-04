"""Underlying engine for handling/filtering data"""

import polars as pl
from pathlib import Path
from rich import print

from typing import Callable
from enum import Enum


Reader = Callable[[Path], pl.DataFrame]


def read_csv(path: Path) -> pl.DataFrame:
    return pl.read_csv(path, try_parse_dates=True)


def read_parquet(path: Path) -> pl.DataFrame:
    return pl.read_parquet(path)


def reader(path: Path) -> pl.DataFrame:
    READERS: dict[str, Reader] = {
        "csv": read_csv,
        "pqt": read_parquet,
        "parquet": read_parquet,
    }

    extension = path.suffix.lower().lstrip(".")

    try:
        return READERS[extension](path)
    except KeyError:
        raise ValueError(f"The file extension {extension} is not supported")
