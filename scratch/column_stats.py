from functools import reduce
from pathlib import Path

import polars as pl
from pique.engine import Engine

sample_path = (
    Path(__file__).parent
    / "../sample_data/property-transfer-statistics-june-2023-quarter.pqt"
)

engine = Engine(sample_path)


def column_stats(lf: pl.LazyFrame) -> pl.DataFrame:
    """per-column statistics (number of unique values, nulls, etc)"""

    def _transpose_result(
        frame: pl.LazyFrame | pl.DataFrame, result_name: str
    ) -> pl.DataFrame:
        if isinstance(frame, pl.LazyFrame):
            frame = frame.collect()

        return frame.transpose(
            include_header=True, header_name="column_name", column_names=[result_name]
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
        lambda x, y: x.join(y, on="column_name", how="outer_coalesce", validate="1:1"),
        [dtypes, n_unique, null_count, min_value, max_value],
    )

    return result
