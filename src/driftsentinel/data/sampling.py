
"""Memory-conscious sampling utilities for experiments."""

from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq


def read_parquet_sample(
    path: str | Path,
    n_rows: int,
) -> pd.DataFrame:
    """Read at most n_rows from a Parquet file in batches."""

    if n_rows <= 0:
        raise ValueError("n_rows must be positive.")

    parquet_file = pq.ParquetFile(path)

    if parquet_file.metadata.num_rows < n_rows:
        raise ValueError(
            f"Requested {n_rows} rows, but file contains "
            f"{parquet_file.metadata.num_rows}."
        )

    batches = parquet_file.iter_batches(
        batch_size=n_rows,
    )

    batch = next(batches)

    return batch.to_pandas()
