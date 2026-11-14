
from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class TemporalSplit:
    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame


def chronological_split(
    df: pd.DataFrame,
    timestamp_column: str = "TransactionDT",
    train_ratio: float = 0.70,
    validation_ratio: float = 0.15,
) -> TemporalSplit:
    """
    Split labelled data chronologically into train, validation and test.

    Rows with equal timestamps are kept in the same partition.
    The input DataFrame is not modified.
    """

    if not 0 < train_ratio < 1:
        raise ValueError("train_ratio must be between 0 and 1.")

    if not 0 < validation_ratio < 1:
        raise ValueError("validation_ratio must be between 0 and 1.")

    if train_ratio + validation_ratio >= 1:
        raise ValueError(
            "train_ratio + validation_ratio must be less than 1."
        )

    if timestamp_column not in df.columns:
        raise ValueError(
            f"Missing timestamp column: {timestamp_column}"
        )

    if df.empty:
        raise ValueError("Cannot split an empty DataFrame.")

    if df[timestamp_column].isna().any():
        raise ValueError("Timestamp column contains missing values.")

    ordered = df.sort_values(
        by=timestamp_column,
        kind="mergesort",
    ).reset_index(drop=True)

    n_rows = len(ordered)

    train_boundary = int(n_rows * train_ratio)
    validation_boundary = int(
        n_rows * (train_ratio + validation_ratio)
    )

    timestamps = ordered[timestamp_column].to_numpy()

    # Move boundaries forward to avoid splitting equal timestamps.
    if train_boundary < n_rows:
        train_time = timestamps[train_boundary]
        while (
            train_boundary < n_rows
            and timestamps[train_boundary] == train_time
        ):
            train_boundary += 1

    if validation_boundary < n_rows:
        validation_time = timestamps[validation_boundary]
        while (
            validation_boundary < n_rows
            and timestamps[validation_boundary] == validation_time
        ):
            validation_boundary += 1

    if not (
        0 < train_boundary < validation_boundary < n_rows
    ):
        raise ValueError(
            "Unable to create three non-empty temporal partitions."
        )

    train = ordered.iloc[:train_boundary].copy()
    validation = ordered.iloc[
        train_boundary:validation_boundary
    ].copy()
    test = ordered.iloc[validation_boundary:].copy()

    return TemporalSplit(
        train=train,
        validation=validation,
        test=test,
    )
