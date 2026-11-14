
from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class TemporalWindows:
    initial_calibration: pd.DataFrame
    adaptation: pd.DataFrame
    evaluation: pd.DataFrame


@dataclass(frozen=True)
class AdaptationSplit:
    adaptation_train: pd.DataFrame
    adaptation_calibration: pd.DataFrame


def split_temporal_windows(
    df: pd.DataFrame,
    timestamp_column: str = "TransactionDT",
) -> TemporalWindows:
    """Split chronological validation data into three windows."""

    if df.empty:
        raise ValueError("Cannot split an empty DataFrame.")

    if timestamp_column not in df.columns:
        raise ValueError(
            f"Missing timestamp column: {timestamp_column}"
        )

    if df[timestamp_column].isna().any():
        raise ValueError("Timestamp column contains missing values.")

    ordered = df.sort_values(
        timestamp_column,
        kind="mergesort",
    ).reset_index(drop=True)

    boundaries = ordered[timestamp_column].quantile(
        [1 / 3, 2 / 3]
    ).tolist()

    initial = ordered[
        ordered[timestamp_column] <= boundaries[0]
    ].copy()

    adaptation = ordered[
        (ordered[timestamp_column] > boundaries[0])
        & (ordered[timestamp_column] <= boundaries[1])
    ].copy()

    evaluation = ordered[
        ordered[timestamp_column] > boundaries[1]
    ].copy()

    if any(
        partition.empty
        for partition in (initial, adaptation, evaluation)
    ):
        raise ValueError(
            "Unable to create three non-empty temporal windows."
        )

    return TemporalWindows(
        initial_calibration=initial,
        adaptation=adaptation,
        evaluation=evaluation,
    )


def split_adaptation_window(
    df: pd.DataFrame,
    timestamp_column: str = "TransactionDT",
) -> AdaptationSplit:
    """Split adaptation data chronologically into two windows."""

    if df.empty:
        raise ValueError("Cannot split an empty adaptation window.")

    if timestamp_column not in df.columns:
        raise ValueError(
            f"Missing timestamp column: {timestamp_column}"
        )

    if df[timestamp_column].isna().any():
        raise ValueError("Timestamp column contains missing values.")

    ordered = df.sort_values(
        timestamp_column,
        kind="mergesort",
    ).reset_index(drop=True)

    if ordered[timestamp_column].nunique() < 2:
        raise ValueError(
            "At least two distinct timestamps are required."
        )

    boundary = ordered[timestamp_column].quantile(0.5)

    adaptation_train = ordered[
        ordered[timestamp_column] <= boundary
    ].copy()

    adaptation_calibration = ordered[
        ordered[timestamp_column] > boundary
    ].copy()

    if adaptation_train.empty or adaptation_calibration.empty:
        raise ValueError(
            "Unable to create two non-empty adaptation partitions."
        )

    return AdaptationSplit(
        adaptation_train=adaptation_train,
        adaptation_calibration=adaptation_calibration,
    )
