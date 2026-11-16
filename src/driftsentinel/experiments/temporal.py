"""Rolling-origin temporal experiment construction."""

from dataclasses import dataclass
from itertools import pairwise

import pandas as pd


@dataclass(frozen=True)
class RollingFold:
    """Chronological partitions for one rolling evaluation fold."""

    fold_id: int
    adaptation_train: pd.DataFrame
    adaptation_calibration: pd.DataFrame
    evaluation: pd.DataFrame


@dataclass(frozen=True)
class RollingExperiment:
    """Initial historical data and sequential rolling evaluation folds."""

    initial_train: pd.DataFrame
    initial_calibration: pd.DataFrame
    folds: tuple[RollingFold, ...]


def split_into_temporal_segments(
    df: pd.DataFrame,
    n_segments: int,
    timestamp_column: str = "TransactionDT",
) -> tuple[pd.DataFrame, ...]:
    """Split data into approximately equal chronological segments.

    Rows with identical timestamps are always kept together.
    The input DataFrame is not modified.
    """

    if n_segments < 1:
        raise ValueError("n_segments must be at least 1.")

    if df.empty:
        raise ValueError("Cannot segment an empty DataFrame.")

    if timestamp_column not in df.columns:
        raise ValueError(f"Missing timestamp column: {timestamp_column}")

    if df[timestamp_column].isna().any():
        raise ValueError("Timestamp column contains missing values.")

    ordered = df.sort_values(
        timestamp_column,
        kind="mergesort",
    ).reset_index(drop=True)

    group_sizes = ordered.groupby(timestamp_column, sort=True).size().to_numpy()

    n_groups = len(group_sizes)

    if n_groups < n_segments:
        raise ValueError("Not enough distinct timestamps for non-empty segments.")

    prefix = [0]

    for size in group_sizes:
        prefix.append(prefix[-1] + int(size))

    total_rows = len(ordered)
    boundaries = []
    previous_group = 0

    for segment_index in range(1, n_segments):
        target_rows = total_rows * segment_index / n_segments
        remaining_segments = n_segments - segment_index

        first_candidate = previous_group + 1
        last_candidate = n_groups - remaining_segments

        best_group = min(
            range(first_candidate, last_candidate + 1),
            key=lambda group_index: abs(prefix[group_index] - target_rows),
        )

        boundaries.append(prefix[best_group])
        previous_group = best_group

    row_boundaries = [0, *boundaries, total_rows]

    segments = tuple(
        ordered.iloc[start:end].copy() for start, end in pairwise(row_boundaries)
    )

    if any(segment.empty for segment in segments):
        raise ValueError("Temporal segmentation produced an empty segment.")

    return segments


def create_rolling_folds(
    initial_train: pd.DataFrame,
    initial_calibration: pd.DataFrame,
    validation: pd.DataFrame,
    n_folds: int = 5,
    timestamp_column: str = "TransactionDT",
    id_column: str = "TransactionID",
) -> RollingExperiment:
    """Construct expanding-training, rolling-calibration evaluation folds.

    Assumption:
    Labels from a completed validation period are available before
    the next fold's training and calibration process.
    """

    if n_folds < 1:
        raise ValueError("n_folds must be at least 1.")

    partitions = {
        "initial_train": initial_train,
        "initial_calibration": initial_calibration,
        "validation": validation,
    }

    required = {timestamp_column, id_column, "isFraud"}

    for name, df in partitions.items():
        if df.empty:
            raise ValueError(f"{name} cannot be empty.")

        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"{name} is missing required columns: {missing}")

        if df[timestamp_column].isna().any():
            raise ValueError(f"{name} contains missing timestamps.")

        if df[id_column].isna().any():
            raise ValueError(f"{name} contains missing transaction IDs.")

        if df[id_column].duplicated().any():
            raise ValueError(f"{name} contains duplicate transaction IDs.")

        if not df["isFraud"].isin([0, 1]).all():
            raise ValueError(f"{name} contains invalid target labels.")

    ordered = {
        name: df.sort_values(
            timestamp_column,
            kind="mergesort",
        ).reset_index(drop=True)
        for name, df in partitions.items()
    }

    initial_train = ordered["initial_train"]
    initial_calibration = ordered["initial_calibration"]
    validation = ordered["validation"]

    if (
        initial_train[timestamp_column].max()
        >= initial_calibration[timestamp_column].min()
    ):
        raise ValueError("Initial training must precede initial calibration.")

    if (
        initial_calibration[timestamp_column].max()
        >= validation[timestamp_column].min()
    ):
        raise ValueError("Initial calibration must precede validation.")

    all_ids = pd.concat(
        [df[id_column] for df in ordered.values()],
        ignore_index=True,
    )

    if all_ids.duplicated().any():
        raise ValueError("Transaction IDs overlap across development partitions.")

    validation_segments = split_into_temporal_segments(
        validation,
        n_segments=n_folds,
        timestamp_column=timestamp_column,
    )

    folds = []

    for index, evaluation in enumerate(validation_segments):
        if index == 0:
            model_train = initial_train.copy()
            calibration = initial_calibration.copy()

        elif index == 1:
            model_train = initial_train.copy()
            calibration = validation_segments[0].copy()

        else:
            model_train = pd.concat(
                [
                    initial_train,
                    *validation_segments[: index - 1],
                ],
                ignore_index=True,
            )

            calibration = validation_segments[index - 1].copy()

        model_train = model_train.sort_values(
            timestamp_column,
            kind="mergesort",
        ).reset_index(drop=True)

        if not (
            model_train[timestamp_column].max()
            < calibration[timestamp_column].min()
            < evaluation[timestamp_column].min()
        ):
            raise ValueError(f"Fold {index + 1} violates chronological ordering.")

        folds.append(
            RollingFold(
                fold_id=index + 1,
                adaptation_train=model_train,
                adaptation_calibration=calibration,
                evaluation=evaluation.copy(),
            )
        )

    return RollingExperiment(
        initial_train=initial_train.copy(),
        initial_calibration=initial_calibration.copy(),
        folds=tuple(folds),
    )
