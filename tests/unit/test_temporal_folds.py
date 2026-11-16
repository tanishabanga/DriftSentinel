from itertools import pairwise

import pandas as pd
import pytest

from driftsentinel.experiments.temporal import (
    create_rolling_folds,
    split_into_temporal_segments,
)


def make_data():
    return pd.DataFrame(
        {
            "TransactionID": range(1, 13),
            "TransactionDT": [
                100,
                100,
                101,
                101,
                102,
                102,
                103,
                103,
                104,
                104,
                105,
                105,
            ],
            "isFraud": [
                0,
                1,
                0,
                0,
                1,
                0,
                0,
                1,
                0,
                0,
                1,
                0,
            ],
        }
    )


def make_rolling_partitions():
    def create_partition(
        start_id,
        start_timestamp,
        n_timestamps,
    ):
        rows = []

        transaction_id = start_id

        for timestamp in range(
            start_timestamp,
            start_timestamp + n_timestamps,
        ):
            for label in [0, 1]:
                rows.append(
                    {
                        "TransactionID": transaction_id,
                        "TransactionDT": timestamp,
                        "isFraud": label,
                    }
                )
                transaction_id += 1

        return pd.DataFrame(rows)

    initial_train = create_partition(1, 100, 4)
    initial_calibration = create_partition(9, 104, 2)
    validation = create_partition(13, 106, 10)

    return initial_train, initial_calibration, validation


def test_creates_requested_number_of_rolling_folds():
    initial_train, initial_calibration, validation = make_rolling_partitions()

    experiment = create_rolling_folds(
        initial_train,
        initial_calibration,
        validation,
        n_folds=5,
    )

    assert len(experiment.folds) == 5

    assert [fold.fold_id for fold in experiment.folds] == [1, 2, 3, 4, 5]


def test_training_data_expands_across_folds():
    initial_train, initial_calibration, validation = make_rolling_partitions()

    experiment = create_rolling_folds(
        initial_train,
        initial_calibration,
        validation,
        n_folds=5,
    )

    training_sizes = [len(fold.adaptation_train) for fold in experiment.folds]

    assert training_sizes == [8, 8, 12, 16, 20]

    assert all(left <= right for left, right in pairwise(training_sizes))


def test_rolling_fold_partitions_do_not_overlap():
    initial_train, initial_calibration, validation = make_rolling_partitions()

    experiment = create_rolling_folds(
        initial_train,
        initial_calibration,
        validation,
        n_folds=5,
    )

    for fold in experiment.folds:
        train_ids = set(fold.adaptation_train["TransactionID"])
        calibration_ids = set(fold.adaptation_calibration["TransactionID"])
        evaluation_ids = set(fold.evaluation["TransactionID"])

        assert train_ids.isdisjoint(calibration_ids)
        assert train_ids.isdisjoint(evaluation_ids)
        assert calibration_ids.isdisjoint(evaluation_ids)


def test_rolling_folds_preserve_chronological_order():
    initial_train, initial_calibration, validation = make_rolling_partitions()

    experiment = create_rolling_folds(
        initial_train,
        initial_calibration,
        validation,
        n_folds=5,
    )

    for fold in experiment.folds:
        train_max = fold.adaptation_train["TransactionDT"].max()

        calibration_min = fold.adaptation_calibration["TransactionDT"].min()

        calibration_max = fold.adaptation_calibration["TransactionDT"].max()

        evaluation_min = fold.evaluation["TransactionDT"].min()

        assert train_max < calibration_min
        assert calibration_max < evaluation_min


def test_creates_requested_number_of_segments():
    segments = split_into_temporal_segments(
        make_data(),
        n_segments=3,
    )

    assert len(segments) == 3


def test_segments_are_chronological():
    segments = split_into_temporal_segments(
        make_data(),
        n_segments=3,
    )

    for left, right in pairwise(segments):
        assert left["TransactionDT"].max() < right["TransactionDT"].min()


def test_equal_timestamps_are_never_split():
    segments = split_into_temporal_segments(
        make_data(),
        n_segments=3,
    )

    seen_timestamps = set()

    for segment in segments:
        segment_timestamps = set(segment["TransactionDT"])

        # No timestamp should appear in multiple segments.
        assert seen_timestamps.isdisjoint(segment_timestamps)

        seen_timestamps.update(segment_timestamps)


def test_all_rows_are_preserved():
    df = make_data()

    segments = split_into_temporal_segments(
        df,
        n_segments=3,
    )

    assert sum(len(segment) for segment in segments) == len(df)

    combined_ids = pd.concat([segment["TransactionID"] for segment in segments])

    assert set(combined_ids) == set(df["TransactionID"])
    assert combined_ids.is_unique


def test_input_dataframe_is_not_modified():
    df = make_data()
    original = df.copy(deep=True)

    split_into_temporal_segments(df, n_segments=3)

    pd.testing.assert_frame_equal(df, original)


def test_empty_dataframe_is_rejected():
    with pytest.raises(ValueError, match="empty"):
        split_into_temporal_segments(
            pd.DataFrame(),
            n_segments=3,
        )


def test_missing_timestamp_is_rejected():
    df = make_data()
    df.loc[0, "TransactionDT"] = None

    with pytest.raises(ValueError, match="missing"):
        split_into_temporal_segments(df, n_segments=3)


def test_too_few_distinct_timestamps_are_rejected():
    df = pd.DataFrame(
        {
            "TransactionID": [1, 2, 3],
            "TransactionDT": [100, 100, 101],
            "isFraud": [0, 1, 0],
        }
    )

    with pytest.raises(ValueError, match="distinct timestamps"):
        split_into_temporal_segments(df, n_segments=3)


def test_invalid_segment_count_is_rejected():
    with pytest.raises(ValueError):
        split_into_temporal_segments(make_data(), n_segments=0)
