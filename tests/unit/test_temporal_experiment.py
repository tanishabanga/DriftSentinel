
import pandas as pd

from driftsentinel.data.temporal import (
    split_adaptation_window,
    split_temporal_windows,
)


def make_validation_data():
    """Create a small chronological validation dataset."""

    return pd.DataFrame(
        {
            "TransactionID": range(1, 13),
            "TransactionDT": range(100, 112),
            "isFraud": [0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0],
        }
    )


def test_temporal_windows_are_chronological():
    windows = split_temporal_windows(make_validation_data())

    assert (
        windows.initial_calibration["TransactionDT"].max()
        < windows.adaptation["TransactionDT"].min()
    )

    assert (
        windows.adaptation["TransactionDT"].max()
        < windows.evaluation["TransactionDT"].min()
    )


def test_temporal_windows_do_not_overlap():
    windows = split_temporal_windows(make_validation_data())

    initial_ids = set(windows.initial_calibration["TransactionID"])
    adaptation_ids = set(windows.adaptation["TransactionID"])
    evaluation_ids = set(windows.evaluation["TransactionID"])

    assert initial_ids.isdisjoint(adaptation_ids)
    assert initial_ids.isdisjoint(evaluation_ids)
    assert adaptation_ids.isdisjoint(evaluation_ids)


def test_temporal_windows_preserve_all_rows():
    df = make_validation_data()
    windows = split_temporal_windows(df)

    total_rows = (
        len(windows.initial_calibration)
        + len(windows.adaptation)
        + len(windows.evaluation)
    )

    assert total_rows == len(df)


def test_adaptation_split_is_chronological():
    windows = split_temporal_windows(make_validation_data())
    split = split_adaptation_window(windows.adaptation)

    assert (
        split.adaptation_train["TransactionDT"].max()
        < split.adaptation_calibration["TransactionDT"].min()
    )


def test_adaptation_split_has_no_overlap():
    windows = split_temporal_windows(make_validation_data())
    split = split_adaptation_window(windows.adaptation)

    train_ids = set(split.adaptation_train["TransactionID"])
    calibration_ids = set(
        split.adaptation_calibration["TransactionID"]
    )

    assert train_ids.isdisjoint(calibration_ids)


def test_adaptation_split_preserves_all_rows():
    windows = split_temporal_windows(make_validation_data())
    adaptation = windows.adaptation

    split = split_adaptation_window(adaptation)

    assert (
        len(split.adaptation_train)
        + len(split.adaptation_calibration)
        == len(adaptation)
    )


def test_missing_timestamp_is_rejected():
    df = make_validation_data()
    df.loc[0, "TransactionDT"] = None

    try:
        split_temporal_windows(df)
        assert False, "Expected ValueError"
    except ValueError as error:
        assert "missing values" in str(error)
