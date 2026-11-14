
import pandas as pd
import pytest

from driftsentinel.data.splitting import chronological_split


def make_sample_data():
    return pd.DataFrame({
        "TransactionID": range(1, 101),
        "TransactionDT": range(100),
        "isFraud": [0, 1] * 50,
    })


def test_split_has_three_non_empty_partitions():
    df = make_sample_data()

    result = chronological_split(df)

    assert len(result.train) > 0
    assert len(result.validation) > 0
    assert len(result.test) > 0


def test_split_preserves_chronological_order():
    df = make_sample_data()

    result = chronological_split(df)

    assert result.train["TransactionDT"].max() < (
        result.validation["TransactionDT"].min()
    )

    assert result.validation["TransactionDT"].max() < (
        result.test["TransactionDT"].min()
    )


def test_split_does_not_modify_original_dataframe():
    df = make_sample_data()
    original = df.copy(deep=True)

    chronological_split(df)

    pd.testing.assert_frame_equal(df, original)


def test_equal_timestamps_are_not_split():
    df = pd.DataFrame({
        "TransactionID": range(1, 101),
        "TransactionDT": (
            list(range(70)) + [70] * 10 + list(range(71, 91))
        ),
        "isFraud": [0, 1] * 50,
    })

    result = chronological_split(df)

    train_times = set(result.train["TransactionDT"])
    validation_times = set(result.validation["TransactionDT"])
    test_times = set(result.test["TransactionDT"])

    assert train_times.isdisjoint(validation_times)
    assert validation_times.isdisjoint(test_times)


def test_invalid_ratios_raise_error():
    df = make_sample_data()

    with pytest.raises(ValueError):
        chronological_split(df, train_ratio=0.8, validation_ratio=0.3)
