
import pandas as pd
import pytest

from driftsentinel.data.validation import (
    get_missingness_report,
    get_target_distribution,
    validate_identity_data,
    validate_required_columns,
    validate_transaction_data,
)


def make_transactions():
    return pd.DataFrame({
        "TransactionID": [1, 2, 3, 4],
        "TransactionDT": [100, 200, 300, 400],
        "isFraud": [0, 0, 1, 0],
        "TransactionAmt": [100.0, None, 250.0, 80.0],
    })


def test_required_columns_pass():
    df = make_transactions()
    validate_required_columns(df, ["TransactionID", "isFraud"])


def test_missing_required_column_fails():
    df = make_transactions()

    with pytest.raises(ValueError):
        validate_required_columns(df, ["unknown_column"])


def test_valid_transaction_data_passes():
    validate_transaction_data(make_transactions())


def test_duplicate_transaction_ids_fail():
    df = make_transactions()
    df.loc[1, "TransactionID"] = 1

    with pytest.raises(ValueError):
        validate_transaction_data(df)


def test_invalid_target_values_fail():
    df = make_transactions()
    df.loc[0, "isFraud"] = 2

    with pytest.raises(ValueError):
        validate_transaction_data(df)


def test_identity_duplicate_ids_fail():
    df = pd.DataFrame({
        "TransactionID": [1, 1],
        "id_01": [2.0, 3.0],
    })

    with pytest.raises(ValueError):
        validate_identity_data(df)


def test_missingness_report():
    report = get_missingness_report(make_transactions())

    assert report.loc["TransactionAmt", "missing_count"] == 1
    assert report.loc["TransactionAmt", "missing_percentage"] == 25.0


def test_target_distribution():
    report = get_target_distribution(make_transactions())

    assert report.loc[0, "count"] == 3
    assert report.loc[1, "count"] == 1
