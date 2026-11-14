
import pandas as pd
import pytest

from driftsentinel.data.loader import (
    load_identity,
    load_ieee_cis,
    load_transactions,
)


def test_load_transactions(tmp_path):
    path = tmp_path / "transactions.csv"

    pd.DataFrame({
        "TransactionID": [1, 2],
        "isFraud": [0, 1],
    }).to_csv(path, index=False)

    df = load_transactions(path)

    assert len(df) == 2
    assert "isFraud" in df.columns


def test_load_identity_normalizes_columns(tmp_path):
    path = tmp_path / "identity.csv"

    pd.DataFrame({
        "TransactionID": [1],
        "id-01": [2.5],
    }).to_csv(path, index=False)

    df = load_identity(path)

    assert "id_01" in df.columns
    assert "id-01" not in df.columns


def test_missing_file_raises_error(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_transactions(tmp_path / "missing.csv")


def test_load_ieee_cis_merges_correctly(tmp_path):
    transaction_path = tmp_path / "transactions.csv"
    identity_path = tmp_path / "identity.csv"

    pd.DataFrame({
        "TransactionID": [1, 2, 3],
        "isFraud": [0, 1, 0],
        "TransactionDT": [100, 200, 300],
    }).to_csv(transaction_path, index=False)

    pd.DataFrame({
        "TransactionID": [1, 3],
        "id-01": [2.5, 3.5],
    }).to_csv(identity_path, index=False)

    df = load_ieee_cis(transaction_path, identity_path)

    assert len(df) == 3
    assert "id_01" in df.columns
    assert pd.isna(df.loc[1, "id_01"])


def test_duplicate_identity_ids_raise_error(tmp_path):
    transaction_path = tmp_path / "transactions.csv"
    identity_path = tmp_path / "identity.csv"

    pd.DataFrame({
        "TransactionID": [1, 2],
    }).to_csv(transaction_path, index=False)

    pd.DataFrame({
        "TransactionID": [1, 1],
        "id_01": [2.5, 3.5],
    }).to_csv(identity_path, index=False)

    with pytest.raises(ValueError):
        load_ieee_cis(transaction_path, identity_path)
