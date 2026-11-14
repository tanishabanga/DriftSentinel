
from pathlib import Path

import pandas as pd


def load_transactions(path: str | Path) -> pd.DataFrame:
    """Load transaction data from a CSV file."""
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Transaction file not found: {path}")

    return pd.read_csv(path, low_memory=False)


def load_identity(path: str | Path, normalize_columns: bool = True) -> pd.DataFrame:
    """Load identity data and optionally normalize column names."""
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Identity file not found: {path}")

    df = pd.read_csv(path, low_memory=False)

    if normalize_columns:
        df.columns = df.columns.str.replace("-", "_", regex=False)

    return df


def load_ieee_cis(
    transaction_path: str | Path,
    identity_path: str | Path,
) -> pd.DataFrame:
    """Load and merge IEEE-CIS transaction and identity datasets."""
    transactions = load_transactions(transaction_path)
    identity = load_identity(identity_path)

    if "TransactionID" not in transactions.columns:
        raise ValueError("Transaction data is missing TransactionID.")

    if "TransactionID" not in identity.columns:
        raise ValueError("Identity data is missing TransactionID.")

    if identity["TransactionID"].duplicated().any():
        raise ValueError("Identity data contains duplicate TransactionID values.")

    merged = transactions.merge(
        identity,
        on="TransactionID",
        how="left",
        validate="one_to_one",
    )

    return merged
