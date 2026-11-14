
from collections.abc import Iterable

import pandas as pd


def validate_required_columns(
    df: pd.DataFrame,
    required_columns: Iterable[str],
) -> None:
    """Raise an error if required columns are missing."""
    missing = set(required_columns) - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing required columns: {sorted(missing)}"
        )


def validate_transaction_data(
    df: pd.DataFrame,
    target_column: str = "isFraud",
    id_column: str = "TransactionID",
    timestamp_column: str = "TransactionDT",
    require_target: bool = True,
) -> None:
    """Validate the structure and essential fields of transaction data."""

    required = [id_column, timestamp_column]

    if require_target:
        required.append(target_column)

    validate_required_columns(df, required)

    if df[id_column].isna().any():
        raise ValueError(f"{id_column} contains missing values.")

    if df[id_column].duplicated().any():
        raise ValueError(f"{id_column} contains duplicate values.")

    if df[timestamp_column].isna().any():
        raise ValueError(f"{timestamp_column} contains missing values.")

    if require_target:
        if df[target_column].isna().any():
            raise ValueError(f"{target_column} contains missing values.")

        observed = set(df[target_column].unique())

        if not observed.issubset({0, 1}):
            raise ValueError(
                f"Unexpected target values: {observed}"
            )


def validate_identity_data(
    df: pd.DataFrame,
    id_column: str = "TransactionID",
) -> None:
    """Validate identity data and its transaction join key."""

    validate_required_columns(df, [id_column])

    if df[id_column].isna().any():
        raise ValueError(f"{id_column} contains missing values.")

    if df[id_column].duplicated().any():
        raise ValueError(f"{id_column} contains duplicate values.")


def get_missingness_report(df: pd.DataFrame) -> pd.DataFrame:
    """Return missing-value counts and percentages per column."""

    report = pd.DataFrame({
        "missing_count": df.isna().sum(),
        "missing_percentage": df.isna().mean().mul(100),
    })

    return report.sort_values(
        by="missing_percentage",
        ascending=False,
    )


def get_target_distribution(
    df: pd.DataFrame,
    target_column: str = "isFraud",
) -> pd.DataFrame:
    """Return target counts and class percentages."""

    validate_required_columns(df, [target_column])

    counts = df[target_column].value_counts(dropna=False)

    return pd.DataFrame({
        "count": counts,
        "percentage": counts.div(len(df)).mul(100),
    })
