import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

DATA_PATH = Path("data/processed/splits/train.parquet")
SCHEMA_PATH = Path("artifacts/xgboost/feature_schema.json")
OUTPUT_PATH = Path("artifacts/monitoring/reference_profile.json")

NUMERIC_BINS = 10
CATEGORICAL_TOP_K = 50


def numeric_profile(series):
    values = pd.to_numeric(series, errors="coerce").to_numpy(dtype=np.float64)
    finite_mask = np.isfinite(values)

    valid = values[finite_mask]
    missing_count = int((~finite_mask).sum())

    if len(valid) == 0:
        cut_points = []
        bin_counts = []
    else:
        quantiles = np.linspace(0, 1, NUMERIC_BINS + 1)[1:-1]
        cut_points = np.unique(np.quantile(valid, quantiles)).tolist()
        bin_indices = np.searchsorted(cut_points, valid, side="right")
        bin_counts = np.bincount(
            bin_indices,
            minlength=len(cut_points) + 1,
        ).astype(int).tolist()

    total = len(series)

    return {
        "cut_points": cut_points,
        "bin_counts": bin_counts,
        "missing_count": missing_count,
        "missing_rate": missing_count / total,
        "bin_proportions": [count / total for count in bin_counts],
    }


def categorical_profile(series):
    values = series.dropna().astype(str)
    frequencies = values.value_counts()

    top_categories = frequencies.head(CATEGORICAL_TOP_K)
    top_counts = {
        str(category): int(count)
        for category, count in top_categories.items()
    }

    missing_count = int(series.isna().sum())
    other_count = int(len(values) - top_categories.sum())
    total = len(series)

    return {
        "top_categories": top_counts,
        "other_count": other_count,
        "missing_count": missing_count,
        "missing_rate": missing_count / total,
        "category_proportions": {
            category: count / total
            for category, count in top_counts.items()
        },
        "other_proportion": other_count / total,
    }


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Training data not found: {DATA_PATH}")

    if not SCHEMA_PATH.exists():
        raise FileNotFoundError(f"Feature schema not found: {SCHEMA_PATH}")

    print("Loading training reference data...")
    df = pd.read_parquet(DATA_PATH)

    with SCHEMA_PATH.open(encoding="utf-8") as file:
        schema = json.load(file)

    feature_names = schema["feature_names"]
    numerical_columns = schema["numerical_columns"]
    categorical_columns = schema["categorical_columns"]

    if set(feature_names) - set(df.columns):
        missing = sorted(set(feature_names) - set(df.columns))
        raise ValueError(f"Missing model features: {missing}")

    if len(feature_names) != len(numerical_columns) + len(categorical_columns):
        raise ValueError("Feature schema numerical/categorical counts do not match.")

    if set(numerical_columns) & set(categorical_columns):
        raise ValueError("Numerical and categorical features overlap.")

    print(f"Rows: {len(df):,}")
    print(f"Numerical features: {len(numerical_columns)}")
    print(f"Categorical features: {len(categorical_columns)}")

    print("\nBuilding numerical reference profiles...")
    numerical_profiles = {
        column: numeric_profile(df[column])
        for column in numerical_columns
    }

    print("Building categorical reference profiles...")
    categorical_profiles = {
        column: categorical_profile(df[column])
        for column in categorical_columns
    }

    profile = {
        "profile_version": 1,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "source": str(DATA_PATH),
        "schema_source": str(SCHEMA_PATH),
        "row_count": len(df),
        "feature_count": len(feature_names),
        "numerical_feature_count": len(numerical_columns),
        "categorical_feature_count": len(categorical_columns),
        "categorical_top_k": CATEGORICAL_TOP_K,
        "numerical": numerical_profiles,
        "categorical": categorical_profiles,
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with OUTPUT_PATH.open("w", encoding="utf-8") as file:
        json.dump(profile, file, indent=2, allow_nan=False)

    print("\nREFERENCE PROFILE CREATED")
    print(f"Output: {OUTPUT_PATH}")
    print(f"Size: {OUTPUT_PATH.stat().st_size / 1024:.2f} KB")

    print("\nNUMERICAL PROFILE CHECK")
    for column in ["id_24", "D7", "TransactionAmt"]:
        if column in numerical_profiles:
            item = numerical_profiles[column]
            print(
                f"{column}: bins={len(item['bin_counts'])}, "
                f"missing_rate={item['missing_rate']:.2%}"
            )

    print("\nCATEGORICAL PROFILE CHECK")
    for column in ["card1", "DeviceInfo", "ProductCD"]:
        if column in categorical_profiles:
            item = categorical_profiles[column]
            print(
                f"{column}: tracked_categories={len(item['top_categories'])}, "
                f"other={item['other_count']:,}, "
                f"missing_rate={item['missing_rate']:.2%}"
            )


if __name__ == "__main__":
    main()
