
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TRAIN_FILE = ROOT / "data/processed/splits/train.parquet"
OUTPUT_DIR = ROOT / "experiments/results"


def profile_features():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading training partition...")
    df = pd.read_parquet(TRAIN_FILE)

    target = "isFraud"
    identifiers = ["TransactionID"]
    temporal_columns = ["TransactionDT"]

    excluded = set(identifiers + temporal_columns + [target])
    feature_columns = [c for c in df.columns if c not in excluded]

    profiles = []

    for column in feature_columns:
        series = df[column]

        profiles.append({
            "feature": column,
            "dtype": str(series.dtype),
            "missing_count": int(series.isna().sum()),
            "missing_percentage": round(
                float(series.isna().mean() * 100), 3
            ),
            "unique_values": int(series.nunique(dropna=True)),
            "role": (
                "categorical"
                if (
                    pd.api.types.is_object_dtype(series)
                    or isinstance(series.dtype, pd.CategoricalDtype)
                    or pd.api.types.is_string_dtype(series)
                )
                else "numerical"
            ),
        })

    report = pd.DataFrame(profiles)

    report = report.sort_values(
        ["role", "unique_values"],
        ascending=[True, False],
    )

    output_file = OUTPUT_DIR / "feature_profile.csv"
    report.to_csv(output_file, index=False)

    print("\n========== FEATURE PROFILE ==========")
    print("Total training rows:", len(df))
    print("Total model features:", len(feature_columns))

    print("\nFeature types:")
    print(report["role"].value_counts().to_string())

    print("\nOriginal pandas dtypes:")
    print(df[feature_columns].dtypes.astype(str).value_counts().to_string())

    print("\nTop 20 highest-cardinality features:")
    print(
        report.sort_values(
            "unique_values", ascending=False
        ).head(20)[
            ["feature", "dtype", "unique_values", "missing_percentage"]
        ].to_string(index=False)
    )

    print("\nTop 20 most-missing features:")
    print(
        report.sort_values(
            "missing_percentage", ascending=False
        ).head(20)[
            ["feature", "dtype", "missing_percentage", "unique_values"]
        ].to_string(index=False)
    )

    summary = {
        "training_rows": len(df),
        "model_features": len(feature_columns),
        "role_counts": {
            str(k): int(v)
            for k, v in report["role"].value_counts().items()
        },
        "excluded_columns": sorted(excluded),
    }

    with (OUTPUT_DIR / "feature_profile_summary.json").open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(summary, file, indent=4)

    print("\nSaved:", output_file)


if __name__ == "__main__":
    profile_features()
