import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

EPSILON = 1e-6


def calculate_psi(reference, current):
    reference = np.asarray(reference, dtype=np.float64)
    current = np.asarray(current, dtype=np.float64)

    if len(reference) != len(current):
        raise ValueError("Reference and current distributions have different bucket counts.")

    reference = np.clip(reference, EPSILON, None)
    current = np.clip(current, EPSILON, None)

    reference = reference / reference.sum()
    current = current / current.sum()

    return float(
        np.sum((current - reference) * np.log(current / reference))
    )


def numeric_distribution(series, cut_points):
    values = pd.to_numeric(series, errors="coerce").to_numpy(dtype=np.float64)
    finite_mask = np.isfinite(values)

    valid = values[finite_mask]
    missing_count = int((~finite_mask).sum())

    bin_indices = np.searchsorted(cut_points, valid, side="right")
    bin_counts = np.bincount(
        bin_indices,
        minlength=len(cut_points) + 1,
    )

    counts = np.append(bin_counts, missing_count)
    return counts / len(series)


def categorical_distribution(series, top_categories):
    values = series.astype("string")

    missing_count = int(values.isna().sum())
    observed = values.dropna().astype(str)

    tracked = set(top_categories)
    tracked_counts = observed[observed.isin(tracked)].value_counts()

    counts = [
        int(tracked_counts.get(category, 0))
        for category in top_categories
    ]

    other_count = int((~observed.isin(tracked)).sum())

    counts.extend([other_count, missing_count])
    return np.asarray(counts, dtype=np.float64) / len(series)


def evaluate(input_path, profile_path, output_path):
    input_path = Path(input_path)
    profile_path = Path(profile_path)
    output_path = Path(output_path)

    if not input_path.exists():
        raise FileNotFoundError(f"Input data not found: {input_path}")

    if not profile_path.exists():
        raise FileNotFoundError(f"Reference profile not found: {profile_path}")

    with profile_path.open(encoding="utf-8") as file:
        profile = json.load(file)

    if input_path.suffix.lower() == ".parquet":
        df = pd.read_parquet(input_path)
    elif input_path.suffix.lower() == ".csv":
        df = pd.read_csv(input_path)
    else:
        raise ValueError("Input must be a CSV or Parquet file.")

    if df.empty:
        raise ValueError("Current monitoring window is empty.")

    expected_features = set(profile["numerical"]) | set(profile["categorical"])
    missing_features = sorted(expected_features - set(df.columns))

    if missing_features:
        raise ValueError(
            f"Input is missing {len(missing_features)} model features: "
            f"{missing_features[:10]}"
        )

    results = []

    for column, reference in profile["numerical"].items():
        current_distribution = numeric_distribution(
            df[column],
            reference["cut_points"],
        )

        reference_distribution = (
            reference["bin_proportions"]
            + [reference["missing_rate"]]
        )

        psi = calculate_psi(
            reference_distribution,
            current_distribution,
        )

        results.append({
            "feature": column,
            "feature_type": "numerical",
            "psi": psi,
            "reference_missing_rate": reference["missing_rate"],
            "current_missing_rate": float(current_distribution[-1]),
        })

    for column, reference in profile["categorical"].items():
        top_categories = list(reference["top_categories"])

        current_distribution = categorical_distribution(
            df[column],
            top_categories,
        )

        reference_distribution = (
            list(reference["category_proportions"].values())
            + [
                reference["other_proportion"],
                reference["missing_rate"],
            ]
        )

        psi = calculate_psi(
            reference_distribution,
            current_distribution,
        )

        results.append({
            "feature": column,
            "feature_type": "categorical",
            "psi": psi,
            "reference_missing_rate": reference["missing_rate"],
            "current_missing_rate": float(current_distribution[-1]),
        })

    result_df = pd.DataFrame(results).sort_values(
        "psi",
        ascending=False,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    result_df.to_csv(output_path, index=False)

    print("FEATURE DRIFT EVALUATION")
    print(f"Input rows: {len(df):,}")
    print(f"Features evaluated: {len(result_df)}")
    print(f"Output: {output_path}")

    print("\nTOP 10 FEATURES BY PSI")
    print(result_df.head(10).round(6).to_string(index=False))

    print("\nSUMMARY")
    print(f"Maximum PSI: {result_df['psi'].max():.8f}")
    print(f"Mean PSI: {result_df['psi'].mean():.8f}")
    print(f"Features with PSI > 0.10: {(result_df['psi'] > 0.10).sum()}")
    print(f"Features with PSI > 0.25: {(result_df['psi'] > 0.25).sum()}")
    return result_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate feature drift against a fixed reference profile."
    )
    parser.add_argument("--input", required=True, help="CSV or Parquet input window.")
    parser.add_argument(
        "--profile",
        default="artifacts/monitoring/reference_profile.json",
        help="Reference profile JSON.",
    )
    parser.add_argument(
        "--output",
        default="artifacts/monitoring/drift_report.csv",
        help="Output CSV report.",
    )

    args = parser.parse_args()
    evaluate(args.input, args.profile, args.output)
