from pathlib import Path
import json

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy.spatial.distance import jensenshannon
from scipy.stats import ks_2samp


ROOT = Path(__file__).resolve().parents[1]

PARTITIONS = {
    "train": ROOT / "data/processed/splits/train.parquet",
    "validation_evaluation": (
        ROOT / "data/processed/temporal/evaluation.parquet"
    ),
    "final_holdout": ROOT / "data/processed/splits/test.parquet",
}

OUTPUT_DIR = ROOT / "experiments/results/temporal_feature_drift"
DETAIL_PATH = OUTPUT_DIR / "temporal_feature_drift.csv"
SUMMARY_PATH = OUTPUT_DIR / "temporal_feature_drift_summary.json"

EXCLUDED = {"TransactionID", "TransactionDT", "isFraud"}
N_QUANTILE_BINS = 10
EPSILON = 1e-6


def load_feature_columns(path):
    columns = pq.read_schema(path).names
    return [c for c in columns if c not in EXCLUDED]


def load_features(name, path, columns):
    print(f"Loading {name}: {path.name}")
    df = pd.read_parquet(path, columns=columns)
    if df.empty:
        raise ValueError(f"{name} partition is empty.")
    return df


def missing_percentage(series):
    return float(series.isna().mean() * 100)


def numerical_psi(reference, comparison):
    ref = pd.to_numeric(reference, errors="coerce").to_numpy(dtype=float)
    cur = pd.to_numeric(comparison, errors="coerce").to_numpy(dtype=float)

    ref_valid = ref[np.isfinite(ref)]
    cur_valid = cur[np.isfinite(cur)]

    if len(ref_valid) == 0 or len(cur_valid) == 0:
        return np.nan

    quantiles = np.linspace(0, 1, N_QUANTILE_BINS + 1)
    boundaries = np.unique(np.quantile(ref_valid, quantiles))

    if len(boundaries) <= 1:
        inner_edges = np.array([], dtype=float)
    else:
        inner_edges = boundaries[1:-1]

    # Infinite outer edges ensure that comparison-period values
    # outside the reference range are still counted.
    ref_bins = np.searchsorted(inner_edges, ref_valid, side="right")
    cur_bins = np.searchsorted(inner_edges, cur_valid, side="right")

    bin_count = len(inner_edges) + 1

    ref_counts = np.bincount(ref_bins, minlength=bin_count).astype(float)
    cur_counts = np.bincount(cur_bins, minlength=bin_count).astype(float)

    # Treat NaN and +/- infinity as invalid observations in the final bin.
    ref_invalid = int((~np.isfinite(ref)).sum())
    cur_invalid = int((~np.isfinite(cur)).sum())

    ref_counts = np.append(ref_counts, ref_invalid)
    cur_counts = np.append(cur_counts, cur_invalid)

    ref_dist = ref_counts / ref_counts.sum()
    cur_dist = cur_counts / cur_counts.sum()

    ref_dist = np.clip(ref_dist, EPSILON, None)
    cur_dist = np.clip(cur_dist, EPSILON, None)

    ref_dist /= ref_dist.sum()
    cur_dist /= cur_dist.sum()

    return float(
        np.sum((cur_dist - ref_dist) * np.log(cur_dist / ref_dist))
    )


def categorical_js(reference, comparison):
    ref = reference.astype("string").fillna("__MISSING__")
    cur = comparison.astype("string").fillna("__MISSING__")

    ref_counts = ref.value_counts(normalize=False)
    cur_counts = cur.value_counts(normalize=False)

    categories = ref_counts.index.union(cur_counts.index)

    p = ref_counts.reindex(categories, fill_value=0).to_numpy(dtype=float)
    q = cur_counts.reindex(categories, fill_value=0).to_numpy(dtype=float)

    p /= p.sum()
    q /= q.sum()

    # Squared Jensen-Shannon distance is Jensen-Shannon divergence.
    return float(jensenshannon(p, q, base=2) ** 2)


def compare_pair(reference_name, comparison_name, reference, comparison,
                 feature_columns, numerical_features):
    rows = []

    for index, feature in enumerate(feature_columns, start=1):
        ref = reference[feature]
        cur = comparison[feature]

        ref_missing = missing_percentage(ref)
        cur_missing = missing_percentage(cur)

        row = {
            "reference_period": reference_name,
            "comparison_period": comparison_name,
            "feature": feature,
            "feature_type": (
                "numerical" if feature in numerical_features else "categorical"
            ),
            "reference_missing_pct": ref_missing,
            "comparison_missing_pct": cur_missing,
            "missingness_change_pp": cur_missing - ref_missing,
            "ks_statistic": np.nan,
            "ks_pvalue": np.nan,
            "psi": np.nan,
            "js_divergence": np.nan,
        }

        if feature in numerical_features:
            ref_values = pd.to_numeric(ref, errors="coerce").to_numpy(dtype=float)
            cur_values = pd.to_numeric(cur, errors="coerce").to_numpy(dtype=float)

            ref_values = ref_values[np.isfinite(ref_values)]
            cur_values = cur_values[np.isfinite(cur_values)]

            if len(ref_values) and len(cur_values):
                ks = ks_2samp(ref_values, cur_values)
                row["ks_statistic"] = float(ks.statistic)
                row["ks_pvalue"] = float(ks.pvalue)

            row["psi"] = numerical_psi(ref, cur)
        else:
            row["js_divergence"] = categorical_js(ref, cur)

        rows.append(row)

        if index % 100 == 0:
            print(
                f"  Processed {index}/{len(feature_columns)} features "
                f"for {reference_name} -> {comparison_name}"
            )

    return rows


def main():
    if DETAIL_PATH.exists() or SUMMARY_PATH.exists():
        raise FileExistsError(
            "Drift-analysis output already exists. "
            "Do not overwrite existing results."
        )

    for path in PARTITIONS.values():
        if not path.exists():
            raise FileNotFoundError(path)

    feature_columns = load_feature_columns(PARTITIONS["train"])

    if len(feature_columns) != 431:
        raise ValueError(
            f"Expected 431 features, found {len(feature_columns)}."
        )

    for name, path in PARTITIONS.items():
        current = load_feature_columns(path)
        if current != feature_columns:
            raise ValueError(
                f"Feature names or order differ in {name} partition."
            )

    train = load_features("train", PARTITIONS["train"], feature_columns)

    numerical_features = set(
        train.select_dtypes(include="number").columns
    )
    categorical_features = set(feature_columns) - numerical_features

    if len(numerical_features) != 400 or len(categorical_features) != 31:
        raise ValueError(
            "Unexpected feature type counts: "
            f"{len(numerical_features)} numerical, "
            f"{len(categorical_features)} categorical."
        )

    validation = load_features(
        "validation_evaluation",
        PARTITIONS["validation_evaluation"],
        feature_columns,
    )
    holdout = load_features(
        "final_holdout",
        PARTITIONS["final_holdout"],
        feature_columns,
    )

    comparisons = [
        ("train", "validation_evaluation", train, validation),
        ("train", "final_holdout", train, holdout),
        ("validation_evaluation", "final_holdout", validation, holdout),
    ]

    all_rows = []

    for reference_name, comparison_name, reference, comparison in comparisons:
        print(
            f"\nComparing {reference_name} -> {comparison_name}"
        )
        all_rows.extend(
            compare_pair(
                reference_name,
                comparison_name,
                reference,
                comparison,
                feature_columns,
                numerical_features,
            )
        )

    results = pd.DataFrame(all_rows)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    results.to_csv(DETAIL_PATH, index=False)

    summary = {
        "experiment": "temporal_feature_distribution_drift",
        "target_loaded": False,
        "model_features": len(feature_columns),
        "numerical_features": len(numerical_features),
        "categorical_features": len(categorical_features),
        "partitions": {
            name: {
                "rows": int(
                    len(train) if name == "train"
                    else len(validation) if name == "validation_evaluation"
                    else len(holdout)
                ),
                "file": str(path.relative_to(ROOT)),
            }
            for name, path in PARTITIONS.items()
        },
        "comparisons": [
            "train_vs_validation_evaluation",
            "train_vs_final_holdout",
            "validation_evaluation_vs_final_holdout",
        ],
        "methods": {
            "numerical": (
                "Two-sample Kolmogorov-Smirnov statistic and p-value; "
                "PSI using reference-period quantile bins with missingness "
                "as an explicit bin."
            ),
            "categorical": (
                "Squared Jensen-Shannon distance (base 2), including "
                "missing values as a category."
            ),
            "missingness": "Comparison minus reference, percentage points.",
        },
        "psi_quantile_bins": N_QUANTILE_BINS,
        "psi_smoothing_epsilon": EPSILON,
        "interpretation_note": (
            "These are descriptive feature-distribution comparisons. "
            "They measure changes in P(X), not concept drift in P(Y|X). "
            "KS p-values are sample-size sensitive and should not be "
            "interpreted alone."
        ),
        "detail_file": str(DETAIL_PATH.relative_to(ROOT)),
        "feature_comparisons": len(results),
    }

    with SUMMARY_PATH.open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)

    print("\nTEMPORAL FEATURE DRIFT ANALYSIS COMPLETED")
    print("Feature comparisons:", len(results))
    print("Detailed results:", DETAIL_PATH)
    print("Summary:", SUMMARY_PATH)


if __name__ == "__main__":
    main()
