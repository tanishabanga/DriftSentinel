from pathlib import Path
import json

import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon


ROOT = Path(__file__).resolve().parents[1]

RESULTS_DIR = ROOT / "experiments/results/temporal_feature_drift"
INPUT_FILE = RESULTS_DIR / "temporal_feature_drift.csv"

OUTPUT_FILE = RESULTS_DIR / "temporal_feature_drift_enhanced.csv"
SUMMARY_FILE = RESULTS_DIR / "temporal_feature_drift_enhanced_summary.json"

PARTITIONS = {
    "train": ROOT / "data/processed/splits/train.parquet",
    "validation_evaluation": ROOT / "data/processed/temporal/evaluation.parquet",
    "final_holdout": ROOT / "data/processed/splits/test.parquet",
}

EPSILON = 1e-6
QUANTILE_BINS = 10


def benjamini_hochberg(pvalues):
    """Apply Benjamini-Hochberg FDR correction."""
    pvalues = np.asarray(pvalues, dtype=float)
    adjusted = np.full(len(pvalues), np.nan)

    valid = np.isfinite(pvalues)
    p = pvalues[valid]

    if len(p) == 0:
        return adjusted

    order = np.argsort(p)
    ranked = p[order]
    n = len(ranked)

    corrected = ranked * n / np.arange(1, n + 1)
    corrected = np.minimum.accumulate(corrected[::-1])[::-1]
    corrected = np.clip(corrected, 0, 1)

    restored = np.empty(n)
    restored[order] = corrected
    adjusted[valid] = restored

    return adjusted


def observed_only_psi(reference, comparison):
    """PSI using observed numerical values, excluding missing/non-finite data."""
    ref = pd.to_numeric(reference, errors="coerce").to_numpy(dtype=float)
    cur = pd.to_numeric(comparison, errors="coerce").to_numpy(dtype=float)

    ref = ref[np.isfinite(ref)]
    cur = cur[np.isfinite(cur)]

    if len(ref) == 0 or len(cur) == 0:
        return np.nan

    # Bin boundaries are learned only from the reference period.
    quantiles = np.linspace(0, 1, QUANTILE_BINS + 1)
    boundaries = np.unique(np.quantile(ref, quantiles))
    inner_edges = boundaries[1:-1] if len(boundaries) > 1 else np.array([])

    ref_bins = np.searchsorted(inner_edges, ref, side="right")
    cur_bins = np.searchsorted(inner_edges, cur, side="right")

    bin_count = len(inner_edges) + 1

    ref_counts = np.bincount(ref_bins, minlength=bin_count).astype(float)
    cur_counts = np.bincount(cur_bins, minlength=bin_count).astype(float)

    ref_dist = ref_counts / ref_counts.sum()
    cur_dist = cur_counts / cur_counts.sum()

    ref_dist = np.clip(ref_dist, EPSILON, None)
    cur_dist = np.clip(cur_dist, EPSILON, None)

    ref_dist /= ref_dist.sum()
    cur_dist /= cur_dist.sum()

    return float(
        np.sum((cur_dist - ref_dist) * np.log(cur_dist / ref_dist))
    )


def observed_only_js(reference, comparison):
    """Jensen-Shannon divergence over non-missing categorical observations."""
    ref = reference.dropna().astype("string")
    cur = comparison.dropna().astype("string")

    if len(ref) == 0 or len(cur) == 0:
        return np.nan

    ref_counts = ref.value_counts()
    cur_counts = cur.value_counts()

    categories = ref_counts.index.union(cur_counts.index)

    p = ref_counts.reindex(categories, fill_value=0).to_numpy(dtype=float)
    q = cur_counts.reindex(categories, fill_value=0).to_numpy(dtype=float)

    p /= p.sum()
    q /= q.sum()

    return float(jensenshannon(p, q, base=2) ** 2)


def load_partition_features(path, feature_columns):
    return pd.read_parquet(path, columns=feature_columns)


def main():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Original drift results not found: {INPUT_FILE}"
        )

    if OUTPUT_FILE.exists() or SUMMARY_FILE.exists():
        raise FileExistsError(
            "Enhanced output already exists. "
            "Existing results will not be overwritten."
        )

    results = pd.read_csv(INPUT_FILE)

    expected_rows = 431 * 3
    if len(results) != expected_rows:
        raise ValueError(
            f"Expected {expected_rows} comparisons, found {len(results)}."
        )

    if results.duplicated(
        ["reference_period", "comparison_period", "feature"]
    ).any():
        raise ValueError("Duplicate feature comparisons found.")

    feature_columns = results["feature"].drop_duplicates().tolist()

    if len(feature_columns) != 431:
        raise ValueError(
            f"Expected 431 unique features, found {len(feature_columns)}."
        )

    print("Loading feature-only partitions...")
    frames = {
        name: load_partition_features(path, feature_columns)
        for name, path in PARTITIONS.items()
    }

    numerical_features = set(
        results.loc[
            results["feature_type"] == "numerical", "feature"
        ]
    )
    categorical_features = set(feature_columns) - numerical_features

    if len(numerical_features) != 400 or len(categorical_features) != 31:
        raise ValueError("Unexpected numerical/categorical feature counts.")

    results["observed_only_psi"] = np.nan
    results["observed_only_js"] = np.nan
    results["ks_fdr_qvalue"] = np.nan

    comparisons = [
        ("train", "validation_evaluation"),
        ("train", "final_holdout"),
        ("validation_evaluation", "final_holdout"),
    ]

    for reference_name, comparison_name in comparisons:
        print(f"\nProcessing {reference_name} -> {comparison_name}")

        reference = frames[reference_name]
        comparison = frames[comparison_name]

        mask = (
            (results["reference_period"] == reference_name)
            & (results["comparison_period"] == comparison_name)
        )

        numerical_mask = mask & (results["feature_type"] == "numerical")
        categorical_mask = mask & (results["feature_type"] == "categorical")

        numerical_indices = results.index[numerical_mask]
        categorical_indices = results.index[categorical_mask]

        for index in numerical_indices:
            feature = results.at[index, "feature"]
            results.at[index, "observed_only_psi"] = observed_only_psi(
                reference[feature], comparison[feature]
            )

        for index in categorical_indices:
            feature = results.at[index, "feature"]
            results.at[index, "observed_only_js"] = observed_only_js(
                reference[feature], comparison[feature]
            )

        pvalues = results.loc[numerical_indices, "ks_pvalue"].to_numpy(
            dtype=float
        )

        results.loc[numerical_indices, "ks_fdr_qvalue"] = (
            benjamini_hochberg(pvalues)
        )

        print("Numerical features:", len(numerical_indices))
        print("Categorical features:", len(categorical_indices))

    # Validate metric placement and numerical ranges.
    if results.loc[
        results["feature_type"] == "numerical", "observed_only_js"
    ].notna().any():
        raise ValueError("Categorical JS values found on numerical rows.")

    if results.loc[
        results["feature_type"] == "categorical", "observed_only_psi"
    ].notna().any():
        raise ValueError("Numerical PSI values found on categorical rows.")

    for column in ["observed_only_psi", "observed_only_js", "ks_fdr_qvalue"]:
        finite = results[column].dropna()
        if not np.isfinite(finite).all():
            raise ValueError(f"Non-finite values found in {column}.")

    if (
        (results["observed_only_js"].dropna() < 0).any()
        or (results["observed_only_js"].dropna() > 1).any()
    ):
        raise ValueError("JS divergence outside expected [0, 1] range.")

    summary = {
        "experiment": "enhanced_temporal_feature_distribution_drift",
        "target_loaded": False,
        "feature_count": len(feature_columns),
        "numerical_features": len(numerical_features),
        "categorical_features": len(categorical_features),
        "feature_comparisons": len(results),
        "partitions": {
            name: {
                "rows": int(len(frames[name])),
                "file": str(path.relative_to(ROOT)),
            }
            for name, path in PARTITIONS.items()
        },
        "methods": {
            "combined_numerical_psi": (
                "Reference-period quantile bins, including missingness "
                "as an explicit bin."
            ),
            "observed_only_numerical_psi": (
                "Reference-period quantile bins, calculated after excluding "
                "missing and non-finite observations."
            ),
            "combined_categorical_js": (
                "Jensen-Shannon divergence including missingness as a category."
            ),
            "observed_only_categorical_js": (
                "Jensen-Shannon divergence after excluding missing categories."
            ),
            "missingness": (
                "Comparison-period missing percentage minus "
                "reference-period missing percentage, in percentage points."
            ),
            "ks_multiple_testing": (
                "Benjamini-Hochberg false discovery rate correction, "
                "applied separately to the 400 numerical KS tests "
                "within each comparison."
            ),
        },
        "psi_quantile_bins": QUANTILE_BINS,
        "psi_smoothing_epsilon": EPSILON,
        "comparison_summaries": {},
        "interpretation_note": (
            "These metrics describe marginal feature-distribution changes "
            "P(X). They do not establish concept drift P(Y|X), nor do they "
            "prove that feature drift caused changes in predictive performance. "
            "PSI depends on binning choices; KS significance is sample-size "
            "sensitive. Observed-only metrics exclude missing values, so "
            "missingness changes must be interpreted separately."
        ),
        "detail_file": str(OUTPUT_FILE.relative_to(ROOT)),
    }

    for reference_name, comparison_name in comparisons:
        mask = (
            (results["reference_period"] == reference_name)
            & (results["comparison_period"] == comparison_name)
        )

        part = results[mask]
        numerical = part[part["feature_type"] == "numerical"]
        categorical = part[part["feature_type"] == "categorical"]

        summary["comparison_summaries"][
            f"{reference_name}_vs_{comparison_name}"
        ] = {
            "numerical_ks_mean": float(numerical["ks_statistic"].mean()),
            "numerical_ks_median": float(numerical["ks_statistic"].median()),
            "ks_bh_q_below_0_05": int(
                (numerical["ks_fdr_qvalue"] < 0.05).sum()
            ),
            "observed_only_psi_mean": float(
                numerical["observed_only_psi"].mean()
            ),
            "observed_only_psi_median": float(
                numerical["observed_only_psi"].median()
            ),
            "combined_psi_mean": float(numerical["psi"].mean()),
            "observed_only_js_mean": float(
                categorical["observed_only_js"].mean()
            ),
            "combined_js_mean": float(
                categorical["js_divergence"].mean()
            ),
            "mean_absolute_missingness_change_pp": float(
                part["missingness_change_pp"].abs().mean()
            ),
            "top_numerical_observed_drift": (
                numerical.nlargest(5, "observed_only_psi")[
                    ["feature", "observed_only_psi"]
                ].to_dict(orient="records")
            ),
            "top_categorical_observed_drift": (
                categorical.nlargest(5, "observed_only_js")[
                    ["feature", "observed_only_js"]
                ].to_dict(orient="records")
            ),
        }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    results.to_csv(OUTPUT_FILE, index=False)

    with SUMMARY_FILE.open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)

    print("\nENHANCED TEMPORAL DRIFT ANALYSIS COMPLETED")
    print("Rows:", len(results))
    print("Enhanced CSV:", OUTPUT_FILE)
    print("Enhanced summary:", SUMMARY_FILE)


if __name__ == "__main__":
    main()
