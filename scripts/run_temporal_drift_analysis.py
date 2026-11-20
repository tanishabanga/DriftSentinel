import json
from pathlib import Path

import pandas as pd

from driftsentinel.monitoring.drift import evaluate_feature_drift

ROOT = Path(".")
PROFILE_PATH = ROOT / "artifacts/monitoring/reference_profile.json"
SCHEMA_PATH = ROOT / "artifacts/xgboost/feature_schema.json"
OUTPUT_DIR = ROOT / "artifacts/monitoring/temporal"

WINDOWS = [
    "initial_calibration",
    "adaptation",
    "evaluation",
]

PSI_PARITY_TOLERANCE = 1e-10


def unseen_category_metrics(df, categorical_columns, vocabularies):
    unseen_matrix = pd.DataFrame(index=df.index)
    feature_rows = []

    for column in categorical_columns:
        if column not in df.columns:
            raise ValueError(f"Missing categorical feature: {column}")

        vocabulary = {str(value) for value in vocabularies[column]}

        observed = df[column].astype("string")
        unseen_mask = observed.notna() & ~observed.isin(vocabulary)

        unseen_matrix[column] = unseen_mask

        feature_rows.append({
            "feature": column,
            "unseen_categories": int(observed[unseen_mask].nunique()),
            "unseen_rows": int(unseen_mask.sum()),
            "unseen_row_rate": float(unseen_mask.mean()),
            "missing_rate": float(df[column].isna().mean()),
        })

    feature_report = pd.DataFrame(feature_rows).sort_values(
        "unseen_row_rate",
        ascending=False,
    )

    affected = int(unseen_matrix.any(axis=1).sum())
    occurrences = int(unseen_matrix.sum().sum())

    return feature_report, {
        "transactions_with_unseen_categories": affected,
        "affected_transaction_rate": affected / len(df),
        "unseen_feature_row_occurrences": occurrences,
        "features_with_unseen_categories": int(
            (feature_report["unseen_categories"] > 0).sum()
        ),
    }


def compare_legacy_psi(new_report, legacy_path, window):
    """Compare new PSI values with the previously generated report."""
    if not legacy_path.exists():
        print(f"WARNING: No legacy PSI report found for {window}: {legacy_path}")
        return None

    legacy = pd.read_csv(legacy_path)

    required_columns = {"feature", "psi"}
    if not required_columns.issubset(legacy.columns):
        raise ValueError(
            f"Legacy report for {window} lacks required columns: "
            f"{sorted(required_columns - set(legacy.columns))}"
        )

    if legacy["feature"].duplicated().any():
        raise ValueError(f"Legacy PSI report contains duplicate features: {window}")

    comparison = new_report[["feature", "psi"]].merge(
        legacy[["feature", "psi"]],
        on="feature",
        how="outer",
        suffixes=("_new", "_legacy"),
        indicator=True,
        validate="one_to_one",
    )

    if not (comparison["_merge"] == "both").all():
        raise ValueError(
            f"Feature sets differ between new and legacy PSI reports for {window}."
        )

    differences = (
        comparison["psi_new"] - comparison["psi_legacy"]
    ).abs()

    max_difference = float(differences.max())

    print(f"Legacy PSI parity for {window}:")
    print(f"  Features compared: {len(comparison)}")
    print(f"  Maximum absolute PSI difference: {max_difference:.12g}")

    if max_difference > PSI_PARITY_TOLERANCE:
        worst = comparison.loc[differences.idxmax()]
        raise AssertionError(
            f"PSI parity failed for {window}. "
            f"Feature={worst['feature']}, "
            f"new={worst['psi_new']}, "
            f"legacy={worst['psi_legacy']}, "
            f"difference={max_difference}"
        )

    print("  Result: PASS")
    return max_difference


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not PROFILE_PATH.exists():
        raise FileNotFoundError(PROFILE_PATH)

    if not SCHEMA_PATH.exists():
        raise FileNotFoundError(SCHEMA_PATH)

    reference_profile = json.loads(
        PROFILE_PATH.read_text(encoding="utf-8")
    )

    schema = json.loads(
        SCHEMA_PATH.read_text(encoding="utf-8")
    )

    categorical_columns = schema["categorical_columns"]
    vocabularies = schema["categorical_vocabularies"]

    summary_rows = []
    pending_reports = {}

    for window in WINDOWS:
        print("\n" + "=" * 70)
        print(f"TEMPORAL WINDOW: {window.upper()}")
        print("=" * 70)

        input_path = ROOT / "data/processed/temporal" / f"{window}.parquet"

        if not input_path.exists():
            raise FileNotFoundError(input_path)

        df = pd.read_parquet(input_path)

        if df.empty:
            raise ValueError(f"{window} is empty.")

        if "TransactionDT" not in df.columns:
            raise ValueError(f"{window} is missing TransactionDT.")

        print(f"Rows: {len(df):,}")
        print(
            f"TransactionDT: {df['TransactionDT'].min()} "
            f"to {df['TransactionDT'].max()}"
        )

        # Evaluate feature drift using the reusable monitoring engine.
        psi_report = evaluate_feature_drift(df, reference_profile)
        psi_report = psi_report.sort_values(
            "psi",
            ascending=False,
        ).reset_index(drop=True)

        if len(psi_report) != reference_profile["feature_count"]:
            raise ValueError(
                f"Expected {reference_profile['feature_count']} features, "
                f"evaluated {len(psi_report)} for {window}."
            )

        # Compare against the existing legacy report before overwriting it.
        psi_path = OUTPUT_DIR / f"{window}_psi.csv"
        parity_difference = compare_legacy_psi(
            psi_report,
            psi_path,
            window,
        )

        # Calculate unseen categorical values.
        unseen_report, unseen_summary = unseen_category_metrics(
            df,
            categorical_columns,
            vocabularies,
        )

        unseen_path = OUTPUT_DIR / f"{window}_unseen_categories.csv"

        # The new engine treats non-finite numeric values as missing.
        mean_absolute_missing_change = float(
            psi_report["missing_rate_change"].abs().mean()
        )

        summary = {
            "window": window,
            "rows": len(df),
            "timestamp_min": int(df["TransactionDT"].min()),
            "timestamp_max": int(df["TransactionDT"].max()),
            "mean_psi": float(psi_report["psi"].mean()),
            "median_psi": float(psi_report["psi"].median()),
            "max_psi": float(psi_report["psi"].max()),
            "features_psi_gt_010": int(
                (psi_report["psi"] > 0.10).sum()
            ),
            "features_psi_gt_025": int(
                (psi_report["psi"] > 0.25).sum()
            ),
            "mean_absolute_missing_rate_change": (
                mean_absolute_missing_change
            ),
            "psi_parity_max_abs_difference": parity_difference,
            **unseen_summary,
        }

        summary_rows.append(summary)

        pending_reports[window] = {
            "psi": psi_report,
            "unseen": unseen_report,
            "unseen_path": unseen_path,
            "psi_path": psi_path,
        }

        print("\nUNSEEN CATEGORY SUMMARY")
        print(
            "Features affected: "
            f"{unseen_summary['features_with_unseen_categories']}"
        )
        print(
            "Transactions affected: "
            f"{unseen_summary['transactions_with_unseen_categories']:,}"
        )
        print(
            "Affected transaction rate: "
            f"{unseen_summary['affected_transaction_rate']:.4%}"
        )

    # Save reports only after every available parity check has passed.
    for reports in pending_reports.values():
        reports["psi"].to_csv(reports["psi_path"], index=False)
        reports["unseen"].to_csv(reports["unseen_path"], index=False)

    summary_df = pd.DataFrame(summary_rows)
    summary_path = OUTPUT_DIR / "temporal_drift_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    print("\n" + "=" * 70)
    print("COMBINED TEMPORAL DRIFT SUMMARY")
    print("=" * 70)
    print(summary_df.round(8).to_string(index=False))

    print(f"\nSaved combined summary: {summary_path}")
    print(f"All temporal reports: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
