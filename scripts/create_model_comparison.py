
"""Create a consolidated validation model comparison."""

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "experiments/results"

EXPERIMENT_FILES = [
    "logistic_regression_50k_benchmark.json",
    "logistic_regression_100k_benchmark.json",
    "logistic_regression_200k_benchmark.json",
    "xgboost_full_metadata.json",
]

OUTPUT_CSV = RESULTS_DIR / "model_comparison.csv"
OUTPUT_JSON = RESULTS_DIR / "model_comparison_summary.json"


def load_experiment(filename: str) -> dict:
    """Load one completed experiment."""
    path = RESULTS_DIR / filename

    if not path.exists():
        raise FileNotFoundError(f"Missing experiment: {path}")

    data = json.loads(path.read_text(encoding="utf-8"))

    if data.get("status") not in {"completed", "validation_evaluated"}:
        raise ValueError(
            f"Experiment is not complete: {filename}"
        )

    metrics = data["metrics"]
    sampling = data["sampling"]
    resources = data["resources"]

    return {
        "experiment": data["experiment"],
        "model": data["model"],
        "training_rows": sampling["training_rows"],
        "validation_rows": sampling["validation_rows"],
        "training_fraud_rate": sampling["training_fraud_rate"],
        "validation_fraud_rate": sampling["validation_fraud_rate"],
        "average_precision": metrics["average_precision"],
        "roc_auc": metrics["roc_auc"],
        "brier_score": metrics["brier_score"],
        "log_loss": metrics["log_loss"],
        "training_seconds": resources["training_seconds"],
        "inference_seconds": resources["inference_seconds"],
        "convergence_warnings": resources.get(
            "convergence_warnings"
        ),
        "iterations": resources.get("iterations"),
        "threshold": data.get("threshold", 0.5),
        "sampling_method": sampling["method"],
    }


def main() -> None:
    rows = [
        load_experiment(filename)
        for filename in EXPERIMENT_FILES
    ]

    comparison = pd.DataFrame(rows)

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)

    comparison.to_csv(OUTPUT_CSV, index=False)

    summary = {
        "purpose": "Validation performance comparison",
        "validation_rows": 88581,
        "test_set_used": False,
        "important_limitation": (
            "Logistic Regression experiments use chronological "
            "training prefixes of different sizes. XGBoost uses "
            "the full training partition. Differences in performance "
            "cannot be attributed solely to model choice."
        ),
        "experiments": rows,
    }

    OUTPUT_JSON.write_text(
        json.dumps(summary, indent=4),
        encoding="utf-8",
    )

    print("\nMODEL COMPARISON")
    print(
        comparison[
            [
                "experiment",
                "training_rows",
                "average_precision",
                "roc_auc",
                "brier_score",
                "log_loss",
                "training_seconds",
            ]
        ].to_string(index=False)
    )

    print(f"\nCSV saved: {OUTPUT_CSV}")
    print(f"Summary saved: {OUTPUT_JSON}")


if __name__ == "__main__":
    main()
