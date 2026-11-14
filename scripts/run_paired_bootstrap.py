
"""Paired bootstrap evaluation of four temporal fraud models."""

import json
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)

ROOT = Path(__file__).resolve().parents[1]

PREDICTION_DIR = ROOT / "experiments/results/temporal_predictions"
RESULT_DIR = ROOT / "experiments/results"

MODELS = [
    "frozen",
    "recent_only",
    "matched_size",
    "retrained",
]

METRICS = [
    "average_precision",
    "roc_auc",
    "brier_score",
    "log_loss",
]

N_BOOTSTRAPS = 2_000
RANDOM_SEED = 42
CONFIDENCE_LEVEL = 0.95


def load_predictions(model: str) -> pd.DataFrame:
    """Load evaluation predictions for one model."""

    path = PREDICTION_DIR / f"evaluation_{model}_predictions.parquet"

    if not path.exists():
        raise FileNotFoundError(path)

    df = pd.read_parquet(path)

    required = {
        "TransactionID",
        "TransactionDT",
        "isFraud",
        "probability",
    }

    missing = required - set(df.columns)

    if missing:
        raise ValueError(f"{model}: missing columns {missing}")

    if df["TransactionID"].duplicated().any():
        raise ValueError(f"{model}: duplicate transaction IDs")

    if not df["probability"].between(0, 1).all():
        raise ValueError(f"{model}: invalid probabilities")

    return df


def calculate_metrics(y_true: np.ndarray, probabilities: np.ndarray) -> dict:
    """Calculate classification and probability-quality metrics."""

    return {
        "average_precision": float(
            average_precision_score(y_true, probabilities)
        ),
        "roc_auc": float(
            roc_auc_score(y_true, probabilities)
        ),
        "brier_score": float(
            brier_score_loss(y_true, probabilities)
        ),
        "log_loss": float(
            log_loss(y_true, probabilities, labels=[0, 1])
        ),
    }


def load_aligned_predictions():
    """Load all models and verify identical evaluation observations."""

    frames = {
        model: load_predictions(model)
        for model in MODELS
    }

    reference = frames["frozen"]

    for model, candidate in frames.items():
        if not reference["TransactionID"].equals(
            candidate["TransactionID"]
        ):
            raise ValueError(f"Transaction ID mismatch: {model}")

        if not reference["TransactionDT"].equals(
            candidate["TransactionDT"]
        ):
            raise ValueError(f"Timestamp mismatch: {model}")

        if not reference["isFraud"].equals(
            candidate["isFraud"]
        ):
            raise ValueError(f"Target mismatch: {model}")

    y_true = reference["isFraud"].to_numpy(dtype=np.int8)

    probabilities = {
        model: frames[model]["probability"].to_numpy(dtype=float)
        for model in MODELS
    }

    return reference, y_true, probabilities


def confidence_interval(values: list[float]) -> tuple[float, float]:
    """Calculate a percentile confidence interval."""

    alpha = 1.0 - CONFIDENCE_LEVEL

    lower = np.percentile(values, 100 * alpha / 2)
    upper = np.percentile(values, 100 * (1 - alpha / 2))

    return float(lower), float(upper)


def main() -> None:
    reference, y_true, probabilities = load_aligned_predictions()

    n_samples = len(y_true)

    if len(np.unique(y_true)) != 2:
        raise ValueError("Evaluation data must contain both classes.")

    print("PAIRED BOOTSTRAP EVALUATION")
    print("=" * 65)
    print(f"Evaluation observations: {n_samples:,}")
    print(f"Fraud cases: {int(y_true.sum()):,}")
    print(f"Bootstrap repetitions: {N_BOOTSTRAPS:,}")
    print(f"Random seed: {RANDOM_SEED}")
    print("Resampling: paired, with replacement")

    # Point estimates on the original evaluation window.
    point_estimates = {
        model: calculate_metrics(y_true, probabilities[model])
        for model in MODELS
    }

    # Store bootstrap metric values for every model.
    bootstrap_values = {
        model: {metric: [] for metric in METRICS}
        for model in MODELS
    }

    rng = np.random.default_rng(RANDOM_SEED)

    for iteration in range(N_BOOTSTRAPS):
        # One shared sample for all four models.
        indices = rng.integers(
            low=0,
            high=n_samples,
            size=n_samples,
        )

        sampled_labels = y_true[indices]

        # Extremely unlikely, but guard against one-class samples.
        if len(np.unique(sampled_labels)) != 2:
            continue

        for model in MODELS:
            sampled_probabilities = probabilities[model][indices]

            metrics = calculate_metrics(
                sampled_labels,
                sampled_probabilities,
            )

            for metric in METRICS:
                bootstrap_values[model][metric].append(
                    metrics[metric]
                )

        if (iteration + 1) % 250 == 0:
            print(
                f"Completed {iteration + 1:,}/{N_BOOTSTRAPS:,} "
                "bootstrap iterations."
            )

    # Model-level confidence intervals.
    model_results = []

    for model in MODELS:
        for metric in METRICS:
            lower, upper = confidence_interval(
                bootstrap_values[model][metric]
            )

            model_results.append({
                "model": model,
                "metric": metric,
                "point_estimate": point_estimates[model][metric],
                "ci_lower": lower,
                "ci_upper": upper,
                "valid_bootstraps": len(
                    bootstrap_values[model][metric]
                ),
            })

    model_results_df = pd.DataFrame(model_results)

    # Paired differences for every model pair.
    # Positive is better for AP and ROC-AUC.
    # Negative is better for Brier score and log loss.
    difference_results = []

    for model_a, model_b in combinations(MODELS, 2):
        for metric in METRICS:
            differences = (
                np.asarray(bootstrap_values[model_b][metric])
                - np.asarray(bootstrap_values[model_a][metric])
            )

            lower, upper = confidence_interval(
                differences.tolist()
            )

            point_difference = (
                point_estimates[model_b][metric]
                - point_estimates[model_a][metric]
            )

            difference_results.append({
                "reference_model": model_a,
                "comparison_model": model_b,
                "metric": metric,
                "point_difference": point_difference,
                "ci_lower": lower,
                "ci_upper": upper,
                "valid_bootstraps": len(differences),
            })

    differences_df = pd.DataFrame(difference_results)

    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    model_output = RESULT_DIR / "bootstrap_model_intervals.csv"
    difference_output = (
        RESULT_DIR / "bootstrap_paired_differences.csv"
    )
    metadata_output = RESULT_DIR / "bootstrap_metadata.json"

    model_results_df.to_csv(model_output, index=False)
    differences_df.to_csv(difference_output, index=False)

    metadata = {
        "experiment": "paired_bootstrap_temporal_models",
        "evaluation_rows": n_samples,
        "fraud_cases": int(y_true.sum()),
        "fraud_rate": float(y_true.mean()),
        "models": MODELS,
        "metrics": METRICS,
        "bootstrap_repetitions_requested": N_BOOTSTRAPS,
        "bootstrap_repetitions_valid": min(
            len(bootstrap_values[model][metric])
            for model in MODELS
            for metric in METRICS
        ),
        "random_seed": RANDOM_SEED,
        "confidence_level": CONFIDENCE_LEVEL,
        "resampling": "paired bootstrap with replacement",
        "evaluation_timestamp_start": int(
            reference["TransactionDT"].min()
        ),
        "evaluation_timestamp_end": int(
            reference["TransactionDT"].max()
        ),
    }

    metadata_output.write_text(
        json.dumps(metadata, indent=4),
        encoding="utf-8",
    )

    print("\nMODEL CONFIDENCE INTERVALS")
    print("=" * 100)
    print(model_results_df.to_string(index=False))

    print("\nPAIRED MODEL DIFFERENCES")
    print("=" * 100)
    print(differences_df.to_string(index=False))

    print("\nResults saved:")
    print(model_output)
    print(difference_output)
    print(metadata_output)


if __name__ == "__main__":
    main()
