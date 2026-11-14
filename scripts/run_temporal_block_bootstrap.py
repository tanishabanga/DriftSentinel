
"""Paired moving block bootstrap for temporal fraud model evaluation."""

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

MODELS = ["frozen", "recent_only", "matched_size", "retrained"]

BLOCK_LENGTHS = [250, 500, 1000]
N_BOOTSTRAPS = 1000
RANDOM_SEED = 42


def load_predictions(model: str) -> pd.DataFrame:
    path = PREDICTION_DIR / f"evaluation_{model}_predictions.parquet"
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

    return df


def load_aligned_predictions():
    frames = {model: load_predictions(model) for model in MODELS}
    reference = frames["frozen"]

    for model, candidate in frames.items():
        for column in ["TransactionID", "TransactionDT", "isFraud"]:
            if not reference[column].equals(candidate[column]):
                raise ValueError(
                    f"{column} mismatch for model {model}"
                )

    y = reference["isFraud"].to_numpy(dtype=np.int8)

    probabilities = {
        model: frames[model]["probability"].to_numpy(dtype=float)
        for model in MODELS
    }

    return y, probabilities


def calculate_metrics(y, probabilities):
    return {
        "average_precision": float(
            average_precision_score(y, probabilities)
        ),
        "roc_auc": float(roc_auc_score(y, probabilities)),
        "brier_score": float(
            brier_score_loss(y, probabilities)
        ),
        "log_loss": float(
            log_loss(y, probabilities, labels=[0, 1])
        ),
    }


def generate_block_sample(
    n_samples: int,
    block_length: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample contiguous, non-circular blocks with replacement."""

    if block_length <= 0 or block_length > n_samples:
        raise ValueError("Invalid block length.")

    n_blocks = int(np.ceil(n_samples / block_length))

    starts = rng.integers(
        low=0,
        high=n_samples - block_length + 1,
        size=n_blocks,
    )

    indices = np.concatenate([
        np.arange(start, start + block_length)
        for start in starts
    ])

    return indices[:n_samples]


def confidence_interval(values):
    return (
        float(np.percentile(values, 2.5)),
        float(np.percentile(values, 97.5)),
    )


def run_bootstrap(y, probabilities, block_length):
    n_samples = len(y)

    model_values = {
        model: {
            metric: []
            for metric in [
                "average_precision",
                "roc_auc",
                "brier_score",
                "log_loss",
            ]
        }
        for model in MODELS
    }

    rng = np.random.default_rng(
        RANDOM_SEED + block_length
    )

    valid = 0

    for iteration in range(N_BOOTSTRAPS):
        indices = generate_block_sample(
            n_samples,
            block_length,
            rng,
        )

        sampled_y = y[indices]

        if len(np.unique(sampled_y)) < 2:
            continue

        valid += 1

        for model in MODELS:
            sampled_probabilities = probabilities[model][indices]

            metrics = calculate_metrics(
                sampled_y,
                sampled_probabilities,
            )

            for metric, value in metrics.items():
                model_values[model][metric].append(value)

        if (iteration + 1) % 250 == 0:
            print(
                f"Block {block_length}: "
                f"{iteration + 1}/{N_BOOTSTRAPS}"
            )

    if valid < N_BOOTSTRAPS * 0.95:
        raise RuntimeError(
            f"Too many invalid bootstrap samples: {valid}"
        )

    return model_values, valid


def main():
    y, probabilities = load_aligned_predictions()

    print("PAIRED TEMPORAL BLOCK BOOTSTRAP")
    print("=" * 65)
    print(f"Evaluation observations: {len(y):,}")
    print(f"Fraud cases: {int(y.sum()):,}")
    print(f"Fraud rate: {y.mean():.4%}")

    point_estimates = {
        model: calculate_metrics(y, probabilities[model])
        for model in MODELS
    }

    all_model_results = []
    all_difference_results = []

    for block_length in BLOCK_LENGTHS:
        print(f"\nRunning block length: {block_length}")

        model_values, valid = run_bootstrap(
            y,
            probabilities,
            block_length,
        )

        for model in MODELS:
            for metric, values in model_values[model].items():
                lower, upper = confidence_interval(values)

                all_model_results.append({
                    "block_length": block_length,
                    "model": model,
                    "metric": metric,
                    "point_estimate": point_estimates[model][metric],
                    "ci_lower": lower,
                    "ci_upper": upper,
                    "valid_bootstraps": valid,
                })

        for model_a, model_b in combinations(MODELS, 2):
            for metric in model_values[model_a]:
                differences = (
                    np.asarray(model_values[model_b][metric])
                    - np.asarray(model_values[model_a][metric])
                )

                lower, upper = confidence_interval(differences)

                all_difference_results.append({
                    "block_length": block_length,
                    "reference_model": model_a,
                    "comparison_model": model_b,
                    "metric": metric,
                    "point_difference": (
                        point_estimates[model_b][metric]
                        - point_estimates[model_a][metric]
                    ),
                    "ci_lower": lower,
                    "ci_upper": upper,
                    "valid_bootstraps": valid,
                })

    model_df = pd.DataFrame(all_model_results)
    difference_df = pd.DataFrame(all_difference_results)

    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    model_path = RESULT_DIR / "temporal_block_bootstrap_intervals.csv"
    difference_path = (
        RESULT_DIR / "temporal_block_bootstrap_differences.csv"
    )

    model_df.to_csv(model_path, index=False)
    difference_df.to_csv(difference_path, index=False)

    print("\nMODEL CONFIDENCE INTERVALS")
    print(model_df.to_string(index=False))

    print("\nPAIRED MODEL DIFFERENCES")
    print(difference_df.to_string(index=False))

    print("\nSaved:")
    print(model_path)
    print(difference_path)


if __name__ == "__main__":
    main()
