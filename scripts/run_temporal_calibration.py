
"""Evaluate initial probability calibration on a future time window."""

import json
from pathlib import Path

import pandas as pd

from driftsentinel.calibration.calibrator import ProbabilityCalibrator
from driftsentinel.calibration.metrics import evaluate_calibration

PREDICTION_DIR = Path(
    "experiments/results/temporal_predictions"
)

OUTPUT_PATH = Path(
    "experiments/results/temporal_static_calibration.json"
)


def main():
    print("Loading temporal predictions...")

    initial = pd.read_parquet(
        PREDICTION_DIR / "initial_calibration_predictions.parquet"
    )

    evaluation = pd.read_parquet(
        PREDICTION_DIR / "evaluation_predictions.parquet"
    )

    y_initial = initial["isFraud"].to_numpy()
    p_initial = initial["probability"].to_numpy()

    y_evaluation = evaluation["isFraud"].to_numpy()
    p_evaluation = evaluation["probability"].to_numpy()

    results = {}

    # Uncalibrated frozen model.
    results["uncalibrated"] = evaluate_calibration(
        y_evaluation,
        p_evaluation,
    )

    # Fit each calibrator using only the initial window.
    for method in ["sigmoid", "isotonic"]:
        print(f"\nFitting {method} calibrator...")

        calibrator = ProbabilityCalibrator(method=method)
        calibrator.fit(y_initial, p_initial)

        calibrated_probabilities = calibrator.predict(
            p_evaluation
        )

        results[method] = evaluate_calibration(
            y_evaluation,
            calibrated_probabilities,
        )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with OUTPUT_PATH.open("w", encoding="utf-8") as file:
        json.dump(results, file, indent=4)

    comparison = pd.DataFrame(results).T
    comparison.index.name = "method"

    print("\nOUT-OF-TIME CALIBRATION COMPARISON")
    print(comparison.round(6).to_string())

    print(f"\nResults saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
