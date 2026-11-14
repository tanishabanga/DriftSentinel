
"""Compare uncalibrated, sigmoid and isotonic probabilities."""

import json
from pathlib import Path

import pandas as pd

from driftsentinel.calibration.calibrator import ProbabilityCalibrator
from driftsentinel.calibration.metrics import evaluate_calibration


def main():
    input_path = Path(
        "experiments/results/logistic_regression_pilot_predictions.parquet"
    )
    output_path = Path(
        "experiments/results/calibration_comparison.json"
    )

    print("Loading saved validation predictions...")
    df = pd.read_parquet(input_path)
    df = df.sort_values("TransactionDT").reset_index(drop=True)

    # Chronological separation.
    calibration_df = df.iloc[:2500].copy()
    evaluation_df = df.iloc[2500:].copy()

    y_calibration = calibration_df["isFraud"].to_numpy()
    p_calibration = calibration_df["probability"].to_numpy()

    y_evaluation = evaluation_df["isFraud"].to_numpy()
    p_evaluation = evaluation_df["probability"].to_numpy()

    print(f"Calibration rows: {len(calibration_df):,}")
    print(f"Evaluation rows: {len(evaluation_df):,}")

    if len(set(y_calibration)) < 2 or len(set(y_evaluation)) < 2:
        raise ValueError("Both segments must contain both classes.")

    results = {
        "experiment": "probability_calibration_pilot",
        "status": "pilot_not_final",
        "calibration_rows": len(calibration_df),
        "evaluation_rows": len(evaluation_df),
        "methods": {},
    }

    # Original probabilities: control condition.
    results["methods"]["uncalibrated"] = evaluate_calibration(
        y_evaluation,
        p_evaluation,
    )

    # Fit each calibrator on the earlier segment.
    for method in ("sigmoid", "isotonic"):
        print(f"\nFitting {method} calibrator...")

        calibrator = ProbabilityCalibrator(method=method)
        calibrator.fit(y_calibration, p_calibration)

        calibrated_probabilities = calibrator.predict(p_evaluation)

        results["methods"][method] = evaluate_calibration(
            y_evaluation,
            calibrated_probabilities,
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(results, indent=4),
        encoding="utf-8",
    )

    print("\nCALIBRATION COMPARISON")

    comparison = pd.DataFrame(results["methods"]).T
    print(comparison.round(5).to_string())

    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
