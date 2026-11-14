
"""Compare static and adaptive probability calibration."""

from pathlib import Path

import pandas as pd

from driftsentinel.calibration.calibrator import ProbabilityCalibrator
from driftsentinel.calibration.metrics import evaluate_calibration

ROOT = Path(__file__).resolve().parents[1]

PREDICTION_DIR = (
    ROOT / "experiments" / "results" / "temporal_predictions"
)
RESULT_DIR = ROOT / "experiments" / "results"

CALIBRATION_FILE = "initial_calibration_predictions.parquet"
ADAPTATION_FILE = "adaptation_predictions.parquet"
EVALUATION_FILE = "evaluation_predictions.parquet"


def load_predictions(filename: str) -> pd.DataFrame:
    path = PREDICTION_DIR / filename

    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    df = pd.read_parquet(path)

    required = {"isFraud", "probability"}
    missing = required - set(df.columns)

    if missing:
        raise ValueError(f"Missing columns in {filename}: {missing}")

    return df


def main() -> None:
    calibration_df = load_predictions(CALIBRATION_FILE)
    adaptation_df = load_predictions(ADAPTATION_FILE)
    evaluation_df = load_predictions(EVALUATION_FILE)

    y_calibration = calibration_df["isFraud"]
    p_calibration = calibration_df["probability"]

    y_adaptation = adaptation_df["isFraud"]
    p_adaptation = adaptation_df["probability"]

    y_evaluation = evaluation_df["isFraud"]
    p_evaluation = evaluation_df["probability"]

    # Static calibrators: fitted on the initial calibration window.
    static_sigmoid = ProbabilityCalibrator(method="sigmoid")
    static_sigmoid.fit(y_calibration, p_calibration)

    static_isotonic = ProbabilityCalibrator(method="isotonic")
    static_isotonic.fit(y_calibration, p_calibration)

    # Adaptive calibrators: fitted on the newer adaptation window.
    adaptive_sigmoid = ProbabilityCalibrator(method="sigmoid")
    adaptive_sigmoid.fit(y_adaptation, p_adaptation)

    adaptive_isotonic = ProbabilityCalibrator(method="isotonic")
    adaptive_isotonic.fit(y_adaptation, p_adaptation)

    # Evaluate every strategy on exactly the same future window.
    strategies = {
        "uncalibrated": p_evaluation,
        "static_sigmoid": static_sigmoid.predict(p_evaluation),
        "static_isotonic": static_isotonic.predict(p_evaluation),
        "adaptive_sigmoid": adaptive_sigmoid.predict(p_evaluation),
        "adaptive_isotonic": adaptive_isotonic.predict(p_evaluation),
    }

    results = []

    for name, probabilities in strategies.items():
        metrics = evaluate_calibration(
            y_evaluation,
            probabilities,
        )

        results.append({
            "strategy": name,
            **metrics,
        })

    results_df = pd.DataFrame(results)

    output_path = RESULT_DIR / "adaptive_calibration_comparison.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(output_path, index=False)

    print("\nADAPTIVE CALIBRATION COMPARISON")
    print("=" * 85)
    print(results_df.to_string(index=False))
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
