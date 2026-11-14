
"""Run cost-matrix and review-capacity sensitivity experiments."""

from pathlib import Path

import numpy as np
import pandas as pd

from driftsentinel.calibration.calibrator import ProbabilityCalibrator
from driftsentinel.policies.cost_evaluation import evaluate_cost_policy
from driftsentinel.policies.cost_scenarios import COST_SCENARIOS

ROOT = Path(__file__).resolve().parents[1]

PREDICTION_DIR = (
    ROOT / "experiments" / "results" / "temporal_predictions"
)

RESULT_DIR = ROOT / "experiments" / "results"

CAPACITY_RATES = [
    0.0,
    0.0025,
    0.005,
    0.01,
    0.02,
    0.03,
]


def load_predictions(filename: str) -> pd.DataFrame:
    """Load and validate a prediction file."""

    path = PREDICTION_DIR / filename

    if not path.exists():
        raise FileNotFoundError(f"Prediction file not found: {path}")

    df = pd.read_parquet(path)

    required = {
        "TransactionID",
        "TransactionDT",
        "isFraud",
        "probability",
    }

    missing = required - set(df.columns)

    if missing:
        raise ValueError(f"Missing columns in {filename}: {missing}")

    if df["TransactionID"].duplicated().any():
        raise ValueError(f"Duplicate transaction IDs in {filename}.")

    if not df["isFraud"].isin([0, 1]).all():
        raise ValueError(f"Invalid labels in {filename}.")

    probabilities = df["probability"].to_numpy(dtype=float)

    if not np.isfinite(probabilities).all():
        raise ValueError(f"Non-finite probabilities in {filename}.")

    if ((probabilities < 0) | (probabilities > 1)).any():
        raise ValueError(f"Invalid probabilities in {filename}.")

    return df


def validate_same_transactions(
    reference: pd.DataFrame,
    candidate: pd.DataFrame,
) -> None:
    """Verify identical transaction IDs, timestamps and labels."""

    columns = ["TransactionID", "TransactionDT", "isFraud"]

    reference = (
        reference[columns]
        .sort_values("TransactionID")
        .reset_index(drop=True)
    )

    candidate = (
        candidate[columns]
        .sort_values("TransactionID")
        .reset_index(drop=True)
    )

    for column in columns:
        if not reference[column].equals(candidate[column]):
            raise ValueError(
                f"Prediction files differ in column: {column}"
            )


def build_strategies() -> tuple[pd.Series, dict[str, np.ndarray]]:
    """Reconstruct the 14 existing prediction strategies."""

    initial = load_predictions(
        "initial_calibration_predictions.parquet"
    )

    adaptation_frozen = load_predictions(
        "adaptation_calibration_frozen_predictions.parquet"
    )

    adaptation_retrained = load_predictions(
        "adaptation_calibration_retrained_predictions.parquet"
    )

    adaptation_recent = load_predictions(
        "adaptation_calibration_recent_only_predictions.parquet"
    )

    adaptation_matched = load_predictions(
        "adaptation_calibration_matched_size_predictions.parquet"
    )

    evaluation_frozen = load_predictions(
        "evaluation_frozen_predictions.parquet"
    )

    evaluation_retrained = load_predictions(
        "evaluation_retrained_predictions.parquet"
    )

    evaluation_recent = load_predictions(
        "evaluation_recent_only_predictions.parquet"
    )

    evaluation_matched = load_predictions(
        "evaluation_matched_size_predictions.parquet"
    )

    # Validate all calibration-window prediction alignment.
    for candidate in [
        adaptation_retrained,
        adaptation_recent,
        adaptation_matched,
    ]:
        validate_same_transactions(adaptation_frozen, candidate)

    # Validate all evaluation-window prediction alignment.
    for candidate in [
        evaluation_retrained,
        evaluation_recent,
        evaluation_matched,
    ]:
        validate_same_transactions(evaluation_frozen, candidate)

    y_initial = initial["isFraud"]
    y_adaptation = adaptation_frozen["isFraud"]
    y_evaluation = evaluation_frozen["isFraud"]

    p_initial = initial["probability"]

    p_adaptation_frozen = adaptation_frozen["probability"]
    p_adaptation_retrained = adaptation_retrained["probability"]
    p_adaptation_recent = adaptation_recent["probability"]
    p_adaptation_matched = adaptation_matched["probability"]

    p_evaluation_frozen = evaluation_frozen["probability"].to_numpy()
    p_evaluation_retrained = evaluation_retrained["probability"].to_numpy()
    p_evaluation_recent = evaluation_recent["probability"].to_numpy()
    p_evaluation_matched = evaluation_matched["probability"].to_numpy()

    # Static calibration: initial calibration window.
    static_sigmoid = ProbabilityCalibrator("sigmoid").fit(
        y_initial, p_initial
    )

    static_isotonic = ProbabilityCalibrator("isotonic").fit(
        y_initial, p_initial
    )

    # Adaptive calibration: adaptation calibration window.
    adaptive_sigmoid = ProbabilityCalibrator("sigmoid").fit(
        y_adaptation, p_adaptation_frozen
    )

    adaptive_isotonic = ProbabilityCalibrator("isotonic").fit(
        y_adaptation, p_adaptation_frozen
    )

    # Retrained model calibration.
    retrained_sigmoid = ProbabilityCalibrator("sigmoid").fit(
        y_adaptation, p_adaptation_retrained
    )

    retrained_isotonic = ProbabilityCalibrator("isotonic").fit(
        y_adaptation, p_adaptation_retrained
    )

    # Recent-only model calibration.
    recent_sigmoid = ProbabilityCalibrator("sigmoid").fit(
        y_adaptation, p_adaptation_recent
    )

    recent_isotonic = ProbabilityCalibrator("isotonic").fit(
        y_adaptation, p_adaptation_recent
    )

    # Matched-size model calibration.
    matched_sigmoid = ProbabilityCalibrator("sigmoid").fit(
        y_adaptation, p_adaptation_matched
    )

    matched_isotonic = ProbabilityCalibrator("isotonic").fit(
        y_adaptation, p_adaptation_matched
    )

    strategies = {
        "frozen_uncalibrated": p_evaluation_frozen,
        "frozen_static_sigmoid": static_sigmoid.predict(
            p_evaluation_frozen
        ),
        "frozen_static_isotonic": static_isotonic.predict(
            p_evaluation_frozen
        ),
        "frozen_adaptive_sigmoid": adaptive_sigmoid.predict(
            p_evaluation_frozen
        ),
        "frozen_adaptive_isotonic": adaptive_isotonic.predict(
            p_evaluation_frozen
        ),
        "retrained_uncalibrated": p_evaluation_retrained,
        "retrained_sigmoid": retrained_sigmoid.predict(
            p_evaluation_retrained
        ),
        "retrained_isotonic": retrained_isotonic.predict(
            p_evaluation_retrained
        ),
        "recent_only_uncalibrated": p_evaluation_recent,
        "recent_only_sigmoid": recent_sigmoid.predict(
            p_evaluation_recent
        ),
        "recent_only_isotonic": recent_isotonic.predict(
            p_evaluation_recent
        ),
        "matched_size_uncalibrated": p_evaluation_matched,
        "matched_size_sigmoid": matched_sigmoid.predict(
            p_evaluation_matched
        ),
        "matched_size_isotonic": matched_isotonic.predict(
            p_evaluation_matched
        ),
    }

    return y_evaluation.reset_index(drop=True), strategies


def run_experiment() -> pd.DataFrame:
    """Evaluate every strategy, cost scenario and review policy."""

    y_true, strategies = build_strategies()

    total_transactions = len(y_true)

    policies = [
        {
            "policy": "unconstrained",
            "capacity_rate": None,
            "capacity": None,
        }
    ]

    for rate in CAPACITY_RATES:
        capacity = int(np.floor(total_transactions * rate))

        policies.append({
            "policy": "capacity_constrained",
            "capacity_rate": rate,
            "capacity": capacity,
        })

    results = []

    total_runs = (
        len(strategies)
        * len(COST_SCENARIOS)
        * len(policies)
    )

    print(f"Evaluation transactions: {total_transactions}")
    print(f"Prediction strategies: {len(strategies)}")
    print(f"Cost scenarios: {len(COST_SCENARIOS)}")
    print(f"Policies: {len(policies)}")
    print(f"Total comparisons: {total_runs}")

    for strategy_name, probabilities in strategies.items():

        for scenario_name, costs in COST_SCENARIOS.items():

            for policy in policies:

                metrics = evaluate_cost_policy(
                    y_true=y_true,
                    probabilities=probabilities,
                    costs=costs,
                    review_capacity=policy["capacity"],
                )

                results.append({
                    "strategy": strategy_name,
                    "cost_scenario": scenario_name,
                    "policy": policy["policy"],
                    "review_capacity_rate": policy["capacity_rate"],
                    "review_capacity": policy["capacity"],
                    **metrics,
                })

    results_df = pd.DataFrame(results)

    if len(results_df) != total_runs:
        raise RuntimeError(
            f"Expected {total_runs} results, got {len(results_df)}."
        )

    return results_df


def main() -> None:
    print("Starting cost-sensitivity experiment...")

    results = run_experiment()

    output_path = RESULT_DIR / "cost_sensitivity_results.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    results.to_csv(output_path, index=False)

    print("\nCOST-SENSITIVITY EXPERIMENT COMPLETE")
    print("=" * 70)
    print(f"Comparisons completed: {len(results)}")
    print(f"Unique strategies: {results['strategy'].nunique()}")
    print(f"Cost scenarios: {results['cost_scenario'].nunique()}")
    policy_configurations = results[
        ["policy", "review_capacity_rate"]
    ].drop_duplicates()

    print(f"Policy configurations: {len(policy_configurations)}")
    print(f"\nResults saved to: {output_path}")

    print("\nSAMPLE RESULTS")
    print(
        results[
            [
                "strategy",
                "cost_scenario",
                "policy",
                "review_capacity_rate",
                "total_realized_cost",
                "realized_cost_per_transaction",
            ]
        ].head(10).to_string(index=False)
    )


if __name__ == "__main__":
    main()
