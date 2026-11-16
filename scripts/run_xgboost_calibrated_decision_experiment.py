
"""Compare calibrated and uncalibrated XGBoost decision policies."""

from pathlib import Path

import numpy as np
import pandas as pd

from driftsentinel.policies.cost_evaluation import evaluate_cost_policy
from driftsentinel.policies.cost_scenarios import COST_SCENARIOS

ROOT = Path(__file__).resolve().parents[1]

PREDICTIONS_PATH = (
    ROOT
    / "experiments/results/"
    "xgboost_calibrated_evaluation_predictions.parquet"
)

RESULTS_PATH = (
    ROOT
    / "experiments/results/"
    "xgboost_calibrated_decision_results.csv"
)

WINNERS_PATH = (
    ROOT
    / "experiments/results/"
    "xgboost_calibrated_decision_winners.csv"
)

CAPACITY_RATES = [
    0.0,
    0.0025,
    0.005,
    0.01,
    0.02,
    0.03,
    0.05,
]

PROBABILITY_COLUMNS = {
    "uncalibrated": "probability_uncalibrated",
    "sigmoid": "probability_sigmoid",
}


def load_predictions() -> pd.DataFrame:
    """Load and validate the chronological evaluation predictions."""

    if not PREDICTIONS_PATH.exists():
        raise FileNotFoundError(
            f"Predictions file not found: {PREDICTIONS_PATH}"
        )

    df = pd.read_parquet(PREDICTIONS_PATH)

    required = {
        "TransactionID",
        "TransactionDT",
        "isFraud",
        *PROBABILITY_COLUMNS.values(),
    }

    missing = required - set(df.columns)

    if missing:
        raise ValueError(f"Missing prediction columns: {missing}")

    if df.empty:
        raise ValueError("Prediction file is empty.")

    if df["TransactionID"].duplicated().any():
        raise ValueError("Duplicate transaction IDs found.")

    if not df["isFraud"].isin([0, 1]).all():
        raise ValueError("Labels must contain only 0 and 1.")

    for column in PROBABILITY_COLUMNS.values():
        probabilities = df[column].to_numpy(dtype=float)

        if not np.isfinite(probabilities).all():
            raise ValueError(
                f"{column} contains non-finite values."
            )

        if ((probabilities < 0) | (probabilities > 1)).any():
            raise ValueError(
                f"{column} contains values outside [0, 1]."
            )

    return df.sort_values(
        "TransactionDT",
        kind="stable",
    ).reset_index(drop=True)


def build_policies(total_transactions: int) -> list[dict]:
    """Build unconstrained and capacity-constrained policies."""

    policies = [
        {
            "policy_name": "unconstrained",
            "capacity_rate": None,
            "capacity": None,
        }
    ]

    for rate in CAPACITY_RATES:
        capacity = int(np.floor(total_transactions * rate))

        policies.append({
            "policy_name": "capacity_constrained",
            "capacity_rate": rate,
            "capacity": capacity,
        })

    return policies


def run_experiment(df: pd.DataFrame) -> pd.DataFrame:
    """Evaluate both probability strategies across all policies."""

    y_true = df["isFraud"].to_numpy()
    policies = build_policies(len(df))

    total_runs = (
        len(PROBABILITY_COLUMNS)
        * len(COST_SCENARIOS)
        * len(policies)
    )

    print(f"Evaluation transactions: {len(df):,}")
    print(f"Fraud prevalence: {y_true.mean():.4%}")
    print(f"Probability strategies: {len(PROBABILITY_COLUMNS)}")
    print(f"Cost scenarios: {len(COST_SCENARIOS)}")
    print(f"Review policies: {len(policies)}")
    print(f"Total evaluations: {total_runs}")

    results = []

    for strategy_name, probability_column in (
        PROBABILITY_COLUMNS.items()
    ):
        probabilities = df[probability_column].to_numpy(
            dtype=float
        )

        print(f"\nEvaluating strategy: {strategy_name}")

        for scenario_name, costs in COST_SCENARIOS.items():

            for policy in policies:

                metrics = evaluate_cost_policy(
                    y_true=y_true,
                    probabilities=probabilities,
                    costs=costs,
                    review_capacity=policy["capacity"],
                )

                results.append({
                    "model": "XGBoost",
                    "probability_strategy": strategy_name,
                    "cost_scenario": scenario_name,
                    "policy_name": policy["policy_name"],
                    "review_capacity_rate": policy["capacity_rate"],
                    **metrics,
                })

    results_df = pd.DataFrame(results)

    if len(results_df) != total_runs:
        raise RuntimeError(
            f"Expected {total_runs} results, "
            f"got {len(results_df)}."
        )

    return results_df


def find_cost_winners(results: pd.DataFrame) -> pd.DataFrame:
    """Find expected and realized cost winners per strategy and scenario."""

    winners = []

    grouped = results.groupby(
        ["probability_strategy", "cost_scenario"]
    )

    for (strategy, scenario), group in grouped:

        expected_winner = group.loc[
            group["total_expected_cost"].idxmin()
        ]

        realized_winner = group.loc[
            group["total_realized_cost"].idxmin()
        ]

        expected_capacity = expected_winner["review_capacity"]
        realized_capacity = realized_winner["review_capacity"]

        capacities_match = (
            (
                pd.isna(expected_capacity)
                and pd.isna(realized_capacity)
            )
            or expected_capacity == realized_capacity
        )

        winners.append({
            "probability_strategy": strategy,
            "cost_scenario": scenario,
            "expected_cost_winner": expected_winner["policy_name"],
            "expected_cost_capacity": expected_capacity,
            "minimum_expected_cost": (
                expected_winner["total_expected_cost"]
            ),
            "realized_cost_winner": realized_winner["policy_name"],
            "realized_cost_capacity": realized_capacity,
            "minimum_realized_cost": (
                realized_winner["total_realized_cost"]
            ),
            "winners_agree": (
                expected_winner["policy_name"]
                == realized_winner["policy_name"]
                and capacities_match
            ),
        })

    return pd.DataFrame(winners)


def main() -> None:
    print("Loading calibrated evaluation predictions...")

    df = load_predictions()

    print("\nStarting calibrated decision-policy experiment...")

    results = run_experiment(df)
    winners = find_cost_winners(results)

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)

    results.to_csv(RESULTS_PATH, index=False)
    winners.to_csv(WINNERS_PATH, index=False)

    print("\nEXPERIMENT COMPLETE")
    print("=" * 70)
    print(f"Evaluations completed: {len(results)}")

    print("\nCOST-SCENARIO WINNERS")
    print(winners.to_string(index=False))

    print("\nRESULTS SAVED")
    print(RESULTS_PATH)
    print(WINNERS_PATH)


if __name__ == "__main__":
    main()
