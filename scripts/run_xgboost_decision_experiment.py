
"""Evaluate XGBoost predictions across cost scenarios and review capacities."""

from pathlib import Path

import numpy as np
import pandas as pd

from driftsentinel.policies.cost_evaluation import evaluate_cost_policy
from driftsentinel.policies.cost_scenarios import COST_SCENARIOS

ROOT = Path(__file__).resolve().parents[1]

PREDICTIONS_PATH = (
    ROOT / "experiments/results/xgboost_full_predictions.parquet"
)

RESULTS_PATH = (
    ROOT / "experiments/results/xgboost_cost_sensitivity_results.csv"
)

WINNERS_PATH = (
    ROOT / "experiments/results/xgboost_cost_sensitivity_winners.csv"
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


def load_predictions() -> pd.DataFrame:
    """Load and validate saved XGBoost validation predictions."""

    if not PREDICTIONS_PATH.exists():
        raise FileNotFoundError(
            f"Predictions file not found: {PREDICTIONS_PATH}"
        )

    df = pd.read_parquet(PREDICTIONS_PATH)

    required = {
        "TransactionID",
        "TransactionDT",
        "isFraud",
        "probability",
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

    probabilities = df["probability"].to_numpy(dtype=float)

    if not np.isfinite(probabilities).all():
        raise ValueError("Probabilities contain non-finite values.")

    if ((probabilities < 0) | (probabilities > 1)).any():
        raise ValueError("Probabilities must be between 0 and 1.")

    return df


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
    """Evaluate all cost scenarios and review-capacity policies."""

    y_true = df["isFraud"].to_numpy()
    probabilities = df["probability"].to_numpy(dtype=float)

    policies = build_policies(len(df))

    total_runs = len(COST_SCENARIOS) * len(policies)

    print(f"Validation transactions: {len(df):,}")
    print(f"Fraud prevalence: {y_true.mean():.4%}")
    print(f"Cost scenarios: {len(COST_SCENARIOS)}")
    print(f"Review policies: {len(policies)}")
    print(f"Total evaluations: {total_runs}")

    results = []

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
                "cost_scenario": scenario_name,
                "policy_name": policy["policy_name"],
                "review_capacity_rate": policy["capacity_rate"],
                **metrics,
            })

    results_df = pd.DataFrame(results)

    if len(results_df) != total_runs:
        raise RuntimeError(
            f"Expected {total_runs} results, got {len(results_df)}."
        )

    return results_df



def find_cost_winners(results: pd.DataFrame) -> pd.DataFrame:
    """Find expected-cost and realized-cost winners per scenario."""

    winners = []

    for scenario_name, group in results.groupby("cost_scenario"):
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
            "cost_scenario": scenario_name,
            "expected_cost_winner": expected_winner["policy_name"],
            "expected_cost_capacity": expected_capacity,
            "minimum_expected_cost": expected_winner["total_expected_cost"],
            "realized_cost_winner": realized_winner["policy_name"],
            "realized_cost_capacity": realized_capacity,
            "minimum_realized_cost": realized_winner["total_realized_cost"],
            "winners_agree": (
                expected_winner["policy_name"]
                == realized_winner["policy_name"]
                and capacities_match
            ),
        })

    return pd.DataFrame(winners)



def main() -> None:
    print("Loading XGBoost validation predictions...")

    df = load_predictions()

    print("\nStarting XGBoost cost-sensitive experiment...")

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
