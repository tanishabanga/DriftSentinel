
"""Analyze action-conditional calibration and cost gaps."""

from pathlib import Path

import pandas as pd

from driftsentinel.policies.cost_evaluation import calculate_realized_cost
from driftsentinel.policies.costs import DecisionCosts
from driftsentinel.policies.decision import make_decisions

ROOT = Path(__file__).resolve().parents[1]
RESULT_DIR = ROOT / "experiments" / "results"


def analyze_action_calibration(
    y_true,
    probabilities,
    strategy_name: str,
    costs: DecisionCosts,
) -> tuple[list[dict], list[dict]]:
    """Measure calibration and cost gaps for each selected action."""

    decisions = make_decisions(probabilities, costs)
    action_cost_differences = {
        "approve": costs.approve_fraud - costs.approve_legitimate,
        "review": costs.review_fraud - costs.review_legitimate,
        "block": costs.block_fraud - costs.block_legitimate,
    }

    labels = pd.Series(y_true).reset_index(drop=True)
    actual_costs = calculate_realized_cost(
        labels.to_numpy(),
        decisions["action"].to_numpy(),
        costs,
    )

    decisions["actual_cost"] = actual_costs
    decisions["actual_fraud"] = labels.to_numpy()

    calibration_rows = []
    cost_rows = []

    for action, cost_difference in action_cost_differences.items():
        group = decisions[decisions["action"] == action]

        count = len(group)

        if count == 0:
            mean_probability = float("nan")
            observed_fraud_rate = float("nan")
            calibration_gap = float("nan")
        else:
            mean_probability = float(group["probability"].mean())
            observed_fraud_rate = float(group["actual_fraud"].mean())
            calibration_gap = observed_fraud_rate - mean_probability

        expected_cost = float(group["expected_cost"].sum())
        realized_cost = float(group["actual_cost"].sum())

        calibration_rows.append({
            "strategy": strategy_name,
            "action": action,
            "transactions": count,
            "mean_predicted_risk": mean_probability,
            "observed_fraud_rate": observed_fraud_rate,
            "calibration_gap": calibration_gap,
            "absolute_calibration_gap": abs(calibration_gap),
        })

        cost_rows.append({
            "strategy": strategy_name,
            "action": action,
            "transactions": count,
            "expected_cost": expected_cost,
            "realized_cost": realized_cost,
            "cost_gap": realized_cost - expected_cost,
            "independent_gap_contribution": (
                float(
                    (
                        group["actual_fraud"]
                        - group["probability"]
                    ).sum()
                    * cost_difference
                )
                if count else 0.0
            ),
        })

    return calibration_rows, cost_rows


def main() -> None:
    """Run action-conditional diagnostics for all strategies."""
    # Reuse the existing canonical 14-strategy construction.
    from run_cost_sensitivity import build_strategies

    y_true, strategies = build_strategies()
    costs = DecisionCosts()

    calibration_results = []
    cost_results = []

    for strategy_name, probabilities in strategies.items():
        calibration_rows, cost_rows = analyze_action_calibration(
            y_true=y_true,
            probabilities=probabilities,
            strategy_name=strategy_name,
            costs=costs,
        )

        calibration_results.extend(calibration_rows)
        cost_results.extend(cost_rows)

    calibration_df = pd.DataFrame(calibration_results)
    cost_df = pd.DataFrame(cost_results)

    calibration_path = (
        RESULT_DIR / "action_calibration_diagnostics.csv"
    )
    cost_path = RESULT_DIR / "action_cost_gap_diagnostics.csv"

    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    calibration_df.to_csv(calibration_path, index=False)
    cost_df.to_csv(cost_path, index=False)

    print("\nACTION-CONDITIONAL CALIBRATION")
    print("=" * 80)
    print(calibration_df.to_string(index=False))

    print("\nACTION-LEVEL COST GAPS")
    print("=" * 80)
    print(cost_df.to_string(index=False))

    print("\nDIAGNOSTIC COMPLETE")
    print(f"Strategies: {calibration_df['strategy'].nunique()}")
    print(f"Calibration rows: {len(calibration_df)}")
    print(f"Cost rows: {len(cost_df)}")
    print(f"Calibration output: {calibration_path}")
    print(f"Cost output: {cost_path}")


if __name__ == "__main__":
    main()
