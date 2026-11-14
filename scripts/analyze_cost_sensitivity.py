
"""Analyze cost-sensitivity and review-capacity experiment results."""

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESULT_DIR = ROOT / "experiments" / "results"

INPUT_PATH = RESULT_DIR / "cost_sensitivity_results.csv"


def load_results() -> pd.DataFrame:
    """Load and validate cost-sensitivity results."""

    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"Results not found: {INPUT_PATH}")

    df = pd.read_csv(INPUT_PATH)

    required = {
        "strategy",
        "cost_scenario",
        "policy",
        "review_capacity_rate",
        "total_expected_cost",
        "total_realized_cost",
        "realized_cost_per_transaction",
        "review_count",
    }

    missing = required - set(df.columns)

    if missing:
        raise ValueError(f"Missing columns: {missing}")

    if len(df) != 686:
        raise ValueError(f"Expected 686 rows, found {len(df)}.")

    if df.duplicated(
        ["strategy", "cost_scenario", "policy", "review_capacity_rate"]
    ).any():
        raise ValueError("Duplicate experiment comparisons detected.")

    return df


def analyze_winners(df: pd.DataFrame) -> pd.DataFrame:
    """Identify expected-cost and realized-cost winners."""

    grouping = [
        "cost_scenario",
        "policy",
        "review_capacity_rate",
    ]

    expected_winners = (
        df.loc[
            df.groupby(grouping, dropna=False)[
                "total_expected_cost"
            ].idxmin()
        ]
        .copy()
        .rename(columns={
            "strategy": "expected_cost_winner",
            "total_expected_cost": "minimum_expected_cost",
        })
    )

    realized_winners = (
        df.loc[
            df.groupby(grouping, dropna=False)[
                "total_realized_cost"
            ].idxmin()
        ]
        .copy()
        .rename(columns={
            "strategy": "realized_cost_winner",
            "total_realized_cost": "minimum_realized_cost",
        })
    )

    winner_columns = grouping

    expected_winners = expected_winners[
        winner_columns
        + ["expected_cost_winner", "minimum_expected_cost"]
    ]

    realized_winners = realized_winners[
        winner_columns
        + ["realized_cost_winner", "minimum_realized_cost"]
    ]

    winners = expected_winners.merge(
        realized_winners,
        on=winner_columns,
        validate="one_to_one",
    )

    winners["winners_agree"] = (
        winners["expected_cost_winner"]
        == winners["realized_cost_winner"]
    )

    return winners.sort_values(
        ["cost_scenario", "policy", "review_capacity_rate"],
        na_position="first",
    )


def analyze_capacity_impact(df: pd.DataFrame) -> pd.DataFrame:
    """Measure the cost penalty of capacity limits."""

    keys = ["strategy", "cost_scenario"]

    unconstrained = df.loc[
        df["policy"] == "unconstrained",
        keys + [
            "total_expected_cost",
            "total_realized_cost",
        ],
    ].rename(columns={
        "total_expected_cost": "unconstrained_expected_cost",
        "total_realized_cost": "unconstrained_realized_cost",
    })

    constrained = df.loc[
        df["policy"] == "capacity_constrained"
    ].copy()

    comparison = constrained.merge(
        unconstrained,
        on=keys,
        validate="many_to_one",
    )

    comparison["realized_cost_increase"] = (
        comparison["total_realized_cost"]
        - comparison["unconstrained_realized_cost"]
    )

    comparison["realized_cost_increase_pct"] = (
        comparison["realized_cost_increase"]
        / comparison["unconstrained_realized_cost"].replace(0, pd.NA)
        * 100
    )

    comparison["expected_cost_increase"] = (
        comparison["total_expected_cost"]
        - comparison["unconstrained_expected_cost"]
    )

    return comparison.sort_values(
        ["cost_scenario", "strategy", "review_capacity_rate"]
    )


def analyze_strategy_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Summarize realized performance across operating conditions."""

    summary = (
        df.groupby("strategy", as_index=False)
        .agg(
            mean_realized_cost=(
                "total_realized_cost", "mean"
            ),
            median_realized_cost=(
                "total_realized_cost", "median"
            ),
            mean_expected_cost=(
                "total_expected_cost", "mean"
            ),
            mean_cost_per_transaction=(
                "realized_cost_per_transaction", "mean"
            ),
            mean_fraud_approval_rate=(
                "fraud_approval_rate", "mean"
            ),
            mean_legitimate_block_rate=(
                "legitimate_block_rate", "mean"
            ),
        )
        .sort_values("mean_realized_cost")
        .reset_index(drop=True)
    )

    return summary


def main() -> None:
    print("Loading cost-sensitivity results...")

    df = load_results()

    winners = analyze_winners(df)
    capacity_impact = analyze_capacity_impact(df)
    strategy_summary = analyze_strategy_summary(df)

    outputs = {
        "cost_sensitivity_winners.csv": winners,
        "cost_sensitivity_capacity_impact.csv": capacity_impact,
        "cost_sensitivity_strategy_summary.csv": strategy_summary,
    }

    for filename, result in outputs.items():
        path = RESULT_DIR / filename
        result.to_csv(path, index=False)
        print(f"Saved: {path.name} ({len(result)} rows)")

    print("\nEXPERIMENT ANALYSIS SUMMARY")
    print("=" * 70)

    print(f"Experiment comparisons: {len(df)}")
    print(f"Unique strategies: {df['strategy'].nunique()}")
    print(f"Cost scenarios: {df['cost_scenario'].nunique()}")

    configurations = df[
        ["policy", "review_capacity_rate"]
    ].drop_duplicates()

    print(f"Policy configurations: {len(configurations)}")

    print("\nREALIZED-COST WINNERS")
    print(
        winners[
            [
                "cost_scenario",
                "policy",
                "review_capacity_rate",
                "realized_cost_winner",
                "minimum_realized_cost",
            ]
        ].to_string(index=False)
    )

    agreement = winners["winners_agree"].mean() * 100

    print(
        "\nExpected-cost and realized-cost winner agreement: "
        f"{agreement:.2f}%"
    )

    print("\nAVERAGE STRATEGY PERFORMANCE")
    print(strategy_summary.to_string(index=False))


if __name__ == "__main__":
    main()
