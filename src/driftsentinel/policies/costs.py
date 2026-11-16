
"""Configurable costs for fraud decision policies."""

from dataclasses import dataclass
from math import isfinite


@dataclass(frozen=True)
class DecisionCosts:
    """Cost assumptions for each actual class and decision."""

    approve_legitimate: float = 0.0
    approve_fraud: float = 100.0

    review_legitimate: float = 2.0
    review_fraud: float = 5.0

    block_legitimate: float = 10.0
    block_fraud: float = 0.0

    def __post_init__(self):
        for name, value in self.__dict__.items():
            if not isfinite(value) or value < 0:
                raise ValueError(
                    f"{name} must be a finite, non-negative number."
                )
