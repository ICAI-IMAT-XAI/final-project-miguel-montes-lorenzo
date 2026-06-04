"""Strategy registry."""

from __future__ import annotations

from src.strategies.base import BaseStrategy
from src.strategies.pointwise.full_rebalancing import FullRebalancingStrategy
from src.strategies.pointwise.partial_rebalancing import PartialRebalancingStrategy
from src.strategies.probabilistic.black_litterman import BlackLittermanStrategy

STRATEGY_REGISTRY: dict[str, type[BaseStrategy]] = {
    "full_rebalancing": FullRebalancingStrategy,
    "partial_rebalancing": PartialRebalancingStrategy,
    "black_litterman": BlackLittermanStrategy,
}

POINTWISE_STRATEGIES: set[str] = {
    "full_rebalancing",
    "partial_rebalancing",
}

PROBABILISTIC_STRATEGIES: set[str] = {
    "black_litterman",
}


def get_strategy(name: str) -> type[BaseStrategy]:
    strategy_cls = STRATEGY_REGISTRY.get(name)
    if strategy_cls is None:
        raise ValueError(
            f"Unknown strategy {name!r}. Available: {sorted(STRATEGY_REGISTRY)}."
        )
    return strategy_cls
