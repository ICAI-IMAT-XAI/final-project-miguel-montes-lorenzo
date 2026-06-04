"""Metric aggregation utilities."""

from __future__ import annotations

import numpy as np

from src.metrics.performance.drawdown import max_drawdown
from src.metrics.performance.returns import annualized_return, cumulative_return
from src.metrics.performance.sharpe import sharpe_ratio
from src.metrics.performance.sortino import sortino_ratio
from src.metrics.stability.herfindahl import mean_herfindahl
from src.metrics.stability.turnover import mean_turnover

ALL_BACKTEST_METRICS: tuple[str, ...] = (
    "cumulative_return",
    "annualized_return",
    "annualized_volatility",
    "sharpe_ratio",
    "sortino_ratio",
    "max_drawdown",
    "mean_turnover",
    "mean_herfindahl",
)


def compute_backtest_metrics(
    portfolio_values: np.ndarray,
    log_returns: np.ndarray,
    weights: np.ndarray,
    trading_days_per_year: int,
) -> dict[str, float]:
    """Compute performance and stability metrics for a backtest.

    Args:
        portfolio_values: Portfolio value path.
        log_returns: Daily portfolio log returns.
        weights: Daily allocation weights.
        trading_days_per_year: Number of trading days per year.

    Returns:
        Dictionary of scalar metrics.
    """
    return {
        "cumulative_return": cumulative_return(portfolio_values=portfolio_values),
        "annualized_return": annualized_return(
            log_returns=log_returns,
            trading_days_per_year=trading_days_per_year,
        ),
        "annualized_volatility": float(
            np.std(a=log_returns, ddof=1) * np.sqrt(trading_days_per_year)
        )
        if log_returns.size > 1
        else 0.0,
        "sharpe_ratio": sharpe_ratio(
            log_returns=log_returns,
            trading_days_per_year=trading_days_per_year,
        ),
        "sortino_ratio": sortino_ratio(
            log_returns=log_returns,
            trading_days_per_year=trading_days_per_year,
        ),
        "max_drawdown": max_drawdown(portfolio_values=portfolio_values),
        "mean_turnover": mean_turnover(weights=weights),
        "mean_herfindahl": mean_herfindahl(weights=weights),
    }


def select_backtest_metrics(
    metrics: dict[str, float],
    requested_metrics: list[str] | tuple[str, ...] | None,
) -> dict[str, float]:
    """Filter metric output to the requested subset."""
    if requested_metrics is None:
        requested = list(ALL_BACKTEST_METRICS)
    else:
        requested = [str(name) for name in requested_metrics]
    unknown = sorted(set(requested).difference(metrics))
    if unknown:
        raise KeyError(
            f"Unknown metric(s) {unknown}. Available metrics: "
            f"{list(ALL_BACKTEST_METRICS)}."
        )
    return {name: metrics[name] for name in requested}
