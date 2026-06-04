"""Black-Litterman rebalancing using model outputs as views."""

from __future__ import annotations

import numpy as np

from src.strategies.base import BaseStrategy


def mean_variance_weights(
    mu: np.ndarray,
    covariance: np.ndarray,
    risk_aversion: float,
) -> np.ndarray:
    """Solve a long-only mean-variance allocation problem."""
    scaled_covariance = float(risk_aversion) * covariance
    try:
        raw_weights = np.linalg.solve(scaled_covariance, mu)
    except np.linalg.LinAlgError:
        raw_weights = np.linalg.pinv(scaled_covariance) @ mu
    raw_weights = np.maximum(raw_weights, 0.0)
    total = float(raw_weights.sum())
    if total <= 1.0e-8:
        return np.full_like(raw_weights, fill_value=1.0 / len(raw_weights))
    return raw_weights / total


class BlackLittermanStrategy(BaseStrategy):
    """Blend equilibrium prior and model views into a posterior allocation."""

    requires_uncertainty = True

    def rebalance(
        self,
        current_weights: np.ndarray,
        target_weights: np.ndarray,
        uncertainties: np.ndarray | None,
        returns: np.ndarray,
        t: int,
        cfg: dict,
    ) -> np.ndarray:
        del current_weights
        num_assets = len(target_weights)
        tau = float(cfg.get("tau", 0.05))
        risk_aversion = float(cfg.get("risk_aversion", 1.0))
        history = returns[: t + 1]
        if history.shape[0] < 2:
            covariance = np.eye(num_assets) * 0.01
        else:
            covariance = np.cov(history.T)
        covariance = np.atleast_2d(covariance) + 1.0e-6 * np.eye(num_assets)
        equilibrium_weights = np.ones(num_assets) / num_assets
        equilibrium_returns = risk_aversion * covariance @ equilibrium_weights

        view_matrix = np.eye(num_assets)
        view_returns = target_weights
        omega_diagonal = (
            np.maximum(uncertainties, 1.0e-6)
            if uncertainties is not None
            else np.full(num_assets, 0.01)
        )
        omega = np.diag(omega_diagonal)
        tau_sigma = tau * covariance
        inv_tau_sigma = np.linalg.pinv(tau_sigma)
        inv_omega = np.linalg.pinv(omega)
        posterior_covariance = np.linalg.pinv(
            inv_tau_sigma + view_matrix.T @ inv_omega @ view_matrix
        )
        posterior_mean = posterior_covariance @ (
            inv_tau_sigma @ equilibrium_returns
            + view_matrix.T @ inv_omega @ view_returns
        )
        return mean_variance_weights(
            mu=posterior_mean,
            covariance=covariance + posterior_covariance,
            risk_aversion=risk_aversion,
        )
