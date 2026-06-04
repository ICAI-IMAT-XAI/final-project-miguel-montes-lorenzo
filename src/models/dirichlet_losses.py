"""Loss functions for Dirichlet portfolio models."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F


def alpha_to_weights(alpha: torch.Tensor) -> torch.Tensor:
    """Convert Dirichlet concentrations into mean portfolio weights."""
    return alpha / alpha.sum(dim=-1, keepdim=True).clamp_min(1.0e-8)


def compute_alpha_true(
    label_weights: torch.Tensor,
    lookback: int,
) -> torch.Tensor:
    """Build a positive Dirichlet target from weight labels."""
    num_assets = label_weights.shape[-1]
    alpha_0 = max(float(lookback) / float(num_assets), 1.0)
    return (label_weights.clamp_min(0.0) + 1.0e-6) * alpha_0


def loss_mse(alpha_pred: torch.Tensor, label: torch.Tensor, **_: Any) -> torch.Tensor:
    return F.mse_loss(alpha_to_weights(alpha_pred), label)


def loss_cosine(
    alpha_pred: torch.Tensor,
    label: torch.Tensor,
    **_: Any,
) -> torch.Tensor:
    cosine = F.cosine_similarity(alpha_to_weights(alpha_pred), label, dim=-1)
    return (1.0 - cosine).mean()


def loss_kl(
    alpha_pred: torch.Tensor,
    label: torch.Tensor,
    lookback: int = 60,
    **_: Any,
) -> torch.Tensor:
    alpha_true = compute_alpha_true(label, lookback).to(alpha_pred.device)
    a0_pred = alpha_pred.sum(dim=-1, keepdim=True)
    a0_true = alpha_true.sum(dim=-1, keepdim=True)
    kl = (
        torch.lgamma(a0_pred)
        - torch.lgamma(a0_true)
        - (torch.lgamma(alpha_pred) - torch.lgamma(alpha_true)).sum(dim=-1, keepdim=True)
        + (
            (alpha_pred - alpha_true)
            * (torch.digamma(alpha_pred) - torch.digamma(a0_pred))
        ).sum(dim=-1, keepdim=True)
    )
    return kl.mean()


def loss_jsd(
    alpha_pred: torch.Tensor,
    label: torch.Tensor,
    lookback: int = 60,
    n_samples: int = 50,
    **_: Any,
) -> torch.Tensor:
    alpha_true = compute_alpha_true(label, lookback).to(alpha_pred.device)
    dist_pred = torch.distributions.Dirichlet(alpha_pred)
    dist_true = torch.distributions.Dirichlet(alpha_true)

    x_pred = dist_pred.rsample((n_samples,))
    log_pred_x_pred = dist_pred.log_prob(x_pred)
    log_true_x_pred = dist_true.log_prob(x_pred)
    log_mix_x_pred = torch.logaddexp(log_pred_x_pred, log_true_x_pred) - torch.log(
        torch.tensor(2.0, device=alpha_pred.device)
    )
    kl_pred_mix = (log_pred_x_pred - log_mix_x_pred).mean(0)

    x_true = dist_true.rsample((n_samples,))
    log_pred_x_true = dist_pred.log_prob(x_true)
    log_true_x_true = dist_true.log_prob(x_true)
    log_mix_x_true = torch.logaddexp(log_pred_x_true, log_true_x_true) - torch.log(
        torch.tensor(2.0, device=alpha_pred.device)
    )
    kl_true_mix = (log_true_x_true - log_mix_x_true).mean(0)
    return (0.5 * kl_pred_mix + 0.5 * kl_true_mix).clamp(min=0.0).mean()


def loss_hellinger(
    alpha_pred: torch.Tensor,
    label: torch.Tensor,
    lookback: int = 60,
    n_samples: int = 50,
    **_: Any,
) -> torch.Tensor:
    alpha_true = compute_alpha_true(label, lookback).to(alpha_pred.device)
    dist_pred = torch.distributions.Dirichlet(alpha_pred)
    dist_true = torch.distributions.Dirichlet(alpha_true)
    x_true = dist_true.rsample((n_samples,))
    log_pred = dist_pred.log_prob(x_true)
    log_true = dist_true.log_prob(x_true)
    bhattacharyya = torch.exp(0.5 * (log_pred - log_true)).mean(0)
    return (1.0 - bhattacharyya).clamp(0.0, 1.0).mean()


def loss_wasserstein(
    alpha_pred: torch.Tensor,
    label: torch.Tensor,
    lookback: int = 60,
    n_samples: int = 100,
    n_projections: int = 32,
    **_: Any,
) -> torch.Tensor:
    alpha_true = compute_alpha_true(label, lookback).to(alpha_pred.device)
    dist_pred = torch.distributions.Dirichlet(alpha_pred)
    dist_true = torch.distributions.Dirichlet(alpha_true)
    x_pred = dist_pred.rsample((n_samples,))
    x_true = dist_true.rsample((n_samples,))
    directions = torch.randn(
        n_projections,
        alpha_pred.shape[-1],
        device=alpha_pred.device,
    )
    directions = F.normalize(directions, dim=-1)
    proj_pred = x_pred @ directions.T
    proj_true = x_true @ directions.T
    proj_pred_sorted = proj_pred.sort(dim=0).values
    proj_true_sorted = proj_true.sort(dim=0).values
    return (proj_pred_sorted - proj_true_sorted).abs().mean(0).mean()


def portfolio_variance_penalty(
    alpha_pred: torch.Tensor,
    sigma: torch.Tensor,
) -> torch.Tensor:
    weights = alpha_to_weights(alpha_pred)
    if sigma.dim() == 2:
        sigma = sigma.unsqueeze(0).expand(weights.shape[0], -1, -1)
    portfolio_variance = torch.bmm(
        weights.unsqueeze(1),
        torch.bmm(sigma, weights.unsqueeze(2)),
    ).squeeze(-1).squeeze(-1)
    return portfolio_variance.mean()


def bear_entropy_penalty(
    alpha_pred: torch.Tensor,
    market_ret: torch.Tensor,
) -> torch.Tensor:
    alpha_pred = alpha_pred.clamp(min=1.0e-6)
    market = torch.as_tensor(
        market_ret,
        dtype=torch.float32,
        device=alpha_pred.device,
    ).view(-1)
    if market.numel() == 1:
        market = market.expand(alpha_pred.shape[0])
    alpha_0 = alpha_pred.sum(dim=-1)
    num_assets = alpha_pred.shape[-1]
    log_beta = torch.lgamma(alpha_pred).sum(dim=-1) - torch.lgamma(alpha_0)
    entropy = (
        log_beta
        - (alpha_0 - num_assets) * torch.digamma(alpha_0)
        + ((alpha_pred - 1.0) * torch.digamma(alpha_pred)).sum(dim=-1)
    )
    entropy = torch.nan_to_num(entropy, nan=0.0, posinf=0.0, neginf=0.0)
    return ((market < 0.0).float() * entropy).mean()


LOSS_FUNCTIONS: dict[str, Any] = {
    "mse": loss_mse,
    "cosine": loss_cosine,
    "kl": loss_kl,
    "jsd": loss_jsd,
    "hellinger": loss_hellinger,
    "wasserstein": loss_wasserstein,
}


def build_loss_fn(config: dict[str, Any]):
    """Build a composite Dirichlet loss from config."""
    loss_name = str(config.get("loss", "mse")).lower()
    if loss_name not in LOSS_FUNCTIONS:
        raise ValueError(
            f"Unknown loss {loss_name!r}. Available: {sorted(LOSS_FUNCTIONS)}."
        )
    reconstruction_fn = LOSS_FUNCTIONS[loss_name]
    regularizations: dict[str, Any] = dict(config.get("loss_regularizations") or {})
    covariance_weight = float(regularizations.get("covariance") or 0.0)
    bear_entropy_weight = float(regularizations.get("bear_entropy") or 0.0)

    def composite_loss(
        alpha_pred: torch.Tensor,
        label: torch.Tensor,
        lookback: int = 60,
        sigma: torch.Tensor | None = None,
        market_ret: torch.Tensor | None = None,
    ) -> torch.Tensor:
        loss = reconstruction_fn(
            alpha_pred=alpha_pred,
            label=label,
            lookback=lookback,
        )
        if covariance_weight and sigma is not None:
            loss = loss + covariance_weight * portfolio_variance_penalty(
                alpha_pred=alpha_pred,
                sigma=sigma,
            )
        if bear_entropy_weight and market_ret is not None:
            loss = loss + bear_entropy_weight * bear_entropy_penalty(
                alpha_pred=alpha_pred,
                market_ret=market_ret,
            )
        return loss

    return composite_loss
