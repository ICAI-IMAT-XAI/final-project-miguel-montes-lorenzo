"""Backtesting entry point for running multiple rebalancing strategies."""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.data.dataset import ForexPortfolioDataset, move_batch_to_device
from src.eval import resolve_checkpoint
from src.metrics import (
    ALL_BACKTEST_METRICS,
    compute_backtest_metrics,
    select_backtest_metrics,
)
from src.models import load_model_from_checkpoint
from src.models.base import PortfolioModule, PortfolioPrediction
from src.strategies.registry import (
    PROBABILISTIC_STRATEGIES,
    get_strategy,
)
from src.utils import ensure_dir, load_yaml, write_json


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for strategy backtests."""
    parser = argparse.ArgumentParser(description="Run strategy backtests.")
    parser.add_argument(
        "--checkpoint",
        required=True,
        nargs="+",
        help=(
            "One or more checkpoints to backtest, or 'latest' for the newest "
            "checkpoints/*.pt."
        ),
    )
    parser.add_argument(
        "--config",
        default="configs/backtest.yaml",
        help="Backtest YAML config.",
    )
    return parser.parse_args()


def resolve_enabled_entries(
    config_value: Any,
    valid_names: list[str],
    default_names: list[str],
    field_name: str,
) -> list[str]:
    """Resolve a config field that may be a list or a bool mapping."""
    if config_value is None:
        return list(default_names)
    if isinstance(config_value, dict):
        enabled = [name for name, is_enabled in config_value.items() if bool(is_enabled)]
    elif isinstance(config_value, str):
        enabled = [config_value]
    else:
        enabled = [str(name) for name in config_value]
    unknown = sorted(set(enabled).difference(valid_names))
    if unknown:
        raise KeyError(
            f"Unknown {field_name} {unknown}. Available: {sorted(valid_names)}."
        )
    return enabled


def collect_strategy_predictions(
    model: PortfolioModule,
    dataset: ForexPortfolioDataset,
    device: torch.device,
    mc_samples: int,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray, list[str]]:
    """Collect predicted target weights, optional uncertainties, and realized returns."""
    loader: DataLoader[dict[str, Any]] = DataLoader(
        dataset=dataset,
        batch_size=256,
        shuffle=False,
    )
    model.eval()
    model.to(device=device)
    weights: list[np.ndarray] = []
    uncertainties: list[np.ndarray] = []
    realized_returns: list[np.ndarray] = []
    dates: list[str] = []
    with torch.no_grad():
        for batch in loader:
            moved_batch = move_batch_to_device(batch=batch, device=device)
            prediction: PortfolioPrediction = model.predict(
                batch=moved_batch,
                mc_samples=mc_samples,
            )
            weights.append(prediction.weights.detach().cpu().numpy())
            if prediction.weight_uncertainty is not None:
                uncertainties.append(
                    prediction.weight_uncertainty.detach().cpu().numpy()
                )
            realized_returns.append(
                moved_batch["next_log_returns"].detach().cpu().numpy()
            )
            dates.extend([str(value) for value in batch["date"]])
    uncertainty_array = (
        np.concatenate(uncertainties, axis=0) if uncertainties else None
    )
    return (
        np.concatenate(weights, axis=0),
        uncertainty_array,
        np.concatenate(realized_returns, axis=0),
        dates,
    )


def run_strategy(
    strategy_name: str,
    strategy_cfg: dict[str, Any],
    target_weights_hist: np.ndarray,
    uncertainties_hist: np.ndarray | None,
    next_log_returns: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Run one strategy over a sequence of target model weights."""
    strategy_cls = get_strategy(strategy_name)
    strategy = strategy_cls(strategy_cfg)
    num_periods, num_assets = target_weights_hist.shape
    current_weights = np.ones(num_assets, dtype=np.float64) / num_assets
    portfolio_returns: list[float] = []
    weights_history: list[np.ndarray] = []
    for period_idx in range(num_periods):
        target_weights = target_weights_hist[period_idx]
        uncertainties = (
            uncertainties_hist[period_idx] if uncertainties_hist is not None else None
        )
        new_weights = strategy.rebalance(
            current_weights=current_weights,
            target_weights=target_weights,
            uncertainties=uncertainties,
            returns=next_log_returns[: period_idx + 1],
            t=period_idx,
            cfg=strategy_cfg,
        )
        portfolio_returns.append(float(new_weights @ next_log_returns[period_idx]))
        weights_history.append(new_weights.copy())
        current_weights = new_weights
    return np.asarray(portfolio_returns), np.asarray(weights_history)


def compute_benchmarks(
    next_log_returns: np.ndarray,
    currencies: list[str],
    benchmarks_cfg: dict[str, Any],
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Compute benchmark return and weight histories."""
    num_periods, num_assets = next_log_returns.shape
    results: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    if benchmarks_cfg.get("USD", False):
        usd_weights = np.zeros((num_periods, num_assets), dtype=np.float64)
        usd_weights[:, currencies.index("USD")] = 1.0
        results["benchmark:USD"] = (np.zeros(num_periods, dtype=np.float64), usd_weights)
    if benchmarks_cfg.get("best_currency", False):
        best_idx = int(np.argmax(np.sum(next_log_returns, axis=0)))
        best_weights = np.zeros((num_periods, num_assets), dtype=np.float64)
        best_weights[:, best_idx] = 1.0
        results[f"benchmark:best_currency:{currencies[best_idx]}"] = (
            next_log_returns[:, best_idx],
            best_weights,
        )
    return results


def accumulated_value_path(log_returns: np.ndarray, initial_value: float) -> np.ndarray:
    """Convert log returns into a portfolio value path."""
    cumulative = np.concatenate(
        (np.array([0.0], dtype=np.float64), np.cumsum(log_returns))
    )
    return initial_value * np.exp(cumulative)


def plot_accumulated_returns(
    dates: list[str],
    strategy_returns: dict[str, np.ndarray],
    initial_value: float,
    output_path: str | Path,
) -> None:
    """Plot value paths for all strategies and benchmarks."""
    plt.figure(figsize=(11, 6))
    date_index = pd.to_datetime([dates[0], *dates])
    for name, log_returns in strategy_returns.items():
        values = accumulated_value_path(log_returns, initial_value=initial_value)
        linestyle = "--" if name.startswith("benchmark:") else "-"
        plt.plot(date_index, values, label=name, linestyle=linestyle)
    plt.title("Backtest portfolio values")
    plt.xlabel("Date")
    plt.ylabel("Portfolio value (USD)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, format="svg")
    plt.close()


def plot_allocations_summary(
    strategy_weights: dict[str, np.ndarray],
    currencies: list[str],
    output_path: str | Path,
) -> None:
    """Plot mean allocations by strategy with standard-deviation error bars."""
    strategy_names = [
        name for name in strategy_weights if not name.startswith("benchmark:")
    ]
    if not strategy_names:
        return
    num_strategies = len(strategy_names)
    fig, axes = plt.subplots(
        1,
        num_strategies,
        figsize=(max(5 * num_strategies, 10), 5),
        sharey=True,
    )
    if num_strategies == 1:
        axes = [axes]
    x = np.arange(len(currencies))
    for axis, strategy_name in zip(axes, strategy_names, strict=False):
        weights = strategy_weights[strategy_name]
        mean_weights = weights.mean(axis=0) * 100.0
        std_weights = weights.std(axis=0) * 100.0
        axis.bar(x, mean_weights, yerr=std_weights, color="#4C78A8", alpha=0.9)
        axis.set_xticks(x)
        axis.set_xticklabels(currencies, rotation=40, ha="right")
        axis.set_title(strategy_name)
        axis.set_ylabel("Average allocation (%)")
    fig.tight_layout()
    fig.savefig(output_path, format="svg")
    plt.close(fig)


def evaluate_backtest_strategies(
    checkpoint: str | Path,
    backtest_config: dict[str, Any],
    run_timestamp: str | None = None,
    output_dir: Path | None = None,
    file_prefix: str | None = None,
    write_plots: bool = True,
    shared_plot_paths: dict[str, str | Path] | None = None,
) -> dict[str, Any]:
    """Run multi-strategy backtests for one checkpoint."""
    checkpoint_path = resolve_checkpoint(checkpoint=checkpoint)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, model_config, metadata = load_model_from_checkpoint(
        path=checkpoint_path,
        map_location=device,
    )
    dates_config: dict[str, Any] = dict(backtest_config.get("dates") or {})
    dataset = ForexPortfolioDataset(
        processed_dir=str(backtest_config.get("processed_dir", "data/processed")),
        split="all",
        date_start=dates_config.get("start"),
        date_end=dates_config.get("end"),
    )
    model_eval_config: dict[str, Any] = model_config.get("eval", {})
    mc_samples = int(
        model_eval_config.get(
            "mc_samples",
            model_eval_config.get(
                "n_samples",
                model_config.get("mc_samples", 1),
            ),
        )
    )
    target_weights_hist, uncertainties_hist, next_log_returns, dates = (
        collect_strategy_predictions(
            model=model,
            dataset=dataset,
            device=device,
            mc_samples=mc_samples,
        )
    )
    strategy_names = resolve_enabled_entries(
        config_value=backtest_config.get("strategies"),
        valid_names=["full_rebalancing", "partial_rebalancing", "black_litterman"],
        default_names=["full_rebalancing"],
        field_name="strategies",
    )
    strategy_metrics: dict[str, dict[str, float]] = {}
    strategy_returns: dict[str, np.ndarray] = {}
    strategy_weights: dict[str, np.ndarray] = {}
    requested_metrics = resolve_enabled_entries(
        config_value=backtest_config.get("metrics"),
        valid_names=list(ALL_BACKTEST_METRICS),
        default_names=list(ALL_BACKTEST_METRICS),
        field_name="metrics",
    )
    for strategy_name in strategy_names:
        strategy_kind = (
            "probabilistic" if strategy_name in PROBABILISTIC_STRATEGIES else "pointwise"
        )
        strategy_cfg_path = Path(
            f"configs/strategies/{strategy_kind}/{strategy_name}.yaml"
        )
        strategy_cfg = load_yaml(strategy_cfg_path) if strategy_cfg_path.exists() else {}
        if strategy_name in PROBABILISTIC_STRATEGIES and uncertainties_hist is None:
            print(
                f"Warning: strategy '{strategy_name}' requires uncertainty and was skipped."
            )
            continue
        portfolio_returns, weights_history = run_strategy(
            strategy_name=strategy_name,
            strategy_cfg=strategy_cfg,
            target_weights_hist=target_weights_hist,
            uncertainties_hist=uncertainties_hist,
            next_log_returns=next_log_returns,
        )
        strategy_returns[strategy_name] = portfolio_returns
        strategy_weights[strategy_name] = weights_history
        metrics = compute_backtest_metrics(
            portfolio_values=accumulated_value_path(
                portfolio_returns,
                initial_value=float(backtest_config.get("initial_value", 1.0)),
            ),
            log_returns=portfolio_returns,
            weights=weights_history,
            trading_days_per_year=int(
                backtest_config.get("trading_days_per_year", 252)
            ),
        )
        strategy_metrics[strategy_name] = select_backtest_metrics(
            metrics=metrics,
            requested_metrics=requested_metrics,
        )

    benchmark_results = compute_benchmarks(
        next_log_returns=next_log_returns,
        currencies=list(metadata["currencies"]),
        benchmarks_cfg=dict(backtest_config.get("benchmarks") or {}),
    )
    benchmark_metrics: dict[str, dict[str, float]] = {}
    for benchmark_name, (portfolio_returns, weights_history) in benchmark_results.items():
        strategy_returns[benchmark_name] = portfolio_returns
        strategy_weights[benchmark_name] = weights_history
        metrics = compute_backtest_metrics(
            portfolio_values=accumulated_value_path(
                portfolio_returns,
                initial_value=float(backtest_config.get("initial_value", 1.0)),
            ),
            log_returns=portfolio_returns,
            weights=weights_history,
            trading_days_per_year=int(
                backtest_config.get("trading_days_per_year", 252)
            ),
        )
        benchmark_metrics[benchmark_name] = select_backtest_metrics(
            metrics=metrics,
            requested_metrics=requested_metrics,
        )

    if output_dir is None:
        output_root = ensure_dir(path=backtest_config.get("backtest_dir", "backtests"))
        run_timestamp = run_timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = ensure_dir(path=output_root / f"{run_timestamp}-{checkpoint_path.stem}")
    else:
        output_dir = ensure_dir(path=output_dir)
    prefix = "" if file_prefix is None else f"{file_prefix}_"
    accumulated_returns_path = output_dir / f"{prefix}accumulated_returns.csv"
    allocations_path = output_dir / f"{prefix}allocations.csv"
    metrics_path = output_dir / ("metrics.json" if file_prefix is None else f"{prefix}metrics.json")
    accumulated_returns_svg_path = output_dir / "accumulated_returns.svg"
    allocations_summary_svg_path = output_dir / "allocations_summary.svg"

    returns_frame = pd.DataFrame({"date": dates})
    for name, log_returns in strategy_returns.items():
        returns_frame[name] = log_returns
    returns_frame.to_csv(accumulated_returns_path, index=False)

    allocations_frame = pd.DataFrame({"date": dates})
    for name, weights_history in strategy_weights.items():
        for idx, currency in enumerate(metadata["currencies"]):
            allocations_frame[f"{name}:{currency}"] = weights_history[:, idx]
    allocations_frame.to_csv(allocations_path, index=False)

    if write_plots:
        plot_strategy_returns = {
            name if name.startswith("benchmark:") else f"{model_config.get('name')}:{name}": returns
            for name, returns in strategy_returns.items()
        }
        plot_strategy_weights = {
            f"{model_config.get('name')}:{name}": weights
            for name, weights in strategy_weights.items()
            if not name.startswith("benchmark:")
        }
        plot_accumulated_returns(
            dates=dates,
            strategy_returns=plot_strategy_returns,
            initial_value=float(backtest_config.get("initial_value", 1.0)),
            output_path=accumulated_returns_svg_path,
        )
        plot_allocations_summary(
            strategy_weights=plot_strategy_weights,
            currencies=list(metadata["currencies"]),
            output_path=allocations_summary_svg_path,
        )

    payload: dict[str, Any] = {
        "model_name": model_config.get("name"),
        "checkpoint": str(checkpoint_path),
        "dates": {
            "requested_start": dates_config.get("start"),
            "requested_end": dates_config.get("end"),
            "actual_start": dates[0],
            "actual_end": dates[-1],
        },
        "requested_metrics": list(requested_metrics or ALL_BACKTEST_METRICS),
        "strategies": strategy_metrics,
        "benchmarks": benchmark_metrics,
        "artifacts": {
            "accumulated_returns_csv": str(accumulated_returns_path),
            "allocations_csv": str(allocations_path),
            "accumulated_returns_svg": str(
                shared_plot_paths["accumulated_returns_svg"]
                if shared_plot_paths is not None
                else accumulated_returns_svg_path
            ),
            "allocations_summary_svg": str(
                shared_plot_paths["allocations_summary_svg"]
                if shared_plot_paths is not None
                else allocations_summary_svg_path
            ),
        },
    }
    write_json(data=payload, path=metrics_path)
    payload["metrics_path"] = str(metrics_path)
    payload["_dates"] = dates
    payload["_currencies"] = list(metadata["currencies"])
    payload["_strategy_returns"] = strategy_returns
    payload["_strategy_weights"] = strategy_weights
    return payload


def main() -> None:
    """Run multi-strategy backtests from command-line arguments."""
    args = parse_args()
    config = load_yaml(path=args.config)
    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    payloads: list[dict[str, Any]] = []
    if len(args.checkpoint) == 1:
        payload = evaluate_backtest_strategies(
            checkpoint=args.checkpoint[0],
            backtest_config=config,
            run_timestamp=run_timestamp,
        )
        for key in ("_dates", "_currencies", "_strategy_returns", "_strategy_weights"):
            payload.pop(key, None)
        payloads.append(payload)
    else:
        output_root = ensure_dir(path=config.get("backtest_dir", "backtests"))
        shared_output_dir = ensure_dir(path=output_root / f"{run_timestamp}-multi-backtest")
        shared_plot_paths = {
            "accumulated_returns_svg": shared_output_dir / "accumulated_returns.svg",
            "allocations_summary_svg": shared_output_dir / "allocations_summary.svg",
        }
        combined_returns: dict[str, np.ndarray] = {}
        combined_weights: dict[str, np.ndarray] = {}
        shared_dates: list[str] | None = None
        shared_currencies: list[str] | None = None

        for checkpoint in args.checkpoint:
            checkpoint_stem = resolve_checkpoint(checkpoint=checkpoint).stem
            payload = evaluate_backtest_strategies(
                checkpoint=checkpoint,
                backtest_config=config,
                run_timestamp=run_timestamp,
                output_dir=shared_output_dir,
                file_prefix=checkpoint_stem,
                write_plots=False,
                shared_plot_paths=shared_plot_paths,
            )
            if shared_dates is None:
                shared_dates = list(payload["_dates"])
            if shared_currencies is None:
                shared_currencies = list(payload["_currencies"])

            for name, log_returns in payload["_strategy_returns"].items():
                label = name if name.startswith("benchmark:") else f"{payload['model_name']}:{name}"
                if label not in combined_returns:
                    combined_returns[label] = log_returns
            for name, weights_history in payload["_strategy_weights"].items():
                if name.startswith("benchmark:"):
                    continue
                combined_weights[f"{payload['model_name']}:{name}"] = weights_history

            for key in ("_dates", "_currencies", "_strategy_returns", "_strategy_weights"):
                payload.pop(key, None)
            payloads.append(payload)

        if shared_dates is not None:
            plot_accumulated_returns(
                dates=shared_dates,
                strategy_returns=combined_returns,
                initial_value=float(config.get("initial_value", 1.0)),
                output_path=shared_plot_paths["accumulated_returns_svg"],
            )
        if shared_dates is not None and shared_currencies is not None:
            plot_allocations_summary(
                strategy_weights=combined_weights,
                currencies=shared_currencies,
                output_path=shared_plot_paths["allocations_summary_svg"],
            )
        write_json(data={"backtests": payloads}, path=shared_output_dir / "metrics.json")
    if len(payloads) == 1:
        print(payloads[0])
    else:
        print(payloads)


if __name__ == "__main__":
    main()
