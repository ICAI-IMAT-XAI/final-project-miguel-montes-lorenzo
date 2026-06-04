# Application of HTGNN to Currency Trading (FOREX)

This repository implements the Geometric AI project topic:

> GNN-4: Heterogeneous or Temporal Graphs. Extend standard GNN architectures to handle heterogeneous graphs or temporal/dynamic graphs.

The application is currency portfolio allocation over eight currencies:

```text
USD, EUR, JPY, GBP, CNY, CAD, AUD, CHF
```

The pointwise models use previous currency-return windows or mean-variance weight windows derived inside the model. The graph models add heterogeneous temporal market blocks built from commodities, futures, bond ETFs, FX pairs, and currency-specific bond ETF proxies.

## Repository structure

```text
configs/
  download.yaml
  transform.yaml
  train.yaml
  eval.yaml
  backtest.yaml
  models/
    pointwise/
      NN.yaml
      HTGNN.yaml
    probabilistic/
      BNN.yaml
      BHTGNN.yaml
src/
  data/
    symbols.py        # Symbol universe and target currencies
    download.py       # Yahoo Finance download script
    transform.py      # Returns and supervised tensors
    dataset.py        # PyTorch dataset
  models/
    pointwise/NN/     # Deterministic pointwise MLP
    pointwise/HTGNN/  # Heterogeneous temporal GNN
    probabilistic/BNN/
    probabilistic/BHTGNN/
  metrics/
  train.py
  eval.py
  backtest.py
scripts/
  run_all.sh
```

The `report`, `slides`, and web application directories are intentionally omitted for now.

## Installation

Using `venv`:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Using `uv`:

```bash
uv venv
source .venv/bin/activate
uv pip install -r requirements.txt
```

## Data download

The symbol universe is defined in `src/data/symbols.py`. It includes the symbols you listed, grouped into heterogeneous graph nodes.

Download prices from Yahoo Finance:

```bash
python -m src.data.download --config configs/download.yaml
```

This creates:

```text
data/raw/prices.parquet
data/raw/symbol_metadata.parquet
data/raw/blocks/*.parquet
```

## Data transformation

Create currency returns, graph node tensors, and train/validation/test splits:

```bash
python -m src.data.transform --config configs/transform.yaml
```

This creates:

```text
data/processed/portfolio_features.npy
data/processed/portfolio_raw_returns.npy
data/processed/next_log_returns.npy
data/processed/dates.npy
data/processed/metadata.json
data/processed/nodes/*.npy
```

## Models

### Pointwise

`NN` receives the previous portfolio signal window and predicts the next allocation plus next variance. When `input_format: "weights"`, the mean-variance weight sequence is derived inside the model from returns and rolling covariance.

```bash
python -m src.train --config configs/models/pointwise/NN.yaml
```

`HTGNN` receives the same portfolio signal plus heterogeneous temporal market nodes.

```bash
python -m src.train --config configs/models/pointwise/HTGNN.yaml
```

### Probabilistic

`BNN` is the Bayesian pointwise model, implemented with Bayes-by-backprop
linear layers.

```bash
python -m src.train --config configs/models/probabilistic/BNN.yaml
```

`BHTGNN` is the Bayesian heterogeneous temporal graph model, also implemented with MC dropout.

```bash
python -m src.train --config configs/models/probabilistic/BHTGNN.yaml
```

Training creates checkpoints in:

```text
checkpoints/*.pt
```

and one run directory per checkpoint in:

```text
runs/<checkpoint>/
```

Each run directory stores the TensorBoard event files plus a copy of the exact
`train.yaml` and model YAML used for that execution.

To inspect the train and validation loss curves:

```bash
tensorboard --logdir runs
```

TensorBoard logging is enabled by default in `configs/train.yaml`. If training
fails with a missing dependency error, reinstall project requirements so the
`tensorboard` package is available in the active virtual environment.

## Evaluation and plots

Evaluate any saved checkpoint:

```bash
python -m src.eval --checkpoint checkpoints/NN-YYYYMMDD_HHMMSS.pt
python -m src.eval --checkpoint checkpoints/HTGNN-YYYYMMDD_HHMMSS.pt
python -m src.eval --checkpoint checkpoints/BNN-YYYYMMDD_HHMMSS.pt
python -m src.eval --checkpoint checkpoints/BHTGNN-YYYYMMDD_HHMMSS.pt
```

Use the most recently saved checkpoint with:

```bash
python -m src.eval --checkpoint latest
```

The evaluation period is controlled by `configs/eval.yaml`:

```yaml
dates:
  start: "2025-01-01"
  end: "2026-01-01"
```

Probabilistic sampling settings live in each model's `eval` section. The global
evaluation config is deterministic: it selects data, computes realized returns,
and writes the same artifacts for all models.

Training date windows are controlled independently in `configs/train.yaml`:

```yaml
dates:
  train: ["2018-01-01", "2024-01-01"]
  validation: ["2024-01-01", "2025-01-01"]
```

Each evaluation creates:

```text
evaluation/<eval_timestamp>-<checkpoint>/portfolio_value.png
evaluation/<eval_timestamp>-<checkpoint>/metrics.json
evaluation/<eval_timestamp>-<checkpoint>/predictions.csv
evaluation/<eval_timestamp>-<checkpoint>/mean_currency_allocations.svg
evaluation/<eval_timestamp>-<checkpoint>/allocation_lag1_correlation_heatmap.svg
```

You can choose which scalar metrics are computed in `configs/eval.yaml` and
`configs/backtest.yaml` with:

```yaml
metrics:
  cumulative_return: true
  annualized_return: true
  annualized_volatility: true
  sharpe_ratio: true
  sortino_ratio: true
  max_drawdown: true
  mean_turnover: true
  mean_herfindahl: true
```

The portfolio plot uses USD portfolio value as the dependent variable and includes two dashed baselines:

1. `100% USD`: horizontal line, because holding USD has value 1 in USD terms.
2. `100% best currency`: hindsight line for the currency that appreciated the most against USD during the evaluation period.

## Backtest

Backtest any trained checkpoint with:

```bash
python -m src.backtest --checkpoint checkpoints/NN-YYYYMMDD_HHMMSS.pt --config configs/backtest.yaml
```

Or use the latest checkpoint:

```bash
python -m src.backtest --checkpoint latest
```

You can also backtest several checkpoints in one command:

```bash
python -m src.backtest \
  --checkpoint checkpoints/HTGNN-YYYYMMDD_HHMMSS.pt checkpoints/NN-YYYYMMDD_HHMMSS.pt \
  --config configs/backtest.yaml
```

Unlike `src.eval`, which applies the model's raw predicted allocation directly,
`src.backtest` runs one or more rebalancing strategies on top of the model
predictions over the requested date range.

The default `configs/backtest.yaml` includes:

```yaml
dates:
  start: "2024-01-01"
  end: "2026-01-01"

strategies:
  full_rebalancing: true
  partial_rebalancing: true
  black_litterman: true

benchmarks:
  USD: true
  best_currency: true
  SP500: false
```

Backtests are written to `backtests/`:

```text
backtests/<run_timestamp>-<checkpoint>/metrics.json
backtests/<run_timestamp>-<checkpoint>/accumulated_returns.csv
backtests/<run_timestamp>-<checkpoint>/allocations.csv
backtests/<run_timestamp>-<checkpoint>/accumulated_returns.svg
backtests/<run_timestamp>-<checkpoint>/allocations_summary.svg
```

When several checkpoints are passed in one command, `backtest` writes them into
the same directory:

```text
backtests/<run_timestamp>-multi-backtest/metrics.json
backtests/<run_timestamp>-multi-backtest/<checkpoint_1>_metrics.json
backtests/<run_timestamp>-multi-backtest/<checkpoint_1>_accumulated_returns.csv
backtests/<run_timestamp>-multi-backtest/<checkpoint_1>_allocations.csv
backtests/<run_timestamp>-multi-backtest/<checkpoint_2>_metrics.json
backtests/<run_timestamp>-multi-backtest/<checkpoint_2>_accumulated_returns.csv
backtests/<run_timestamp>-multi-backtest/<checkpoint_2>_allocations.csv
backtests/<run_timestamp>-multi-backtest/accumulated_returns.svg
backtests/<run_timestamp>-multi-backtest/allocations_summary.svg
```

The shared SVG files compare all requested checkpoints together. Benchmarks are
shown in the performance plot, but omitted from `allocations_summary.svg`.

## Analysis visualizations

The repository also includes analysis scripts that write SVG figures under
`visualizations/`.

To inspect the mean-variance target allocations rebuilt from the processed
inputs:

```bash
python -m src.analysis.data
```

This creates:

```text
visualizations/inputs/target_allocation_histogram.svg
visualizations/inputs/target_allocation_lag1_correlation_heatmap.svg
```

To inspect which heterogeneous nodes an `HTGNN` relies on, using node ablation
as a focus proxy:

```bash
python -m src.analysis.HTGNN_focus --checkpoint checkpoints/HTGNN-YYYYMMDD_HHMMSS.pt
```

To do the same for `BHTGNN`:

```bash
python -m src.analysis.BHTGNN_focus --checkpoint checkpoints/BHTGNN-YYYYMMDD_HHMMSS.pt
```

Each checkpoint gets its own output directory:

```text
visualizations/<checkpoint>/node_focus_graph.svg
visualizations/<checkpoint>/node_focus_histogram.svg
```

## Run the full pipeline

```bash
bash scripts/run_all.sh
```
