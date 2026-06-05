# HTGNN FOREX XAI Case Study

This repository contains an end-to-end Explainable AI case study for FOREX
portfolio allocation with heterogeneous temporal graph neural networks.

The task is to predict daily currency portfolio allocations from historical
market windows. The main explained model is an HTGNN/BHTGNN-style portfolio
model trained on heterogeneous temporal nodes such as FX futures, bond ETFs,
commodities, equity futures, and portfolio-signal history.

## Repository Structure

```text
configs/                  Experiment, data, model, evaluation, and backtest configs
src/data/                 Data download, transformation, and PyTorch dataset code
src/models/               NN, HTGNN, BNN, and BHTGNN model implementations
src/strategies/           Portfolio rebalancing strategies used by backtests
src/metrics/              Backtest and portfolio metrics
src/xai/                  XAI method implementations
xai/xai.py                Main XAI pipeline entry point
xai/xai.ipynb             Reproducible notebook for training, backtest, and XAI
requirements.txt          Python dependencies
```

Generated artifacts are written to ignored output directories:

```text
data/
checkpoints/
runs/
backtests/
evaluation/
visualizations/
xai/global_explanations/
xai/local_explanations/
xai/evaluation/
```

## Environment

Create and activate a Python environment:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Alternatively, with `uv`:

```bash
uv venv
source .venv/bin/activate
uv pip install -r requirements.txt
```

## Run the Main Experiments

Download raw market data:

```bash
python -m src.data.download --config configs/download.yaml
```

Transform raw prices into supervised tensors:

```bash
python -m src.data.transform --config configs/transform.yaml
```

Train the main HTGNN model:

```bash
python -m src.train --config configs/models/pointwise/HTGNN.yaml
```

Run the backtest for the latest checkpoint:

```bash
python -m src.backtest --checkpoint latest --config configs/backtest.yaml
```

Run the XAI pipeline for the latest checkpoint:

```bash
python -m xai.xai \
  --checkpoint latest \
  --model auto \
  --split test \
  --target variance \
  --output-dir xai \
  --device cpu \
  --max-samples 128 \
  --background-samples 32 \
  --n-local 3 \
  --top-k 5 \
  --seed 42 \
  --methods all
```

## Notebook

The notebook [xai/xai.ipynb](xai/xai.ipynb) shows the complete workflow:

1. data preparation commands,
2. model training,
3. backtest execution,
4. backtest plot inspection,
5. XAI pipeline execution,
6. XAI plot inspection with short comments.

