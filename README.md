# HTGNN Forex XAI Case Study

This repository contains an end-to-end Explainable AI case study for daily
foreign-exchange portfolio allocation with a heterogeneous temporal graph neural
network.

## Repository Structure

```text
configs/                  Data, model, training, backtest, and evaluation configs
src/data/                 Data download, preprocessing, and dataset construction
src/models/               Neural portfolio models, including HTGNN variants
src/strategies/           Portfolio rebalancing strategies used in backtests
src/metrics/              Portfolio and backtest metrics
src/xai/                  Explainability method implementations
xai/xai.py                Main XAI pipeline entry point
xai/xai.ipynb             Notebook with training, backtest, and XAI workflow
report/main.typ           Written report source
report/main.pdf           Compiled written report
requirements.txt          Python dependencies
```

Generated outputs are intentionally ignored by Git:

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

## Requirements

Create and activate a Python environment, then install the dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The report is written in Typst. To rebuild the PDF, install Typst and run the
compile command shown below.

## Main Experiments

Run the following commands from the repository root.

Download the raw Yahoo Finance market data:

```bash
python -m src.data.download --config configs/download.yaml
```

Transform prices into supervised tensors and heterogeneous node windows:

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
  --max-samples 260 \
  --background-samples 32 \
  --n-local 3 \
  --top-k 5 \
  --seed 42 \
  --methods all
```

Rebuild the written report:

```bash
typst compile report/main.typ report/main.pdf
```

## Notebook

The notebook [xai/xai.ipynb](xai/xai.ipynb) contains the reproducible workflow
required for the project: data loading and preprocessing, model training,
backtest evaluation, XAI computation, visualisation of the report figures, and
short comments on the main results.

## External Resources

External code, libraries, datasets, and XAI methods used in the project are
cited in [report/references.bib](report/references.bib).
