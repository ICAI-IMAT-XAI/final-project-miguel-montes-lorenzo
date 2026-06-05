# Architecture

This repository implements an Explainable AI case study for daily foreign
exchange portfolio allocation. The main explained model is a Heterogeneous
Temporal Graph Neural Network (HTGNN) that allocates across eight currencies
using historical portfolio signals and macro-financial market blocks.

## Repository Layout

```text
configs/
  download.yaml                         Yahoo Finance symbol universe
  transform.yaml                        preprocessing, lookback, and splits
  train.yaml                            global training settings
  eval.yaml                             post-training evaluation settings
  backtest.yaml                         strategy backtest settings
  models/
    pointwise/NN.yaml                   deterministic signal-only baseline
    pointwise/HTGNN.yaml                main graph model
    probabilistic/BNN.yaml              dropout-based signal-only baseline
    probabilistic/BHTGNN.yaml           dropout-based graph model
  strategies/
    pointwise/                          deterministic rebalancing strategies
    probabilistic/                      uncertainty-aware strategies
src/
  data/                                 download, transform, dataset adapters
  models/                               model registry and implementations
  strategies/                           portfolio strategy implementations
  metrics/                              performance and stability metrics
  xai/                                  XAI methods and plotting utilities
  train.py                              training entry point
  eval.py                               evaluation entry point
  backtest.py                           backtest entry point
xai/
  xai.py                                XAI pipeline CLI
  xai.ipynb                             reproducible workflow notebook
report/
  main.typ                              report source
  main.pdf                              compiled report
```

Generated data, checkpoints, runs, backtests, evaluation outputs, and XAI
artifacts are intentionally ignored by Git.

## Data Flow

The project starts from adjusted Yahoo Finance daily prices. The downloader
stores:

```text
data/raw/prices.parquet
data/raw/symbol_metadata.parquet
data/raw/blocks/<node>.parquet
```

The transform step builds the supervised dataset in `data/processed/`. The
current configuration uses:

- lookback window: 20 trading days;
- train split: 2018-01-01 to 2023-12-29 samples;
- validation split: 2024-01-01 to 2024-12-31 samples;
- test split: 2025-01-02 to 2025-12-31 samples;
- cross-sectional centering of currency returns before lookback window construction
  (`center: true` in `configs/transform.yaml`).

The target currency basket is:

```text
USD, EUR, JPY, GBP, CNY, CAD, AUD, CHF
```

For each non-USD currency, the Yahoo Finance pair `USDXXX=X` is interpreted as
units of currency `XXX` per one USD. Holding `XXX` in USD terms therefore has
the opposite log return:

```text
r_USD_value(XXX) = -log_return(USDXXX=X)
r_USD_value(USD) = 0
```

After cross-sectional centering, USD is no longer a permanently zero channel:
it represents the negative average movement of the currency basket at that date.
This is important for the SHAP and Integrated Gradients analyses.

## Supervised Target

The models are trained to imitate a long-only mean-variance teacher portfolio,
not to forecast raw prices directly. For each sample, the teacher allocation is
computed from:

- the next realised cross-section of currency returns as a mean proxy;
- a recent covariance estimate from the lookback window;
- a risk-aversion parameter and ridge term from the model config.

The HTGNN output is a long-only allocation vector on the simplex plus a positive
variance estimate. The deterministic models use this output directly; the
probabilistic variants use Monte Carlo dropout at inference time.

## Heterogeneous Temporal Graph

The graph uses one node per market block, not one node per ticker. The observed
nodes are:

```text
portfolio_signal
commodity_future
commodity_etf
equity_index_future
us_treasury_future
fx_future
us_bond_etf
fx_usd_pair
eur_bond_etf
jpy_bond_etf
gbp_bond_etf
cny_bond_etf
cad_bond_etf
aud_bond_etf
chf_bond_etf
```

Each market node receives a `(batch, lookback, node_features)` tensor containing
the standardised returns of the tickers in that block. The `portfolio_signal`
node receives the 20-day history of the eight centred currency-return channels.

The main HTGNN configuration also adds three learned generic latent nodes:

```text
generic_latent_1
generic_latent_2
generic_latent_3
```

These nodes have learned features rather than observed market inputs. They are
connected to market and portfolio nodes so they can, in principle, capture
cross-feature interactions. The current XAI results suggest that their final
contribution is small.

## HTGNN Model

The main model is configured in `configs/models/pointwise/HTGNN.yaml`:

```text
hidden_dim: 96
generic_node_count: 3
generic_node_feature_dim: 32
gru_layers: 1
message_passing_layers: 3
gate: vectorial
dropout: 0.10
loss: wasserstein
epochs: 300
early_stopping_patience: 25
checkpoint: best
```

The forward pass is:

1. Encode every observed node window with a node-specific GRU.
2. Encode the learned generic-node features with a small MLP.
3. Build a fixed directed heterogeneous graph.
4. Apply relation-specific message passing layers.
5. Read the final `portfolio_signal` state.
6. Produce Dirichlet concentrations for the allocation weights.
7. Produce a positive scalar variance estimate.

The edge set is dense around `portfolio_signal` and the generic nodes:

- every market node connects to `portfolio_signal`;
- `portfolio_signal` connects back to every market node;
- generic nodes connect to portfolio, market, and other generic nodes;
- relation names depend on source/target semantics.

Examples of relation types:

```text
fx_to_portfolio
rates_to_portfolio
commodity_to_portfolio
equity_to_portfolio
portfolio_to_market
portfolio_to_generic
market_to_generic
generic_to_portfolio
generic_to_market
generic_cross
```

Message updates use relation-specific linear transforms, average aggregation by
incoming degree, a residual MLP update, optional vectorial gating, and layer
normalisation.

## Model Families

The model registry exposes four families:

- `NN`: deterministic signal-only MLP baseline.
- `HTGNN`: deterministic heterogeneous temporal graph model.
- `BNN`: signal-only MLP with Monte Carlo dropout.
- `BHTGNN`: graph model with Monte Carlo dropout.

The final report focuses on the deterministic HTGNN checkpoint because it is the
model explained by the XAI pipeline and backtest figures.

## Training, Evaluation, and Backtesting

The canonical run order is:

```bash
python -m src.data.download --config configs/download.yaml
python -m src.data.transform --config configs/transform.yaml
python -m src.train --config configs/models/pointwise/HTGNN.yaml
python -m src.backtest --checkpoint latest --config configs/backtest.yaml
```

Training merges the global `configs/train.yaml` settings with the selected model
config. The best validation checkpoint is saved under `checkpoints/`, and
post-training evaluation writes plots and metrics under `evaluation/`.

The backtest loads a checkpoint and applies configured strategies over
2024-01-01 to 2025-12-31. The main report uses `full_rebalancing` and compares
against USD and best-single-currency benchmarks. Backtest outputs are written to
timestamped directories under `backtests/`.

## XAI Pipeline

The main XAI entry point is:

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

The pipeline writes a manifest to `xai/xai_outputs_manifest.json` and groups
artifacts into:

```text
xai/global_explanations/
xai/local_explanations/
xai/evaluation/
```

The current implemented explanation methods are:

- portfolio-signal SHAP surrogate:
  `portfolio_signal_shap_mean.svg`,
  `portfolio_signal_shap_mean_signed.svg`,
  `portfolio_signal_shap_variance.svg`,
  `portfolio_signal_shap_variance_signed.svg`;
- Integrated Gradients over observed input nodes:
  `integrated_gradients_graph.svg`,
  `integrated_gradients_node_importance.svg`,
  `input_attention_pie_charts.svg`;
- Integrated Gradients from the states before the last message-passing layer:
  `integrated_gradients_last_message_graph.svg`;
- temporal occlusion over five lookback windows:
  `temporal_occlusion_windows.svg`;
- node-wise temporal occlusion over the same five windows:
  `node_temporal_occlusion_windows.svg`;
- deletion/insertion faithfulness checks:
  `deletion_insertion_curve.svg`,
  `deletion_insertion_vs_full_model_curve.svg`.

For the report, the most important XAI artifacts are copied or regenerated under
`report/figures/xai/`.

## Report and Notebook

The report is built with Typst:

```bash
typst compile report/main.typ report/main.pdf
```

The notebook `xai/xai.ipynb` mirrors the full workflow: preprocessing, training,
backtesting, XAI execution, and visual inspection of every figure used in the
report and appendix. Its first cell locates the repository root and switches the
working directory there, so the notebook works whether it is opened from the
repo root or from the `xai/` subdirectory.
