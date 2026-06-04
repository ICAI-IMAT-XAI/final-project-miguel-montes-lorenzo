# Architecture

This repository mirrors the structure of the probabilistic AI portfolio repo but is specialized for the Geometric AI assignment.

The project ignores report, slides, and web deployment code for now. The core tree is:

```text
configs/
  download.yaml
  transform.yaml
  train.yaml
  eval.yaml
  models/
    pointwise/NN.yaml
    pointwise/HTGNN.yaml
    probabilistic/BNN.yaml
    probabilistic/BHTGNN.yaml
src/
  data/
    symbols.py
    download.py
    transform.py
    dataset.py
  models/
    pointwise/NN/
    pointwise/HTGNN/
    probabilistic/BNN/
    probabilistic/BHTGNN/
  metrics/
  train.py
  eval.py
  backtest.py
```

## Data representation

The target portfolio is formed by eight currencies measured in USD terms:

```text
USD, EUR, JPY, GBP, CNY, CAD, AUD, CHF
```

For non-USD currencies the Yahoo series `USDXXX=X` is interpreted as units of `XXX` per USD. Therefore the USD value of one unit of `XXX` is proportional to `1 / USDXXX`, and its log return in USD is `-log_return(USDXXX=X)`. USD receives a zero log return.

The supervised target is a rolling long-only Markowitz allocation over the eight currency returns. The input for the pointwise models is the previous sequence of these rolling optimal allocations plus the previous rolling portfolio volatility. The output is the next allocation and next portfolio variance.

## Heterogeneous temporal graph

The graph model uses one node per market block/category, not one node per individual symbol. This is intentional because the requested symbol universe is naturally grouped into macro/market blocks:

- `portfolio_signal`
- `commodity_future`
- `commodity_etf`
- `equity_index_future`
- `us_treasury_future`
- `fx_future`
- `us_bond_etf`
- `fx_usd_pair`
- `eur_bond_etf`
- `jpy_bond_etf`
- `gbp_bond_etf`
- `cny_bond_etf`
- `cad_bond_etf`
- `aud_bond_etf`
- `chf_bond_etf`

Each market node receives a window of daily returns for all symbols in that block. The `portfolio_signal` node receives the same input used by NN and BNN.

Each node is encoded with a GRU. Then typed message passing sends information between `portfolio_signal` and the market nodes using relation types such as `fx_to_portfolio`, `rates_to_portfolio`, `commodity_to_portfolio`, `equity_to_portfolio`, and `portfolio_to_market`. The portfolio node is used for the final readout.

## Model families

- `NN`: deterministic pointwise MLP.
- `BNN`: MC-dropout Bayesian pointwise MLP.
- `HTGNN`: deterministic heterogeneous temporal graph neural network.
- `BHTGNN`: MC-dropout Bayesian heterogeneous temporal graph neural network.

The Bayesian models are pragmatic Bayesian approximations using dropout at inference time. This keeps the implementation lightweight and compatible with the original assignment scope without introducing PyMC/Pyro complexity.
