= Results

The analysis compares the deterministic HTGNN with a deterministic signal-only
NN. Bayesian variants are omitted so the main question stays clear: does
heterogeneous temporal market context improve allocation?

== HTGNN Performance Against Backtest Baselines

Over 2024-01-01 to 2025-12-31, the HTGNN separates from the reference backtest
baselines. With direct full rebalancing, it reaches 265.7% cumulative return and
87.0% annualized return.

#figure(
  image("../figures/results/htgnn_vs_backtest_baselines.svg", width: 60%),
  caption: [Backtest comparison between the deterministic HTGNN and three reference comparators: the deterministic NN baseline, 100% USD, and 100% CHF as the hindsight-best single-currency benchmark.],
) <fig-htgnn-vs-baselines>

The performance does not come from collapsing into one or two currencies. AUD
(23.0%), JPY (17.3%), and USD (15.9%) are largest on average, but CNY, GBP, CHF,
and CAD also contribute; EUR remains secondary at 3.1%.

#figure(
  image("../figures/results/htgnn_backtest_allocation_histogram.svg", width: 60%),
  caption: [Average backtest allocation of the deterministic HTGNN. The portfolio is not concentrated permanently in one or two currencies, but spread across several relevant exposures.],
) <fig-htgnn-allocation-hist>

@fig-htgnn-vs-baselines and @fig-htgnn-allocation-hist show a profitable but
still diversified allocation, rather than a single-currency allocation.

== HTGNN Versus the Pointwise NN

The direct research comparison is HTGNN versus signal-only NN. The graph model
outperforms the baseline.

#figure(
  image("../figures/results/htgnn_vs_nn.svg", width: 60%),
  caption: [Backtest comparison between the deterministic HTGNN, the deterministic NN baseline, and the 100% USD baseline.],
) <fig-htgnn-vs-nn>

The NN baseline ends with only 1.4% cumulative return and 0.68% annualized
return, barely above 100% USD. The gap suggests that heterogeneous context is
material for this FOREX allocation problem.

The final comparison reports performance metrics and allocation-stability
metrics for HTGNN, NN, and the 100% USD benchmark.

#figure(
  [
    #set text(size: 8pt)
    #table(
      columns: (2.3fr, 0.95fr, 0.95fr, 1fr, 0.95fr, 1fr, 0.95fr),
      inset: (x: 3pt, y: 3pt),
      stroke: 0.35pt + rgb("#d9dee7"),
      [*System*], [*Cum. ret.*], [*Ann. ret.*], [*Ann. vol.*], [*Sharpe*], [*Sortino*], [*Max DD*],
      [HTGNN], [*265.7%*], [*87.0%*], [6.76%], [*9.26*], [*40.82*], [*-0.64%*],
      [NN], [1.4%], [0.68%], [4.90%], [0.14], [0.22], [-7.34%],
      [100% USD], [0.0%], [0.0%], [*0.00%*], [0.00], [0.00], [0.00%],
    )
  ],
  caption: [Performance metrics on the evaluation period. Bold marks the best value in each column, with lower values preferred for annualized volatility and maximum drawdown.],
) <tab-performance-results>

#figure(
  [
    #set text(size: 8pt)
    #table(
      columns: (2.2fr, 1fr, 1fr),
      inset: (x: 4pt, y: 4pt),
      stroke: 0.35pt + rgb("#d9dee7"),
      [*System*], [*Mean turnover*], [*Mean Herfindahl*],
      [HTGNN], [0.728], [0.606],
      [NN], [*0.142*], [*0.138*],
      [100% USD], [*0.000*], [1.000],
    )
  ],
  caption: [Allocation-stability metrics on the evaluation period. Bold marks the best value in each column, with lower values preferred in both cases.],
) <tab-stability-results>

@tab-performance-results shows that the HTGNN is best on every return-oriented
metric among learned models. @tab-stability-results adds one nuance: the NN is
more stable by turnover and concentration, but that
conservatism does not translate into competitive returns or drawdown.

== Lag-1 Correlation Structure

The lag-1 heatmaps compare the input allocation signal with realized HTGNN and
NN allocations.

#figure(
  align(center)[
    #box(
      width: 60%,
      grid(
        columns: (1fr, 1fr, 1fr),
        gutter: 10pt,
        [#align(center)[*Input allocation signal*]],
        [#align(center)[*HTGNN allocations*]],
        [#align(center)[*NN allocations*]],
        [#image("../figures/results/input_lag1_heatmap.svg", width: 100%)],
        [#image("../figures/results/htgnn_lag1_heatmap.svg", width: 100%)],
        [#image("../figures/results/nn_lag1_heatmap.svg", width: 100%)],
      ),
    )
  ],
  caption: [Lag-1 allocation-correlation heatmaps for the input allocation signal, the HTGNN predictions, and the NN predictions.],
) <fig-lag1-heatmaps>

The HTGNN heatmap is closer to the input signal than to the NN. Its lag-1
correlations remain low and diffuse, while the NN is slightly more persistent. This
supports the view that useful FOREX allocation information is cross-asset and
heterogeneous, not purely autoregressive.

== What the HTGNN Pays Attention To

Node focus is estimated by ablation: each node is zeroed out and the resulting
allocation change is measured. It is not GAT attention, but it is an informative
proxy for model reliance.

#figure(
  image("../figures/results/htgnn_focus_graph.svg", width: 60%),
  caption: [Node-focus graph for the final HTGNN checkpoint, estimated by node ablation over the test split. Redder nodes correspond to larger average influence on the final allocation.],
) <fig-htgnn-focus>

@fig-htgnn-focus gives an interpretable ranking. `fx_future` dominates (33.8%),
followed by `us_bond_etf`, `portfolio_signal`, `commodity_future`,
`fx_usd_pair`, and `eur_bond_etf`. Currency-specific bond ETF nodes receive much
less focus, suggesting that broad FX, fixed-income, and commodity regime signals
carry more useful structure here than local treasury proxies alone.
