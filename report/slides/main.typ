// ============================================================
//  slides/main.typ
//  20-slide presentation for the GDL final project
//  Built with diatypst: https://typst.app/universe/package/diatypst
// ============================================================

#import "@preview/diatypst:0.9.1": *

#set par(justify: false)
#set text(size: 10pt)

#let accent = rgb("#1f4e79")
#let soft = rgb("#edf3f8")
#let line = rgb("#c9d6e2")
#let muted = rgb("#4f5f6f")
#let warn = rgb("#f6efe6")
#let todo = rgb("#fff6d7")

#let chip(body) = rect(
  inset: (x: 7pt, y: 3pt),
  radius: 5pt,
  fill: soft,
  stroke: 0.5pt + line,
)[#text(size: 10pt, fill: accent)[#body]]

#let card(body) = rect(
  width: 100%,
  inset: 9pt,
  radius: 6pt,
  fill: rgb("#ffffff"),
  stroke: 0.7pt + line,
)[#body]

#let note(body) = rect(
  width: 100%,
  inset: 8pt,
  radius: 5pt,
  fill: todo,
  stroke: 0.6pt + rgb("#d9b650"),
)[#text(size: 12pt)[#body]]

#let metricbox(title, body) = rect(
  inset: 8pt,
  radius: 5pt,
  fill: soft,
  stroke: 0.7pt + line,
)[
  #text(weight: "bold", fill: accent)[#title]

  #body
]

#show heading.where(level: 2): it => [
  #pagebreak(weak: true)
  #v(0.8cm)
]

#show: slides.with(
  title: "HTGNN for Forex Portfolio Allocation",
  subtitle: "Geometric Deep Learning Final Project",
  authors: ("Miguel Montes Lorenzo",),
  date: "Academic year 2025-2026",
  ratio: 16 / 9,
  layout: "medium",
  title-color: accent,
  theme: "full",
  count: "number",
  footer: true,
  toc: false,
  first-slide: false,
)

== 01. Project in one sentence

#align(center)[
  #text(size: 25pt, weight: "bold", fill: accent)[
    Heterogeneous temporal graph learning for FX allocation
  ]
]

#v(0.5em)

#grid(
  columns: (1.1fr, 0.9fr),
  gutter: 1em,
  [#card[
    *Research question.* Can a heterogeneous temporal graph neural network use
    macro-financial market blocks to improve currency allocation over a
    signal-only neural baseline?

    #v(0.3em)
    The target is not a raw return forecast. The model predicts a long-only
    allocation vector over an eight-currency basket.
  ]],
  [#card[
    #chip[Forex] #chip[HTGNN] #chip[Mean-variance target]
  ]],
)

== 02. Why Forex is a useful financial prediction domain

#grid(
  columns: (1fr, 1fr),
  gutter: 1em,
  [#card[
    *The prediction object is relative.*

    An exchange rate is always defined against another currency. This naturally
    fits a cross-sectional allocation problem: capital is redistributed across
    currencies under a common numeraire.
  ]],
  [#card[
    *Drivers are macro-financial.*

    FX is strongly linked to monetary policy, inflation, external balances,
    global risk appetite, commodities, and rate differentials. These drivers can
    be grouped into interpretable market blocks.
  ]],
)

#v(0.4em)

#card[
  The project fixes USD as numeraire and allocates across
  $cal(C) = {"USD", "EUR", "JPY", "GBP", "CNY", "CAD", "AUD", "CHF"}$.
]

== 03. Advantages over standard equity-return prediction

#grid(
  columns: (1fr, 1fr, 1fr),
  gutter: 0.8em,
  [#metricbox[Less firm-idiosyncratic noise][
    Currencies are less exposed to accounting surprises, management guidance,
    mergers, litigation, or single-firm shocks.
  ]],
  [#metricbox[More accessible context][
    Rates, commodities, equity-risk proxies, and currency futures provide
    observable signals that can be organised as typed nodes.
  ]],
  [#metricbox[Compact output][
    Eight currencies produce an interpretable simplex-valued prediction rather
    than a huge universe of weakly comparable stocks.
  ]],
)

#v(0.5em)

#card[
  This does not make FX easy. It makes the information structure more suitable
  for a small heterogeneous temporal graph experiment.
]

== 04. Main disadvantages and risk sources

#grid(
  columns: (1fr, 1fr),
  gutter: 1em,
  [#rect(width: 100%, inset: 9pt, radius: 6pt, fill: warn, stroke: 0.7pt + line)[
    *No natural productivity-growth premium.*

    Equities can rise because firms become more productive and expand real cash
    flows. A currency allocation is mostly relative value under a chosen
    numeraire.
  ]],
  [#rect(width: 100%, inset: 9pt, radius: 6pt, fill: warn, stroke: 0.7pt + line)[
    *Direct exposure to macro shocks.*

    Inflation, central-bank intervention, capital controls, geopolitical shocks,
    and liquidity regimes can dominate short-horizon returns.
  ]],
)

#v(0.5em)

#card[
  The task is therefore useful for testing structured macro-financial modelling,
  not for claiming that FX returns are easy to predict.
]

== 05. Why a heterogeneous temporal GNN

#align(center + horizon)[
  #image("figures/graphics/htgnn_architecture.png", width: 100%)
]

== 06. Data pipeline

#grid(
  columns: (1fr, 1fr),
  gutter: 1em,
  [#card[
    *Raw source.* Daily adjusted Yahoo Finance price series.

    *Period.* 2017-01-01 to 2026-01-01.

    *Transformation.* Prices are converted to log returns.
  ]],
  [#card[
    *Window.* 20 trading days.

    *Splits.* Chronological train, validation, and test splits.

    *Filtering.* Short gaps are forward-filled and highly sparse instruments are
    removed before tensor construction.
  ]],
)

#v(0.6em)

#align(center)[#text(fill: muted)[Daily market blocks -> temporal tensors -> supervised allocation dataset]]

== 07. Heterogeneous market blocks

#table(
  columns: (1.1fr, 2fr, 2.8fr),
  inset: (x: 5pt, y: 4pt),
  stroke: 0.45pt + line,
  [*Block*], [*Examples*], [*Forex relevance*],
  [`commodity_future`], [`GC=F`, `CL=F`, `NG=F`], [Inflation, terms of trade, AUD/CAD sensitivity.],
  [`equity_index_future`], [`ES=F`, `NQ=F`, `RTY=F`], [Global risk appetite and safe-haven flows.],
  [`us_treasury_future`], [`ZB=F`, `ZN=F`, `ZF=F`], [USD rates curve and duration shocks.],
  [`fx_future`], [`6E=F`, `6J=F`, `6A=F`], [Liquid alternative FX-market signals.],
  [`us_bond_etf`], [`SHY`, `IEF`, `TLT`, `HYG`], [Rates, credit, inflation, and risk regimes.],
  [`fx_usd_pair`], [`USDJPY=X`, `USDCAD=X`], [Direct quotes for target construction.],
)

== 08. USD-denominated currency returns

The Yahoo Finance quote `USDXXX=X` is interpreted as units of currency `XXX` per
one USD. Therefore, the USD-denominated return of holding currency `XXX` uses the
opposite sign of the quoted pair return.

#v(0.4em)

#align(center)[
  $
    r_(t, "XXX")^"USD"
    = log(V_t^"XXX" / V_(t-1)^"XXX")
    = - (log P_t^("USD/XXX") - log P_(t-1)^("USD/XXX"))
  $
]

#align(center)[
  $ r_(t, "USD")^"USD" = 0 $
]

#v(0.4em)

#card[
  This convention makes all target returns comparable inside the same portfolio
  simplex.
]

== 09. Signal structuring: from returns to allocations

The project does not ask the network to predict raw future returns directly.
Instead, it builds an allocation target from a mean-variance teacher portfolio.

#v(0.4em)

#align(center)[
  $
    bold(w)^star_t
    = op("argmax")_(bold(w) in Delta_N)
    bold(mu)_t^T bold(w) - lambda / 2 bold(w)^T Sigma_t bold(w)
  $
]

#v(0.4em)

#grid(
  columns: (1fr, 1fr),
  gutter: 1em,
  [#card[*Input signal.* Historical allocation-like sequence derived from past FX returns.]],
  [#card[*Target signal.* Ex-post teacher allocation derived from the future realised cross-section.]],
)

== 10. Why allocation targets are preferable here

#grid(
  columns: (1fr, 1fr, 1fr),
  gutter: 0.8em,
  [#metricbox[Lower noise objective][
    The model learns the decision implied by returns and covariance, not every
    noisy return component independently.
  ]],
  [#metricbox[Closer to usage][
    The final object needed by a portfolio system is $bold(w)$, not an isolated
    vector of return forecasts.
  ]],
  [#metricbox[Less over-exploited][
    Direct return prediction is a heavily mined target. Allocation imitation
    encodes a financial decision rule instead.
  ]],
)

#v(0.5em)

#align(center)[
  $
    hat(bold(w))_t = bold(alpha)_t / (bold(1)^T bold(alpha)_t), quad
    bold(alpha)_t > 0
  $
]

== 11. Asset allocation strategies

#grid(
  columns: (1fr, 1fr, 1fr),
  gutter: 0.8em,
  [#metricbox[Mean-variance][
    Uses the model allocation signal or inferred expected returns to trade off
    return and risk under long-only simplex constraints.
  ]],
  [#metricbox[Direct model weights][
    Uses the predicted simplex vector directly as the portfolio allocation.
  ]],
  [#metricbox[Black-Litterman style][
    Blends prior equilibrium views with model-implied views when uncertainty
    information is available.
  ]],
)

#v(0.5em)

#card[
  In the current report, the deterministic pipeline is the core analysis target;
  uncertainty-aware extensions are left for future work.
]

== 12. Baseline: deterministic signal-only NN

#grid(
  columns: (1.1fr, 0.9fr),
  gutter: 1em,
  [#card[
    The baseline receives only the portfolio signal. It does not receive the
    heterogeneous market blocks used by the HTGNN.

    This is not a fully fair architecture-only comparison, but it is a realistic
    signal-only financial modelling baseline.
  ]],
  [#card[
    $ bold(z)_t = phi_theta(op("vec")(X_t^"portfolio")) $

    $ bold(alpha)_t = op("softplus")(W_alpha bold(z)_t + bold(b)_alpha) + epsilon $

    $ hat(bold(w))_t = bold(alpha)_t / (bold(1)^T bold(alpha)_t) $
  ]],
)

== 13. Results analysis I - evaluation protocol

#grid(
  columns: (1fr, 1fr),
  gutter: 1em,
  [#card[
    *Protocol.*

    - Training period: 2018-01-01 to 2024-01-01
    - Validation period: 2024-01-01 to 2025-01-01
    - Test period: 2025-01-01 to 2026-01-01
    - Daily data, 20-day lookback, same target construction for both models
  ]],
  [#card[
    *Reference checkpoints.*

    - `HTGNN-20260507_154855.pt`
    - `NN-20260507_165401.pt`

    *Evaluation focus.*

    - Direct portfolio performance
    - Allocation stability
    - Allocation dynamics and interpretability
  ]],
)

#v(0.5em)

#table(
  columns: (1.2fr, 1.3fr, 1.3fr, 1.3fr),
  inset: (x: 5pt, y: 4pt),
  stroke: 0.45pt + line,
  [*Model*], [*Train period*], [*Validation period*], [*Test period*],
  [NN], [2018-2024], [2024-2025], [2025-2026],
  [HTGNN], [2018-2024], [2024-2025], [2025-2026],
)

== 14. Results analysis II - supervised allocation metrics

#grid(
  columns: (1.45fr, 0.85fr),
  gutter: 1em,
  [#image("figures/results/htgnn_vs_backtest_baselines.svg", width: 92%)],
  [#card[
    *Main result.*

    The deterministic HTGNN strongly exceeds all reference comparators over the
    two-year evaluation window.

    - Cumulative return: *265.7%*
    - Annualized return: *87.0%*
    - Max drawdown: *-0.64%*

    The gap is not marginal; it is a clearly different performance regime.

    #v(0.4em)
    Reference comparators shown here: deterministic NN, `100% USD`, and
    `100% CHF` as the hindsight-best single-currency benchmark.
  ]],
)

== 15. Results analysis III - realised portfolio performance

#grid(
  columns: (1.05fr, 0.95fr),
  gutter: 1em,
  [#image("figures/results/htgnn_backtest_allocation_histogram.svg", width: 100%)],
  [#card[
    *Allocation behaviour.*

    The HTGNN is not just betting permanently on one or two currencies.

    Largest average weights:

    - AUD: 23.0%
    - JPY: 17.3%
    - USD: 15.9%

    Other meaningful exposures remain active:

    - CNY: 11.7%
    - GBP: 10.9%
    - CHF: 9.4%
    - CAD: 8.7%
  ]],
)

#v(0.35em)

#card[
  So the result is economically more interesting than a degenerate single-currency
  rule: the portfolio remains relatively balanced while still achieving very
  high realised performance.
]

== 16. Results analysis IV - risk and drawdown diagnostics

#grid(
  columns: (1.3fr, 1fr),
  gutter: 1em,
  [#image("figures/results/htgnn_vs_nn.svg", width: 100%)],
  [#card[
    *HTGNN vs NN.*

    - HTGNN cumulative return: *265.7%*
    - NN cumulative return: *1.4%*
    - HTGNN annualized return: *87.0%*
    - NN annualized return: *0.68%*

    The heterogeneous graph clearly dominates the signal-only MLP.
  ]],
)

#v(0.35em)

#table(
  columns: (1.4fr, 1fr, 1fr, 1fr, 1fr),
  inset: (x: 4pt, y: 3pt),
  stroke: 0.4pt + line,
  [*Model*], [*Cum. ret.*], [*Ann. ret.*], [*Sharpe*], [*Max DD*],
  [HTGNN], [*265.7%*], [*87.0%*], [*9.26*], [*-0.64%*],
  [NN], [1.4%], [0.68%], [0.14], [-7.34%],
  [100% USD], [0.0%], [0.0%], [0.00], [0.00%],
)

== 17. Results analysis V - allocation behaviour

#grid(
  columns: (1.3fr, 1fr),
  gutter: 1em,
  [#grid(
    columns: 3,
    gutter: 0.5em,
    [#image("figures/results/input_lag1_heatmap.svg", width: 100%)],
    [#image("figures/results/htgnn_lag1_heatmap.svg", width: 100%)],
    [#image("figures/results/nn_lag1_heatmap.svg", width: 100%)],
  )],
  [#card[
    *Lag-1 correlation structure.*

    In the input signal and in the HTGNN outputs, lag-1 allocation
    correlations stay generally low and diffuse.

    The NN, in contrast, shows much stronger day-to-day persistence.

    *Interpretation:* this FX task is better suited to combining heterogeneous
    market blocks than to relying mainly on temporal autocorrelation.
  ]],
)

== 18. Results analysis VI - HTGNN vs NN interpretation

#grid(
  columns: (1.2fr, 1fr),
  gutter: 1em,
  [#image("figures/results/htgnn_focus_graph.svg", width: 100%)],
  [#card[
    *What the HTGNN uses most.*

    - `fx_future`: 33.8%
    - `us_bond_etf`: 8.5%
    - `portfolio_signal`: 7.9%
    - `commodity_future`: 7.8%
    - `fx_usd_pair`: 6.0%

    *Interesting detail.*

    Currency-specific treasury proxy nodes receive relatively little attention.
    The model seems to extract more signal from broad cross-asset macro blocks
    than from local sovereign-bond proxies alone.
  ]],
)

== 19. Conclusions

#card[
  *Main takeaways.*

  - The deterministic HTGNN clearly outperforms the deterministic NN baseline
    and the market baselines on the two-year evaluation period.

  - The gain does not come from simple temporal persistence: unlike the NN, the
    HTGNN keeps low lag-1 allocation correlations, closer to the structure of
    the input allocation signal.

  - The node-focus analysis is economically plausible: the model relies most on
    FX futures and broad macro-financial blocks, which supports the value of a
    heterogeneous graph view of the FX allocation problem.
]

== 20. Future work

#grid(
  columns: (1fr, 1fr),
  gutter: 1em,
  [#card[
    *Bayesian HTGNN extension.*

    Financial prediction is dominated by uncertainty and partial information.
    A probabilistic HTGNN could express confidence in allocations and support
    more conservative decision rules.
  ]],
  [#card[
    *Stronger financial evaluation.*

    Add turnover costs, rolling retraining, stress-period analysis, robustness
    across seeds, and ablations over market blocks and relation types.
  ]],
)

#v(0.6em)

#align(center)[
  #image("figures/graphics/htgnn_architecture.png", width: 70%)
]
