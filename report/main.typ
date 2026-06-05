// ============================================================
//  main.typ - AI Ethics and Explainability final report
// ============================================================

#import "@preview/bloated-neurips:0.7.0": neurips2023
#import "template.typ": compact-table, note

#set page(margin: (x: 1.35cm, y: 1.6cm))
#set text(font: "TeX Gyre Termes", size: 9pt)

#let authors = (
  (
    name: "Miguel Montes Lorenzo",
    affl: ("imat", "comillas"),
    email: "202105503@alu.comillas.edu",
  ),
)

#let affls = (
  "imat": (
    department: "Mathematical Engineering and Artificial Intelligence",
    institution: "",
    location: "",
  ),
  "comillas": (
    department: "",
    institution: "Universidad Pontificia Comillas",
    location: "Madrid, Spain",
  ),
)

#show: neurips2023.with(
  title: [
    Explainability of a Heterogeneous Temporal Graph Neural Network for Forex Allocation
    #linebreak()
    #text(size: 12pt, weight: "regular")[AI Ethics and Explainability]
  ],
  authors: (authors, affls),
  keywords: (
    "explainable AI",
    "foreign exchange",
    "heterogeneous temporal graphs",
    "portfolio allocation",
    "model risk",
  ),
  abstract: [
    #set text(size: 9pt)
    Foreign exchange allocation is a high-stakes prediction setting because the
    model output is not an isolated forecast but a capital allocation across
    relative macro assets. This report studies the explainability of a
    Heterogeneous Temporal Graph Neural Network (HTGNN) that allocates across
    USD, EUR, JPY, GBP, CNY, CAD, AUD, and CHF using a portfolio-signal node and
    macro-financial context nodes such as FX futures, currency pairs, rates,
    commodities, and equity risk. The goal is not only to report backtest
    performance, but to assess whether the model's decisions can be related to
    interpretable market channels. The analysis combines three families of XAI
    methods: a surrogate SHAP analysis restricted to the portfolio-signal node,
    gradient-based Integrated Gradients over the full heterogeneous input graph,
    and perturbation-based temporal occlusion plus deletion/insertion
    faithfulness checks. The results show a profitable but diversified HTGNN
    backtest and explanations that consistently point to FX derivatives,
    commodity-linked information, recent temporal windows, and the portfolio
    signal as relevant decision drivers. The report also discusses limitations:
    short evaluation horizon, low-fidelity linear SHAP surrogate, baseline
    sensitivity, turnover risk, and the need for explanation-stability
    monitoring before any domain deployment.
  ],
  accepted: none,
  aux: (
    get-notice: accepted => [],
  ),
)

#set page(
  margin: (left: 2.2cm, right: 2.2cm, top: 1.7cm, bottom: 1.7cm),
  footer: context {
    let page-number = counter(page).at(here()).first()
    if page-number == 1 {
      []
    } else {
      align(center, text(size: 10pt, [#page-number]))
    }
  },
)
#show figure: set block(spacing: 1.6em)
#let small-caption(width: 100%, it) = {
  set align(center)
  block(width: width, context {
    set align(center)
    set text(size: 8pt)
    it.supplement
    if it.numbering != none {
      [ ]
      it.counter.display(it.numbering)
    }
    it.separator
    [ ]
    it.body
  })
}
#show figure.caption.where(kind: image): small-caption.with(width: 60%)
#show figure.caption.where(kind: table): small-caption
#let bottom-figure(it) = {
  let body = block(width: 100%, spacing: 1.6em, {
    set align(center)
    it.body
    v(8pt, weak: true)
    it.caption
  })

  if it.placement == none {
    body
  } else {
    place(it.placement, body, float: true, clearance: 2.3em)
  }
}
#show figure.where(kind: image): bottom-figure
#show figure.where(kind: table): bottom-figure

#let f(path, width: 100%) = image(path, width: width)
#let fx-basket = [USD, EUR, JPY, GBP, CNY, CAD, AUD, CHF]

= Introduction and Motivation

Foreign exchange (Forex or FX) is inherently relative: each currency is priced
against another numeraire. This project fixes USD as numeraire and allocates
across #fx-basket. The prediction target is therefore not a price, but a
long-only portfolio $bold(w)_t in Delta^N$, with
$Delta^N = {bold(w) in RR^N_+ : bold(1)^T bold(w) = 1}$.

Forex is an interesting domain because currencies are macro assets. Their
returns are linked to monetary policy, inflation, external balances, commodity
exposure, safe-haven flows, and global risk appetite. These drivers are not
independent scalar features; they are naturally organised into market blocks.
The HTGNN hypothesis is that a typed temporal graph can use those blocks more
appropriately than a signal-only architecture.

The ethical and explanatory motivation is just as important as the predictive
one. A currency allocation model affects capital allocation and risk exposure.
Relevant stakeholders include portfolio managers, model validators, risk
officers, clients, and counterparties exposed to the resulting trades. For these
stakeholders, a high backtest return is not enough: the model should provide
evidence that its allocations are driven by plausible market channels rather
than by leakage, unstable noise, or accidental correlations. The report therefore
focuses on the question: which nodes, windows, and input currencies appear to
drive the HTGNN's allocation decisions?

= Data and Model

The repository uses daily adjusted Yahoo Finance price series, downloaded with
`yfinance` @yfinance, from 2017-01-01 to 2025-12-31. Prices are converted to log
returns, short gaps are forward-filled with a three-day limit, and instruments
with excessive missingness are removed. The supervised tensors use a 20-day
lookback and chronological splits: 2018-01-01 to 2024-01-01 for training,
2024-01-01 to 2025-01-01 for validation, and 2025-01-01 to 2026-01-01 for test.
The models are implemented in PyTorch @paszke2019pytorch.

For every non-USD currency, the Yahoo Finance quote `USDXXX=X` is interpreted as
units of currency XXX per one USD. Hence the USD-denominated return of holding
currency XXX has the opposite sign of the quoted pair return:

$ r_(t,"XXX")^"USD" = - (log P_t^"USD/XXX" - log P_(t-1)^"USD/XXX"), quad
  r_(t,"USD")^"USD" = 0. $

The final preprocessing centres the eight currency returns cross-sectionally
before window construction. This matters for explainability: without centring,
the raw USD channel would always be zero; after centring, USD represents the
negative average movement of the basket and can receive non-zero attribution.

Training uses a mean-variance teacher inspired by portfolio optimisation
@markowitz1952portfolio. Let $bold(r)_t in RR^N$ be the USD-valued currency
return vector and $R_t$ the recent lookback window. The target mean proxy is the
next realised cross-section centred around its own average,
$tilde(bold(mu))_(t+1) = bold(r)_(t+1) - 1/N bold(1) bold(1)^T bold(r)_(t+1)$.
The covariance $hat(Sigma)_t$ is estimated from the lookback window with a ridge
term, and the ex-post teacher allocation is

$ bold(w)^*_(t+1) =
  arg max_(bold(w) in Delta^N)
  bold(w)^T tilde(bold(mu))_(t+1) -
  gamma / 2 bold(w)^T hat(Sigma)_t bold(w). $

The next return is used only to build the supervised teacher during training.
At inference, the model receives past node windows and outputs an allocation.

#figure(
  f("figures/graphics/htgnn_architecture.png", width: 64%),
  caption: [Implemented HTGNN architecture. Each market block is encoded as a temporal node, relation-specific message passing is applied over a portfolio-centred heterogeneous graph, and the final allocation is read from the portfolio node.]
) <fig-architecture>

For each observed node $v$, the input is a window
$X_(v,t-L+1:t) in RR^(L times F_v)$. Each node has its own GRU input dimension,
but all nodes are mapped to a shared hidden size $H$:
$bold(h)^0_v = op("GRU")_v(X_(v,t-L+1:t)) in RR^H$. After temporal encoding,
the HTGNN applies relation-typed message passing, combining the motivation of
temporal graph modelling @rossi2020temporal with relational and heterogeneous
graph models @schlichtkrull2018modeling @hu2020heterogeneous:

$ bold(m)^ell_v = 1 / |cal(N)(v)| sum_((u,v,r) in cal(E)) W^ell_r bold(h)^ell_u, $

$ bold(h)^(ell+1)_v =
op("LayerNorm")(bold(h)^ell_v + op("MLP")_ell([bold(h)^ell_v; bold(m)^ell_v])). $

The selected HTGNN uses hidden dimension 96, one GRU layer, relation-specific
message passing, GELU activations, dropout 0.10, batch size 64, and a sliced
Wasserstein allocation loss. The final `portfolio_signal` state is mapped to
Dirichlet concentrations, $bold(alpha)_t = op("softplus")(bold(a)_t) + epsilon$,
and the predicted allocation is
$hat(bold(w))_t = bold(alpha)_t / (bold(1)^T bold(alpha)_t)$.

= XAI Methodology

The XAI pipeline is organised into three complementary classes:
surrogate/approximation methods, gradient-based attribution methods, and
perturbation-based methods. This classification is useful because each class
answers a different question. Surrogates ask whether a simpler model can
approximate part of the HTGNN. Gradients ask how a local output changes along an
interpolation path. Perturbations ask what happens to realised performance when
information is removed.

== Surrogate / approximation method: portfolio-signal SHAP

SHAP values are based on additive feature attributions @lundberg2017unified. A
full SHAP analysis over the HTGNN input would be too large: the model receives
many nodes, each with a 20-day window and a different number of symbols, and the
output has eight currency allocations. Instead, the implemented SHAP experiment
focuses on the `portfolio_signal` node. This is not the whole model, but it is
the most directly interpretable allocation-history input.

The full 20-day `portfolio_signal` window is flattened into 160 features:
20 lags times 8 currencies. For each output currency, a standardised ridge
surrogate is fitted to the HTGNN output. Linear SHAP-style contributions are
then computed as

$ phi_j(x) = beta_j (z_j - overline(z)_j), $

where $z_j$ is the standardised surrogate feature and $beta_j$ is the fitted
coefficient. Contributions are summed across the 20 lags for each input
currency, producing an 8-by-8 grid. The absolute SHAP grid is used to rank
importance. The signed grid is also saved because it can reveal net directional
tendencies, but signed means can be close to zero when positive and negative
effects cancel. In this report, "mean" means the predicted mean allocation
weight, not expected currency return.

== Gradient-based attribution: Integrated Gradients

Integrated Gradients (IG) attributes a differentiable output to input features by
integrating gradients along a path from a baseline $x'$ to the observed input
$x$ @sundararajan2017axiomatic:

$ "IG"_i(x) = (x_i - x'_i)
  integral_0^1 (partial F(x' + alpha (x - x'))) / (partial x_i) dif alpha. $

For this model, IG is computed over the observed heterogeneous nodes. The
baseline is the mean tensor of each node over a background subset, not a zero
tensor. This choice is important: after standardisation, zero may be meaningful
for some channels, but replacing all nodes by zero is less faithful to the
empirical graph distribution than using a background mean. The implementation
uses 24 interpolation steps and aggregates absolute attributions by node. It
also decomposes attribution inside each node to produce feature-level pie charts.

The explained score is aligned with the model's own allocation direction: for
each local batch, the prediction is used as a direction vector and the scalar
score is the inner product between predicted weights and that direction. This
keeps IG focused on allocation behaviour rather than on a single arbitrary
currency output.

== Perturbation methods: occlusion and deletion/insertion

Occlusion methods replace part of the input and measure output or performance
changes, an idea related to early perturbation-based explanation work
@zeiler2014visualizing. Here the replacement value is again the background mean
tensor per node. This is preferred to zero because masking should represent
"typical missing information", not an artificial market state.

Temporal occlusion splits the 20-day lookback into five windows: 0--4, 4--8,
8--12, 12--16, and 16--20. In global temporal occlusion, the selected window is
replaced by the background mean slice for every observed node. In node-window
occlusion, the same operation is applied to one node at a time. The metric is
the drop in performance ratio, where performance is annualized gross growth of
the model strategy divided by annualized gross growth of an equal-weight
portfolio.

Deletion/insertion is a faithfulness check inspired by perturbation tests such
as deletion/insertion curves in vision explanations @petsiuk2018rise. Nodes are
ordered by explanation score. Deletion starts from the full input and
progressively replaces the top nodes by their background baselines. Insertion
starts from the complementary masked input and progressively restores the same
nodes. Two views are reported: one relative to the equal-weight standard
portfolio and one relative to the full unperturbed model. The report uses the
full-model view because it directly asks how much of the trained model's own
performance is retained.

= Results

== Backtest and allocation behaviour

@fig-backtest-returns and @fig-backtest-allocations summarise the latest HTGNN
checkpoint. The data split is chronological: train from 2018-01-01 to
2023-12-29, validation from 2024-01-01 to 2024-12-31, and test from 2025-01-02
to 2025-12-31. The displayed backtest uses the configured strategy-evaluation
window 2024-01-01--2025-12-31, so it covers both the validation year and the
test year. The accumulated-return curve separates clearly from the USD benchmark
and from the hindsight best single currency, CHF.
The model reaches 256.2% cumulative return and 84.6% annualized return, with
4.8% annualized volatility and -0.84% maximum drawdown. CHF, the best
single-currency benchmark, reaches only 6.0% cumulative return.

#grid(
  columns: (1fr, 1fr),
  gutter: 8pt,
  [
    #figure(
      f("figures/backtest/accumulated_returns.svg", width: 100%),
      caption: [Accumulated returns for the latest HTGNN checkpoint against enabled benchmarks.]
    ) <fig-backtest-returns>
  ],
  [
    #figure(
      f("figures/backtest/allocations_summary.svg", width: 100%),
      caption: [Average currency allocations for the latest HTGNN checkpoint.]
    ) <fig-backtest-allocations>
  ],
)

The allocation histogram in @fig-backtest-allocations is important because it rules
out a trivial explanation of the return curve. The strategy does not simply hold
CHF or collapse into one currency. Average allocation is largest in AUD
(27.1%) and JPY (20.3%), followed by GBP (14.0%), CHF (11.7%), USD (10.5%), CAD
(7.6%), CNY (5.9%), and EUR (2.8%). This supports the interpretation that the
HTGNN is learning a diversified relative-value policy, although the high
turnover observed in the backtest still makes transaction-cost analysis
necessary.

#figure(
  table(
    columns: (1.5fr, 1fr, 1fr, 1fr),
    align: (left, right, right, right),
    inset: (x: 4pt, y: 3pt),
    stroke: 0.35pt + rgb("#d9dee7"),
    [Metric], [HTGNN], [USD], [Best single FX (CHF)],
    [Cumulative return], [256.2%], [0.0%], [6.0%],
    [Annualized return], [84.6%], [0.0%], [2.8%],
    [Annualized volatility], [4.8%], [0.0%], [5.0%],
    [Sharpe ratio], [12.84], [0.00], [0.56],
    [Maximum drawdown], [-0.84%], [0.0%], [-4.65%],
    [Mean turnover], [0.703], [0.000], [0.000],
  ),
  caption: [Backtest metrics for the latest HTGNN checkpoint and enabled benchmarks.]
) <tab-backtest>

== Key XAI results

@fig-shap-mean shows the absolute SHAP grid for predicted mean allocations. The
strongest cells concentrate in the AUD output. JPY, AUD, EUR, GBP, CAD, USD,
CHF, and CNY all contribute to the AUD allocation, with JPY and AUD largest.
This is useful because AUD is also the largest average allocation in
@fig-backtest-allocations. The signed version in @fig-app-shap-mean-signed is
almost zero everywhere. The contrast is informative: absolute SHAP reveals that
the portfolio-signal inputs matter in magnitude, while signed SHAP shows that
their direction is not stable across samples and largely cancels out. However,
the SHAP surrogate has low fidelity: the mean-allocation surrogate $R^2$ is
0.050. The correct conclusion is therefore not that the whole HTGNN is linear in
the portfolio signal, but that within this restricted portfolio-signal view, AUD
is the most sensitive output.

#figure(
  f("figures/xai/portfolio_signal_shap_mean.svg", width: 93%),
  caption: [Absolute portfolio-signal SHAP influence on predicted mean allocations. Each subplot fixes one output currency.]
) <fig-shap-mean>

Integrated Gradients, shown in @fig-ig-graph, gives a broader full-graph view.
Five nodes receive especially high focus: `fx_future`, `commodity_future`,
`portfolio_signal`, `fx_usd_pair`, and `commodity_etf`. This supports a market
interpretation of the HTGNN: direct currency derivative information is central,
commodities help contextualise commodity-sensitive currencies such as AUD and
CAD, and the portfolio signal still matters as a rolling allocation state. The
histogram in @fig-app-ig-hist shows the same ranking and, importantly, small
variance intervals for each node's focus, meaning that the local examples do not
produce a highly unstable attention pattern.

Given the strong backtest, one possible concern is data leakage: perhaps one
feature inside a highly attended node is accidentally carrying future
information. The feature-level pie charts in @fig-app-pies reduce that concern:
inside the most attended nodes, attribution is visually spread across several
features rather than concentrated in one suspicious channel. This does not prove
absence of leakage, but it makes a single-feature leakage explanation less
plausible.

Finally, the generic latent nodes are almost zero in @fig-ig-graph because that
IG graph propagates gradients all the way to observed input features, and the
generic nodes are not observed input-feature blocks. The last-message graph in
@fig-app-ig-last-message stops the attribution at the states before the final
message passing layer. It shows that generic nodes are not exactly zero, but
their focus remains negligible, around 0.08--0.09% each, compared with
`portfolio_signal`, `fx_future`, and `fx_usd_pair`. This suggests that the
model is not making effective use of the generic nodes to model useful
cross-feature information across the observed feature nodes.

#figure(
  f("figures/xai/integrated_gradients_graph.svg", width: 68%),
  caption: [Integrated Gradients node focus on the HTGNN topology.]
) <fig-ig-graph>

@fig-di-full and @fig-temporal-global combine two perturbation results.
Deletion/insertion relative to the full model indicates that the first three removed nodes
(`fx_usd_pair`, `equity_index_future`, and `portfolio_signal`) only modestly
reduce retained performance. The major break appears when `fx_future` is also
deleted: retained performance falls to roughly 65% of the full model. Insertion
mirrors this behaviour: restoring the first three nodes is not enough, while
restoring the fourth recovers almost full-model performance. This suggests
joint dependence and partial redundancy among the top market blocks.

The temporal occlusion panel in @fig-temporal-global shows that the most
recent window, days 16--20, is the most helpful: masking it lowers the
performance ratio by about 0.118. By contrast, masking days 12--16 improves the
performance ratio by about 0.103. This is one of the most actionable XAI
findings: the model benefits from very recent information but may also carry a
mid-window pattern that is harmful on the test sample.

The node-wise temporal occlusion appendix figure (@fig-app-node-occ) adds a
negative nuance for `commodity_future`. This node receives high IG focus, but
occluding it does not produce a robust performance drop across windows: the
perturbed strategy remains profitable in every window, and for several later
windows the drop is negative, meaning that masking the node actually improves
the performance ratio. This suggests that commodity futures are influential but
may inject noisy or harmful temporal context in part of the test period.

#grid(
  columns: (1fr, 1fr),
  gutter: 8pt,
  [
    #figure(
      f("figures/xai/deletion_insertion_vs_full_model_curve.svg", width: 100%),
      caption: [Deletion/insertion faithfulness check relative to the full model.]
    ) <fig-di-full>
  ],
  [
    #figure(
      f("figures/xai/temporal_occlusion_windows.svg", width: 100%),
      caption: [Global temporal occlusion over five lookback windows.]
    ) <fig-temporal-global>
  ],
)

Additional figures are included in the appendix: signed SHAP for mean,
absolute/signed SHAP for variance, the IG node histogram, the last-message IG
graph, IG feature pie charts, and node-wise temporal occlusion. The
deletion/insertion curve relative to the equal-weight benchmark is omitted
because @fig-di-full is more directly aligned with model faithfulness.

= Actions and Insights

== Insights from XAI

The XAI results point to a coherent model story. First, the strongest global
allocation behaviour is AUD-heavy, and the portfolio-signal SHAP grid also
shows AUD as the output most sensitive to recent allocation-regime inputs.
Second, the full-graph IG view shifts attention from currencies to market
channels: `fx_future`, commodities, and the portfolio signal dominate. Third,
the perturbation checks show that the model is not dependent on one isolated
node; performance degrades sharply only after several top nodes are removed.

Taken together, these observations suggest that the HTGNN uses both direct FX
information and broader macro-financial context. This is exactly the type of
behaviour the heterogeneous graph architecture was designed to allow. The
explanations also identify a weaker point: the model is temporally asymmetric.
Recent information is useful, but the previous four-day window can be harmful.
That finding is not visible from backtest performance alone.

== Model actions

The first action is to improve temporal selectivity. Since the most recent
window helps and the 12--16 window hurts, the model should be extended with a
learned temporal attention or gating mechanism before message passing. This
would let the model learn which parts of the 20-day sequence should be
emphasised rather than compressing the whole window through a GRU state.

The second action is explanation monitoring. The IG graph indicates that
`fx_future` and commodity futures are major drivers. In a future deployment, data
quality alerts should be node-specific: stale FX futures or missing commodity
data should be treated as model-risk events, not merely as generic missing
values.

The third action is not to over-correct every surprising XAI result. Low signed
SHAP values are not necessarily a problem because signed contributions cancel by
construction when effects change sign across samples. Similarly, redundancy in
deletion/insertion is not automatically bad; it can indicate that the graph has
multiple related channels carrying overlapping market information. What should
be mitigated is unexplained instability, not every non-sparse explanation.

== Domain recommendations

For portfolio use, the model should be reported through economic channels:
"FX derivatives", "commodity-linked risk", "portfolio-regime signal", and
"equity-risk context", rather than raw tensor names. This makes the explanation
usable for risk committees and clients.

The model should also be evaluated under transaction costs and turnover
constraints before any realistic trading interpretation. The backtest return is
large, but mean turnover is 0.703, so implementation frictions could materially
change performance. Finally, because FX regimes are policy-sensitive, any
allocation recommendation should be accompanied by rolling-window and
stress-period explanations. A model that looks interpretable in a calm period
may rely on different nodes during a central-bank or geopolitical shock.

= Discussion: Limitations, Risks, and Future Work

The main limitation is the short evaluation horizon. The 2024--2025 backtest is
strong, but FX markets can change abruptly through inflation surprises,
monetary-policy divergence, interventions, and geopolitical events. A two-year
test period is not enough to establish structural profitability.

The second limitation is explanatory fidelity. SHAP is deliberately restricted
to `portfolio_signal`, and its ridge surrogate has low $R^2$. It is useful for a
focused question about allocation-signal currencies, but not a full explanation
of the HTGNN. IG and perturbation methods cover the full graph, but they depend
on the background mean baseline. A different baseline, such as zero, rolling
normal conditions, or stress-period averages, could change the magnitude of the
explanations.

There are also model-risk and ethical risks. A high-return black-box allocation
model can encourage automation bias: users may trust the output because the
backtest is strong, even when the explanation says the model depends on a small
set of market blocks. The model may also transfer poorly to regimes where
central-bank interventions dominate historical relationships. Finally, the
strategy may create operational risk through turnover, liquidity constraints,
and unmodelled transaction costs.

Future work should include rolling retraining, transaction-cost-aware training,
crisis-period slicing, explanation stability metrics, and relation-level XAI.
The deletion/insertion result especially motivates interaction-aware
explanations: the model appears to depend on groups of nodes jointly, so future
work should measure node-pair and relation-type effects, not only individual
node scores.

#pagebreak()

#bibliography("references.bib", title: [References])

#pagebreak()

#align(center)[#text(size: 20pt, weight: "bold")[Appendix: Additional XAI Figures]]

#v(10pt)

#figure(
  f("figures/xai/portfolio_signal_shap_mean_signed.svg", width: 95%),
  caption: [Signed SHAP influence on predicted mean allocations. This view is useful for net direction, but cancellations make magnitudes small.]
) <fig-app-shap-mean-signed>

#figure(
  f("figures/xai/portfolio_signal_shap_variance.svg", width: 95%),
  caption: [Absolute SHAP influence on predicted marginal allocation uncertainty.]
) <fig-app-shap-var-abs>

#figure(
  f("figures/xai/portfolio_signal_shap_variance_signed.svg", width: 95%),
  caption: [Signed SHAP influence on predicted marginal allocation uncertainty.]
) <fig-app-shap-var-signed>

#figure(
  f("figures/xai/integrated_gradients_node_importance.svg", width: 82%),
  caption: [Integrated Gradients node-importance histogram.]
) <fig-app-ig-hist>

#figure(
  f("figures/xai/integrated_gradients_last_message_graph.svg", width: 82%),
  caption: [Integrated Gradients node focus when attribution starts before the final message passing layer.]
) <fig-app-ig-last-message>

#figure(
  f("figures/xai/input_attention_pie_charts.svg", width: 92%),
  caption: [Feature-level decomposition inside each observed node using Integrated Gradients.]
) <fig-app-pies>

#figure(
  f("figures/xai/node_temporal_occlusion_windows.svg", width: 95%),
  caption: [Node-wise temporal occlusion: five vertical mini-histograms per node.]
) <fig-app-node-occ>
