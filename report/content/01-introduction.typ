// ============================================================
//  01-introduction.typ
// ============================================================

= Introduction and Research Question

Foreign exchange (Forex or FX) is inherently relative: each currency is priced
against another numeraire. This project fixes USD as numeraire and allocates
across $cal(C) = {"USD", "EUR", "JPY", "GBP", "CNY", "CAD", "AUD", "CHF"}$.
The prediction target is therefore not a price, but a long-only portfolio
$bold(w)_t in Delta_N$, with
$Delta_N = {bold(w) in RR_+^N : bold(1)^T bold(w) = 1}$.

The research question is: *can a heterogeneous temporal graph neural network use
macro-financial market blocks to improve currency allocation over signal-only
neural baselines?* The hypothesis is that exchange rates are partly shaped by
structured relations among currency returns, rates, commodities, equity risk,
and FX futures. Encoding these sources as typed nodes and relations should fit
the problem better than a pointwise MLP that only observes past portfolio
signals.

== Why Forex is an Interesting Prediction Domain

Forex is attractive for this project because currencies are macro assets. Their
returns are linked to monetary policy, inflation, external balances, commodity
exposure, and global risk appetite, which can be organised into meaningful market
blocks. The task is also naturally cross-sectional: the model allocates among
currencies under a common numeraire rather than forecasting isolated price
levels.

The target universe is compact and interpretable. With eight currencies, one can
inspect whether the model shifts capital toward plausible regimes, such as USD
and CHF under stress or AUD and CAD in commodity-sensitive environments. The
main limitation is that currencies lack the long-run growth story of equities:
the problem is mostly relative value, exposed to inflation, policy expectations,
central-bank intervention, and geopolitical shocks.

== Why an HTGNN is Appropriate

A standard temporal model treats inputs as one homogeneous sequence, but the data
here are grouped by economic role: FX pairs, FX futures, Treasury futures, bond
ETFs, commodities, and equity-index futures. A heterogeneous graph preserves
that structure while allowing information to pass through learned relation types.

Temporality is also necessary because allocation depends on recent paths. In the
implemented HTGNN, each market block receives a return window, is encoded by a
GRU, and then participates in typed message passing centred on the
`portfolio_signal` node. The final allocation is read from that node, matching
the intended design: market blocks provide context, and the portfolio node
produces the currency weights.

== Related Work and Scope

The project belongs to the heterogeneous and temporal GNN setting. Relational
GNNs introduce relation-specific transformations for typed edges, making them a
natural reference for heterogeneous financial graphs @schlichtkrull2018modeling.
Temporal graph networks motivate the idea that node representations should be
updated with time-dependent information rather than treated as static embeddings
@rossi2020temporal. The implemented model is simpler than a full continuous-time
TGN, but it follows the same high-level idea: temporal node states are built
before graph aggregation.

The allocation target is based on mean-variance portfolio theory, where a
portfolio is chosen by trading expected return against variance
@markowitz1952portfolio. The repository also includes a Black-Litterman-style
allocation strategy that can blend prior equilibrium returns with model views
@black1992global, but the main report focuses on the deterministic signal and
HTGNN allocation pipeline.
