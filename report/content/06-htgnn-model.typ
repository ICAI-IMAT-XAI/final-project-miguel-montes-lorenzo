// ============================================================
//  06-htgnn-model.typ
// ============================================================

= Heterogeneous Temporal Graph Neural Network

The HTGNN is the main geometric deep learning model in the project. It models a
fixed heterogeneous graph whose nodes are market blocks and whose edges are typed
according to their economic role. The portfolio signal is one node in this
graph. Other nodes provide contextual information: FX-market nodes, rate-market
nodes, commodity-market nodes, equity-risk nodes, and a small number of learned
generic latent nodes.

#figure(
  image("../figures/graphics/htgnn_architecture.png", width: 60%),
  caption: [Implemented HTGNN architecture. Each market block is encoded as a temporal node, relation-specific message passing is applied over a portfolio-centred heterogeneous graph, and the final allocation is read from the portfolio node.],
) <fig-htgnn-architecture>

== Temporal Node Encoding

For each observed node $v$, the input is a window
$X_(v,t-L+1:t) in RR^(L times F_v)$. Since each market block has a different
number of symbols, each node has its own GRU input dimension but all nodes are
mapped into the same hidden size $H$:

$ bold(h)_v^0 = op("GRU")_v(X_(v,t-L+1:t)) in RR^H. $ <eq-gru-node>

This temporal encoder compresses the recent behaviour of a whole market block
into a single state. For example, the `commodity_future` node receives the
returns of commodity futures over the lookback window and maps them into one
hidden state; the `us_bond_etf` node does the same for the US fixed-income ETF
block.

The implementation also supports learned generic latent nodes. These nodes do
not correspond to observed Yahoo symbols. They are trainable embeddings that can
act as global latent factors or information-routing nodes. In the selected HTGNN
configuration, three generic latent nodes are included.

== Heterogeneous Message Passing

After temporal encoding, the model applies relation-typed message passing. For a
target node $v$, incoming messages are transformed according to their relation
$r$:

$ bold(m)_v^ell = 1/(|cal(N)(v)|) sum_((u,v,r) in cal(E)) W_r^ell bold(h)_u^ell. $

The node update combines the previous node state with the aggregated message and
uses a residual layer-normalised update:

$ bold(h)_v^(ell+1) = op("LayerNorm")(
  bold(h)_v^ell + op("MLP")_ell([bold(h)_v^ell ; bold(m)_v^ell])
). $ <eq-typed-message-passing>

The important detail is that $W_r$ depends on the edge type. A message from FX
pairs to the portfolio node is not forced to use the same transformation as a
message from bond ETFs or commodities. The relation names in the implementation
include `fx_to_portfolio`, `rates_to_portfolio`, `commodity_to_portfolio`,
`equity_to_portfolio`, `portfolio_to_market`, `market_cross`, and analogous
relations involving generic latent nodes.

== Portfolio Readout

Only the final `portfolio_signal` state is used for prediction. After $K$ message
passing layers,

$ bold(z)_t = rho(bold(h)_("portfolio")^K), quad
bold(alpha)_t = op("softplus")(W_alpha bold(z)_t + bold(b)_alpha) + epsilon, $

$ hat(bold(w))_t = bold(alpha)_t / (bold(1)^T bold(alpha)_t), quad
hat(sigma_t^2) = op("softplus")(W_sigma bold(z)_t + b_sigma) + epsilon. $ <eq-readout>

The readout therefore predicts both the allocation and a positive variance
estimate. The selected configuration uses hidden dimension 96, one GRU layer,
two message-passing layers, GELU activations, dropout 0.10, batch size 64, and a
sliced Wasserstein allocation loss.

== Difference from a Simple GNN: Heterogeneity

A simple GNN usually assumes a homogeneous graph: every node has the same feature
semantics and every edge applies the same message function. This is not well
matched to the present data. A commodity-futures node, a bond-ETF node, and an
FX-pair node do not represent the same kind of information. Their relation to
the portfolio signal should be transformed differently.

The HTGNN uses heterogeneity in two places. First, each node has a different
input feature dimension because each market block contains a different number of
Yahoo Finance symbols. Second, edges have relation types, and each relation type
has its own linear message transform. This is the part that makes the graph
model economically interpretable: rates-to-portfolio, FX-to-portfolio,
commodity-to-portfolio, and equity-to-portfolio are different learned channels.

== Difference from a Simple GNN: Temporality

A simple static GNN would consume one graph snapshot at one date. The model here
is temporal because each node contains a 20-day sequence. The GRU encoders allow
recent path information to affect the node state before graph aggregation. This
matters in financial data because the meaning of today's return often depends on
recent trend, reversal, volatility, and co-movement patterns.

The model is not a continuous-time event graph and does not update memory after
every trade. It is a daily-window temporal graph model. This is appropriate for
the available data frequency and for the portfolio horizon considered in the
repository.

== How the HTGNN Uses the Project-Specific Structure

The implemented graph is portfolio-centred. Directed edges are created when the
source or target is the `portfolio_signal` node, or when a generic latent node is
involved. This design avoids an unnecessarily dense fully connected market graph
while still allowing every contextual block to influence the final portfolio
state. Market-to-market interactions can also occur through generic latent nodes
and through relation types such as `market_cross` when allowed by the edge
builder.

This structure matches the financial interpretation of the task. The model does
not need to predict every market block; it needs to transform those blocks into
a currency allocation. The portfolio node therefore acts as the central readout
node, while other nodes act as typed sources of macro-financial context.
