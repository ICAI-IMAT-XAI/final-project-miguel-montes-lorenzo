// ============================================================
//  05-baselines.typ
// ============================================================

= Comparative Baseline

One pointwise neural baseline is implemented for comparison with the HTGNN: a
deterministic neural network (NN). It receives only the portfolio signal. In the
selected configuration, that signal is the sequence of rolling mean-variance
allocations derived from the past currency return window. It does not receive the
additional heterogeneous market blocks.

== Deterministic NN

The NN baseline is a multilayer perceptron. It flattens the 20-day portfolio
signal and maps it through BatchNorm, GELU activations, dropout, and linear
layers:

$ bold(z)_t = phi_theta(op("vec")(X_t^"portfolio")), quad
bold(alpha)_t = op("softplus")(W_alpha bold(z)_t + bold(b)_alpha) + epsilon. $

The predicted allocation is the Dirichlet mean $hat(bold(w))_t =
bold(alpha)_t / (bold(1)^T bold(alpha)_t)$. This baseline tests how much can be
learned from the allocation signal alone, without graph structure.

== Fairness of the Comparison

The comparison is not fully fair if interpreted only as a model-architecture
comparison, because the HTGNN receives more information than the NN. The
pointwise baseline sees only the portfolio signal, while the graph model also
receives commodities, futures, bond ETFs, FX pairs, and currency-region bond
proxies.

However, the comparison remains informative from a practical financial-prediction
perspective. Many signal models are trained only on the historical signals they
later predict or rank. Under that convention, the pointwise model represents a
realistic signal-only baseline. The HTGNN then answers a more specific question:
does adding structured heterogeneous market context improve the allocation
pipeline beyond a signal-only neural model?
