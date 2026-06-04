// ============================================================
//  04-allocation-strategies.typ
// ============================================================

= Asset Allocation Strategies

The repository separates model prediction from portfolio rebalancing. A model
outputs a target allocation $hat(bold(w))_t$. A strategy then decides which
portfolio is actually held. Three strategies are implemented.

*Full rebalancing* applies the model target directly:

$ bold(w)_t^"hold" = hat(bold(w))_t. $

This is the cleanest strategy for comparing model outputs, but it can imply high
turnover because the portfolio fully adapts to every daily prediction.

*Partial rebalancing* moves only a fraction $alpha$ of the way from the current
portfolio to the model target:

$ bold(w)_t^"hold" = op("normalize")(
  bold(w)_(t-1)^"hold" + alpha (hat(bold(w))_t - bold(w)_(t-1)^"hold")
). $

The default configuration uses $alpha = 0.2$. This is more conservative and can
reduce turnover, but it also delays adaptation when the model detects a new
regime.

*Black-Litterman rebalancing* is available for probabilistic models. It treats
the model output as a view and uses model uncertainty to scale the view-noise
matrix. With an equilibrium prior $bold(Pi) = delta Sigma bold(w)^"eq"$, view
matrix $P = I$, view vector $bold(q)=hat(bold(w))_t$, and uncertainty matrix
$Omega$, the posterior mean is

$ bold(mu)_"BL" = [(tau Sigma)^(-1) + P^T Omega^(-1) P]^(-1)
  [(tau Sigma)^(-1) bold(Pi) + P^T Omega^(-1) bold(q)]. $ <eq-bl>

The final strategy then applies a long-only mean-variance step to
$bold(mu)_"BL"$ and the posterior-adjusted covariance. This strategy is not the
central comparison for the deterministic HTGNN, but it is useful as an extension
point for future uncertainty-aware graph models.
