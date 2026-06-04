// ============================================================
//  03-signal-structuring.typ
// ============================================================

= Signal Structuring: Learning Allocations Instead of Returns

The main methodological choice in the repository is to avoid treating the task
as direct next-return regression. Daily financial returns have low signal-to-noise
ratios, and a model trained to predict each return independently can learn
unstable point forecasts that are not aligned with the final decision. The final
use of the model is not to report tomorrow's EUR or JPY return; it is to decide
how much capital should be allocated to each currency. For this reason, the
training signal is constructed in allocation space.

This design has three practical advantages. First, the output is constrained to
the long-only simplex, which prevents nonsensical unconstrained predictions.
Second, the target incorporates risk through a covariance matrix, not only
through expected return. Third, the supervised target is closer to the downstream
objective: a portfolio allocation. The model is therefore trained to imitate the
kind of decision a portfolio optimiser would make, rather than to solve a harder
and more weakly identified forecasting problem as an intermediate step.

== Mean-Variance Teacher

Let $bold(r)_t in RR^N$ be the USD-denominated log-return vector of the eight
currencies. At date $t$, the recent return window is
$R_t = [bold(r)_(t-L+1), ..., bold(r)_t]$. The covariance estimator used by the
training loss is

$ hat(Sigma)_t = 1/(L-1) sum_(i=t-L+1)^t
  (bold(r)_i - bar(bold(r))_t)(bold(r)_i - bar(bold(r))_t)^T + epsilon bold(I), $

where $epsilon$ is the ridge term. The target mean proxy is the realised next
cross-section centred around its own average:

$ tilde(bold(mu))_(t+1) = bold(r)_(t+1) - 1/N bold(1) bold(1)^T bold(r)_(t+1). $

The ex-post teacher allocation is then obtained from a long-only mean-variance
problem:

$ bold(w)_(t+1)^star = op("argmax")_(bold(w) in Delta_N)
  bold(w)^T tilde(bold(mu))_(t+1)
  - gamma / 2 bold(w)^T hat(Sigma)_t bold(w). $ <eq-mv-teacher>

The unconstrained solution is proportional to
$(gamma hat(Sigma)_t)^(-1) tilde(bold(mu))_(t+1)$. Since the project uses
long-only portfolios, this raw solution is projected onto the simplex:

$ bold(w)_(t+1)^star = Pi_(Delta_N) ((gamma hat(Sigma)_t)^dagger
  tilde(bold(mu))_(t+1)). $ <eq-simplex-projection>

This does not imply that the model has access to $bold(r)_(t+1)$ at inference
time. The next return is used only to build the supervised teacher during
training. At inference, the trained model receives only past windows and graph
node features and outputs an allocation estimate.

== Input Signal as Rolling Allocation History

The pointwise models can receive raw currency-return windows, but the selected
configuration uses `input_format: "weights"`. In that mode, the model converts
each prefix of the 20-day return window into a rolling mean-variance weight
vector. The input becomes a sequence of previous allocation regimes rather than
a raw return matrix:

$ X_t^"portfolio" = [bold(w)_(t-L+1)^"in", ..., bold(w)_t^"in"], quad
bold(w)_s^"in" = Pi_(Delta_N)((gamma hat(Sigma)_(s-1))^dagger
  tilde(bold(mu))_s). $ <eq-input-weights>

The first step has insufficient history and is set to equal weights. This input
representation is less sensitive to raw return scale and more directly aligned
with the output. It also makes the comparison between the pointwise NN and
HTGNN conceptually clear: the pointwise baseline sees only the allocation signal,
whereas the HTGNN sees that same signal plus heterogeneous contextual nodes.

== Dirichlet Allocation Head

The implemented models output Dirichlet concentration parameters rather than
unconstrained weights. If $bold(a)_t$ denotes the raw output logits, the
concentrations and predicted allocation are

$ bold(alpha)_t = op("softplus")(bold(a)_t) + epsilon, quad
hat(bold(w))_t = bold(alpha)_t / (bold(1)^T bold(alpha)_t). $ <eq-dirichlet-head>

This is useful even for deterministic models, because the mean of a Dirichlet
naturally lies on the simplex. For stochastic models, the same parameterisation
also provides a distribution over allocations. The training loss selected in the
configuration is a sliced Wasserstein loss between samples from the predicted
Dirichlet and samples from a target Dirichlet centred at the teacher allocation:

$ cal(L)_"SW" = 1/M sum_(m=1)^M W_1(
  {theta_m^T bold(u)_i}_(i=1)^S,
  {theta_m^T bold(v)_i}_(i=1)^S
), $

where $bold(u)_i ~ op("Dir")(bold(alpha)_t)$, $bold(v)_i ~ op("Dir")(bold(alpha)_t^star)$,
and $theta_m$ are random one-dimensional projection directions. This objective
compares distributions in allocation space rather than applying a simple MSE to
weights.

== Why This is Preferable to Direct Return Prediction

A direct-return model would learn a map
$f_theta(X_t) -> hat(bold(r))_(t+1)$ and then require a separate optimiser to
transform predicted returns into weights. This creates a mismatch: small errors
in noisy return forecasts can be amplified by the inverse covariance matrix and
produce unstable allocations. In contrast, the implemented pipeline learns the
allocation mapping directly. The model is penalised for placing capital in the
wrong part of the simplex, which is closer to the final portfolio objective.

This is especially relevant for Forex. The differences between daily currency
returns are often small, and the sign of tomorrow's return is difficult to
predict. However, allocation labels can still encode useful relative information:
which currency was favoured after adjusting for recent covariance and after
centering the realised cross-section. The model therefore learns a smoothed
portfolio decision boundary rather than a set of isolated scalar forecasts.
