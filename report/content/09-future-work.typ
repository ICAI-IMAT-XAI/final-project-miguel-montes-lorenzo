// ============================================================
//  09-future-work.typ
// ============================================================

= Future Work

One extension is uncertainty-aware allocation. Financial inputs are
incomplete and non-stationary, so a deterministic HTGNN can produce strong point
allocations but cannot express confidence directly.

A Bayesian HTGNN is therefore worth testing. The repository already contains an
MC-dropout BHTGNN, but it was not validated enough for the main comparison.
Future work should test deterministic HTGNN, MC-dropout BHTGNN, and possibly a
variational graph model under the same splits and trading rules.

The key question is empirical: whether uncertainty estimates improve realized
portfolio quality after turnover and risk are included. In FOREX, observable
macro drivers may already capture much of the relevant structure, so Bayesian
modelling remains an empirical question rather than a guaranteed improvement.
