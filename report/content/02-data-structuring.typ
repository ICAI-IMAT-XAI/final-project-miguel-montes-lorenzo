// ============================================================
//  02-data-structuring.typ
// ============================================================

= Data Structuring

The repository uses daily adjusted Yahoo Finance price series from 2017-01-01 to
2026-01-01. Prices are converted to log returns, short gaps are forward-filled
with a limit of three days, and instruments with excessive missingness are
filtered before tensor construction. The supervised dataset uses a 20-day
lookback window and chronological splits: 2018-01-01 to 2024-01-01 for training,
2024-01-01 to 2025-01-01 for validation, and 2025-01-01 to 2026-01-01 for test.

The target basket contains eight currencies. For every non-USD currency, the
Yahoo Finance quote `USDXXX=X` is interpreted as units of currency `XXX` per one
USD. Therefore the USD-denominated log return of holding one unit of currency
`XXX` has the opposite sign of the quoted pair return:

$ r_(t, "XXX")^"USD" = log (V_t^"XXX" / V_(t-1)^"XXX")
  = - (log P_t^("USD/XXX") - log P_(t-1)^("USD/XXX")), quad
  r_(t,"USD")^"USD" = 0. $ <eq-fx-inversion>

This convention makes all target returns comparable under the same numeraire.
The model output is then an allocation across the eight USD-valued currency
returns.

== Yahoo Finance Market Blocks

The heterogeneous graph does not create one node per individual ticker. Instead,
each node represents a market block. This keeps the graph compact and forces the
model to learn relations between economic categories rather than between a large
set of weakly interpretable individual symbols.

#figure(
  [
    #set text(size: 7pt)
    #table(
      columns: (1.15fr, 0.45fr, 2.1fr, 2.7fr),
      inset: (x: 2pt, y: 2pt),
      stroke: 0.35pt + rgb("#d9dee7"),
      [*Block*], [*N*], [*Examples*], [*General relevance for Forex*],
      [`commodity_future`], [22], [`GC=F`, `SI=F`, `CL=F`, `BZ=F`, `NG=F`, `ZC=F`], [Commodity prices proxy terms-of-trade shocks, inflation pressure, and risk regimes relevant to commodity-linked currencies such as AUD and CAD.],
      [`commodity_etf`], [12], [`GLD`, `IAU`, `SLV`, `USO`, `UNG`, `DBC`, `GSG`], [ETF versions provide tradable commodity exposure and can be less sparse than some futures histories.],
      [`equity_index_future`], [4], [`ES=F`, `NQ=F`, `YM=F`, `RTY=F`], [Equity-index futures proxy global risk appetite, which affects safe-haven and cyclical currency flows.],
      [`us_treasury_future`], [4], [`ZB=F`, `ZN=F`, `ZF=F`, `ZT=F`], [Treasury futures summarize changes in the US rates curve, a central driver of USD strength and carry conditions.],
      [`fx_future`], [6], [`6E=F`, `6B=F`, `6J=F`, `6S=F`, `6A=F`, `6C=F`], [Currency futures provide alternative FX-market information for liquid developed-market currencies.],
      [`us_bond_etf`], [20], [`SGOV`, `SHV`, `SHY`, `IEF`, `TLT`, `AGG`, `LQD`, `HYG`, `TIP`], [US bond ETFs capture duration, credit, inflation-protected, and risk-on/risk-off fixed-income regimes.],
      [`fx_usd_pair`], [7], [`USDEUR=X`, `USDJPY=X`, `USDGBP=X`, `USDCNY=X`, `USDCAD=X`, `USDAUD=X`, `USDCHF=X`], [The direct FX quotes used to build the target currency-return basket.],
      [`eur_bond_etf`], [4], [`CBE3.L`, `SDEU.L`, `SYBV.DE`, `020Y.L`], [Euro-area government-bond proxies reflect regional rate and duration expectations affecting EUR.],
      [`jpy_bond_etf`], [3], [`236A.T`, `CEB2.DE`, `2510.T`], [Japan bond proxies provide information about JPY rate conditions and safe-haven behaviour.],
      [`gbp_bond_etf`], [9], [`IGLT.L`, `IGLS.L`, `INXG.L`, `VGOV.L`, `GLTY.L`, `GLTL.L`], [UK gilt proxies encode sterling-specific rate and inflation-linked bond information.],
      [`cny_bond_etf`], [13], [`2829.HK`, `CNYB.AS`, `CGB.L`, `ICGB.DE`, `CBGB.L`], [China bond proxies provide context for CNY and emerging-Asia rate conditions.],
      [`cad_bond_etf`], [5], [`XGB.TO`, `CLF.TO`, `CLG.TO`, `ZGB.TO`, `VGV.TO`], [Canadian government-bond proxies support CAD allocation through local-rate information.],
      [`aud_bond_etf`], [8], [`IGB.AX`, `VGB.AX`, `AGVT.AX`, `ALTB.AX`, `XGOV.AX`], [Australian bond proxies support AUD allocation and commodity-sensitive macro regimes.],
      [`chf_bond_etf`], [7], [`CSBGC3.SW`, `CSBGC7.SW`, `CSBGC0.SW`, `SB1CHA.SW`, `SB7CHA.SW`], [Swiss bond proxies give CHF-specific duration and safe-haven context.],
    )
  ],
  caption: [Yahoo Finance symbols grouped into heterogeneous market blocks. The number column is the configured number of tickers before transform-time missing-data filtering.],
) <tab-market-blocks>

== Tensor Construction

After filtering and alignment, every sample contains a lookback window ending at
$t$ and a next-day return vector at $t+1$. The pointwise signal is the currency
return window, later optionally converted inside the model into a sequence of
mean-variance allocation features. The graph input is a dictionary of node
windows:

$ X_t = {X_(v,t-L+1:t) : v in cal(V)}, quad
X_(v,t-L+1:t) in RR^(L times F_v), quad L = 20. $ <eq-node-window>

The metadata stores the list of node names, the input dimension of each node,
the target currencies, the date splits, and the train-only standardisation
statistics. Standardisation is fitted on the training split and then applied to
all splits, avoiding validation or test leakage.
