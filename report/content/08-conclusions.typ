= Conclusions

The results indicate that FOREX allocation benefits from heterogeneous temporal
graph learning. The deterministic HTGNN outperforms the signal-only NN,
suggesting that FX futures, commodity blocks, and fixed-income blocks add useful
predictive context.

The graph model also avoids simply copying temporal persistence: its lag-1
allocation structure remains closer to the weakly correlated input signal than
to the persistent NN outputs. This fits a cross-asset FOREX setting, where useful
information is distributed across markets.

The node-focus analysis adds interpretability. The HTGNN relies most on FX
futures, broad US fixed income, and commodity-linked nodes, while using local
sovereign-bond proxies less. This is consistent with a setting where broad
cross-asset regime signals matter more than local bond proxies alone.
