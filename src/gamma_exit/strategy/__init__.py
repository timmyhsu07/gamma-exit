"""Exit policies (Milestone 4).

Planned modules:
- base.py        ExitPolicy ABC: decide(state_up_to_t) -> HOLD | EXIT
- benchmarks.py  HoldToExpiry, FixedTime
- causal.py      ThetaGammaThreshold, TrailingStop, VolRegime, MLPolicy hook
                 (may consume ONLY vol.forecast and past P&L)
- oracle.py      OracleExitPolicy -- ex-post argmax of cumulative net P&L.
                 NON-CAUSAL, NON-TRADABLE. It is the ceiling causal policies
                 are measured against, never a strategy. Quarantined here.
"""
