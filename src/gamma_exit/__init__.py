"""gamma_exit: research backtester for the gamma-scalping / theta-decay trade-off.

Layer map (see PROJECT_BRIEF.md):
- pricing/    Black-Scholes price, implied vol, Greeks (validated vs QuantLib)
- pnl/        delta-hedged P&L engine (synthetic + replay modes)
- vol/        realized (EX-POST ONLY) and forecast (CAUSAL ONLY) volatility
- data/       canonical schema, providers, write-once cache
- strategy/   exit policies; oracle.py is quarantined and NON-TRADABLE
- backtest/   orchestration (walk-forward, no-look-ahead)
- analytics/  metrics and regime tagging
- validation/ synthetic reconciliation harness (Milestone 1 gate)
"""

__version__ = "0.1.0"
