# gamma-exit

Research backtester for the **gamma-scalping / theta-decay trade-off**: when
should a delta-hedged long option position be *closed* (liquidated), and how
much of the ex-post-optimal exit's edge can *causal* (non-anticipating) rules
capture on real data, net of transaction costs?

Based on (and correcting) Ramkumar (2025), *"The Gamma Scalping–Theta Decay
Trade-Off..."*. Full project spec: `PROJECT_BRIEF.md`. Engine of everything:

```
dPnL ≈ ½ · Γ · S² · (σ_realized² − σ_implied²) · dt
```

## Status

- **Milestone 1 — DONE**: pricing validated vs QuantLib; synthetic delta-hedge
  engine reconciles pathwise against the closed-form identity; drift-invariance
  verified empirically (the paper's `−(μ−r)S²Γ` term does not survive
  self-financing accounting).
- **Milestone 2 — prototyped**: canonical schema, write-once Parquet/DuckDB
  cache, yfinance provider (live chain snapshots + underlying bars + IV
  recomputation from quotes). ThetaData / OptionMetrics readers pending.
- Milestones 3–6 (real-data replay, policies + quarantined oracle, walk-forward
  backtest, paper-figure reproduction): not started.

## Setup

```bash
uv sync --dev            # creates ./.venv and installs everything
source .venv/bin/activate
```

## Run things

```bash
pytest                                        # full validation suite (~2 s)
python -m gamma_exit.validation.harness       # identity-convergence table + plot
python -m gamma_exit.data.snapshot SPY --max-expiries 3   # live yfinance demo
```

The harness writes `reports/pnl_identity_convergence.png`. The snapshot demo
writes immutable Parquet under `cache/` (second run on the same key refuses to
overwrite — that is the point).

## Hard rules enforced in code and tests

| Rule | Where |
|---|---|
| Discrete-hedge P&L must reconcile with the ½ΓS²(σr²−σIV²) identity | `pnl/engine.py`, `tests/test_pnl_identity.py` |
| No drift bias in hedged P&L (paper's μ-term rejected) | `tests/test_pnl_identity.py::TestNoDriftBias` |
| Greeks match QuantLib to 1e-8 or tighter | `tests/test_greeks_vs_quantlib.py` |
| Realized vol (ex-post) vs forecast vol (causal) never crossed | `vol/realized.py` vs `vol/forecast.py` |
| Transaction costs first-class, monotone | `tests/test_cost_model.py`, `configs/baseline.yaml` |
| Raw data immutable (write-once cache) | `data/cache.py`, `tests/test_cache_and_schema.py` |
| Mid-or-drop quote policy (last-trade only in the labeled demo fallback) | `data/schema.py::OptionRecord.mid` |
| Oracle quarantine (non-tradable ceiling) | `strategy/__init__.py` (Milestone 4) |

## Milestone 1 assumptions & known caveats

1. **Option marked at BS(σ_IV) between entry and expiry.** In synthetic mode the
   "market price" of the option *is* its Black-Scholes value at the constant
   hedge vol. Real quotes have a moving IV; replay mode (Milestone 3) will mark
   at observed mids instead, which adds a vega P&L term the identity does not
   contain. The identity test is exactly as strong as this assumption is
   explicit.
2. **Constant σ_real, σ_IV, r, μ; GBM; no dividends in the synthetic engine.**
   The pricing layer supports a continuous dividend yield `q`, but the
   Milestone 1 reconciliation runs at q=0. Dividends matter for SPY replays and
   enter in Milestone 3.
3. **Hedging on the simulation grid.** Paths are exact GBM at grid points, and
   the hedge rebalances at every grid point, so "rehedge frequency" and "grid
   resolution" are the same knob. The identity integral is a left-endpoint
   Riemann sum on the same grid — near-expiry ATM gamma is large, and the
   left-endpoint sum slightly under-resolves it; this shows up as (small)
   residual noise, not bias, and shrinks with frequency as verified.
4. **Costs are proportional per share hedged** (`cost_per_share` ≈ half-spread).
   No option-side spread in synthetic mode (you buy and hold one option; entry
   spread is a constant that shifts all exits equally — it will matter for
   *exit timing* only through the exit-side spread, added in Milestone 3).
5. **Theta convention**: calendar theta `∂V/∂t` per year, matching QuantLib's
   `theta()`. Per-day theta = annual / 365.
6. **IV solver honesty**: deep-ITM/short-dated quotes are vega-degenerate; the
   solver returns *a* vol that reprices the quote, not "the" vol. The data
   layer's liquidity/moneyness filters exist precisely to avoid feeding those
   to anything downstream.
7. **yfinance quirks measured, not assumed**: after-hours chains carry zeroed
   bid/ask/OI (mid-based filtering then correctly yields zero usable quotes),
   and Yahoo's `impliedVolatility` column is unreliable (stored as
   `provider_iv`, advisory only; we recompute from quotes).
8. **Prototype risk-free rate is a constant** (`configs/baseline.yaml`); a real
   short-rate series arrives with Milestone 2.

## Layout

```
src/gamma_exit/
├── pricing/      BS price, IV solver, Greeks (QuantLib-validated)
├── pnl/          delta-hedged P&L engine (synthetic now; replay in M3)
├── vol/          realized (EX-POST ONLY) vs forecast (CAUSAL ONLY)
├── data/         schema, write-once cache, providers (yfinance now)
├── strategy/     exit policies incl. quarantined oracle (M4)
├── backtest/     walk-forward runner (M5)
├── analytics/    metrics, regimes (M5)
└── validation/   synthetic reconciliation harness (M1 gate)
```
