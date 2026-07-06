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
- **Milestone 2 — foundation landed** (eng review 2026-07-02, 18 decisions):
  shared mark-agnostic accounting core (`replay_hedged_position` — the M1
  validation now covers the exact loop replay mode will use), single
  trading-day time basis (`conventions.py`), typed config loader wired into
  the harness and snapshot demo, dividend-yield accounting validated, offline
  provider fixtures, CI, daily snapshot automation. ThetaData / OptionMetrics
  readers deferred until data access exists (`TODOS.md` TD-1).
- **Milestone 3 — replay machinery DONE, awaiting real data**: `pnl/replay.py`
  replays a position through observed quotes (loader over cached chains →
  per-day IV from mids → daily delta-hedge → the validated core) with a daily
  attribution series (gamma / theta / vega / carry / cost / net / cum_net).
  Gated by an **equivalence test**: fed a synthetic chain in canonical quote
  format, the full pipeline reproduces the M1 synthetic engine pathwise to
  IV-solver tolerance — so real-data results inherit the M1 validation and
  the only new trust assumption is the quotes themselves. Runs on real
  vendor data the day TD-1 lands (schema drop-in).
- Milestones 4–6 (policies + quarantined oracle, walk-forward backtest,
  paper-figure reproduction): not started.

## Setup

```bash
uv sync --dev            # creates ./.venv and installs everything
source .venv/bin/activate
```

## Run things

```bash
pytest                                        # full validation suite (~5 s)
python -m gamma_exit.validation.harness       # identity-convergence table + plot
python -m gamma_exit.data.snapshot            # live demo, ^SPX from config
```

Everything takes `--config` (default `configs/baseline.yaml`) — the YAML is
the single source of truth for seeds, rates, universe, quote filters, and
cost levels; code carries no copies of those numbers.

To accumulate daily chain snapshots automatically (they become a validation
slice against the paid historical data later), see `scripts/daily_snapshot.sh`
and the launchd template next to it — schedule it mid US market session.

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
| Raw data immutable (write-once cache; bad pulls quarantined, never deleted) | `data/cache.py`, `tests/test_cache_and_schema.py` |
| One time basis: trading-day years (252/yr), converted only at the data boundary | `conventions.py`, `tests/test_conventions_and_config.py` |
| All runs config-driven; YAML typos fail at load time | `config.py`, `tests/test_conventions_and_config.py` |
| Mid-or-drop quote policy (last-trade only in the labeled demo fallback) | `data/schema.py::OptionRecord.mid` |
| Oracle quarantine (non-tradable ceiling) | `strategy/__init__.py` (Milestone 4) |

## Milestone 1 assumptions & known caveats

1. **Option marked at BS(σ_IV) between entry and expiry.** In synthetic mode the
   "market price" of the option *is* its Black-Scholes value at the constant
   hedge vol. Real quotes have a moving IV; replay mode (Milestone 3) will mark
   at observed mids instead, which adds a vega P&L term the identity does not
   contain. The identity test is exactly as strong as this assumption is
   explicit.
2. **Constant σ_real, σ_IV, r, μ; GBM.** Continuous dividend yield `q` is now
   validated end-to-end (the accounting core credits the dividend flow on the
   hedge shares; `TestDividendYield` reconciles it against the identity).
   Discrete cash dividends for single names remain a Milestone 3 concern —
   the primary ^SPX universe sidesteps them entirely.
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
8. **Risk-free rate is a constant from config** (`rates.mode: constant`); a
   real short-rate series is an upgrade path (`rates.mode: curve`), needed
   before multi-year replays.
9. **Interest accrues on the trading clock** — a deliberate, documented
   approximation worth ~1.5% *of r* on the discount exponent (sub-bp price
   impact at r ≤ 5%); see `conventions.py` for when to revisit.

## Milestone 3 assumptions & known caveats

1. **Stale days are marked to model** (BS at the last solved IV, flagged
   `stale`): the position's *value* on those days is a model statement, not a
   market one. Nothing trades on a stale mark (mid-or-drop at entry/exit is
   enforced), and vega P&L is zero on stale days by construction (IV ffill).
2. **The hedge re-balances daily at the model delta**, even on stale-quote
   days — the underlying is liquid regardless of the option's quote quality.
   Delta uses the day's mid-implied IV: a noisy mid moves the hedge ratio.
3. **Attribution is model-based** (Greeks at the previous close); `residual`
   is the exact plug to the true accounting net. On real data it absorbs
   quote noise and higher-order moves — a persistently large residual is a
   data-quality signal, not something to silently ignore.
4. **Expiry exit = cash settlement at intrinsic on the expiry date's spot.**
   Real SPX monthlies are AM-settled (SET print); daily-bar replay ignores
   that PM/AM distinction. Prefer weeklies/PM-settled series or accept the
   settlement-day noise.
5. **One row per trading day.** The loader dedupes multiple pulls to the last
   per US-Eastern date and drops weekend/holiday pulls; intraday timing of
   the snapshot within the day is not modeled.
6. **Per-share units, one option on one share.** Contract multipliers (100x)
   and position sizing are the M5 runner's job.

## Layout

```
src/gamma_exit/
├── conventions.py  THE time basis: trading-day years, 252/yr (decision 3A)
├── config.py       typed loader for configs/*.yaml (single source of truth)
├── pricing/      BS price, IV solver, Greeks (QuantLib-validated)
├── pnl/          shared self-financing accounting core + synthetic adapter
├── vol/          realized (EX-POST ONLY) vs forecast (CAUSAL ONLY)
├── data/         schema, write-once cache + quarantine, providers (yfinance)
├── strategy/     exit policies incl. quarantined oracle (M4)
├── backtest/     walk-forward runner (M5)
├── analytics/    metrics, regimes (M5)
└── validation/   synthetic reconciliation harness (M1 gate)
```
