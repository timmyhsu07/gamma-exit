# gamma-exit

[![ci](https://github.com/timmyhsu07/gamma-exit/actions/workflows/ci.yml/badge.svg)](https://github.com/timmyhsu07/gamma-exit/actions/workflows/ci.yml)

When should you close a delta-hedged long option position?

A delta-hedged long option is a bet that realized volatility beats the implied
volatility you paid. You make money on gamma when the underlying moves, and you
pay theta every day it doesn't. That sets up a stopping problem: at some point
the convexity you have left stops covering the rent.

This is a research backtester I wrote to study that trade-off. It started from
Ramkumar (2025), "The Gamma Scalping-Theta Decay Trade-Off as a Basis for
American Option Valuation and Optimal Exercise Timing", a paper I ended up
disagreeing with in places. The engineering goal was a pipeline where I can
defend every number: no look-ahead, math checked against a known identity,
transaction costs on everything, and one command to reproduce any result.

Two places where I don't follow the paper:

1. Its "optimal stopping" strategy uses the future. It exits at the ex-post
   argmax of cumulative P&L, which makes it an oracle. That's a useful upper
   bound but it isn't a strategy, so here it lives in its own module, can't be
   plugged into the policy interface, and every tradable rule is scored as a
   fraction of the oracle's edge instead.
2. Its derivation carries a `−(μ−r)S²Γ` drift term that I don't think survives
   self-financing accounting. This engine's P&L is drift-invariant when I test
   it (μ = r + 15%, thousands of paths), and the surfaces below show the same
   thing: P&L moves along the vol-gap axis and not the drift axis.

## Results so far (synthetic worlds; real chains next)

![Mean hedged P&L over (drift spread x vol spread)](docs/figures/fig1_surfaces.png)

Every cell above is the validated engine run on GBM paths, not a plot of a
formula. If the paper's drift term were real, the color would tilt along the
vertical axis. It doesn't.

![Exit-rule cross-section](docs/figures/fig3_summary.png)

The headline numbers (`python scripts/headline_numbers.py` reproduces both,
seeded from the config):

- Perfect-foresight exits improve mean P&L over hold-to-expiry by +55% across
  300 positions in five regimes. Net of full bid-ask costs it's +66%, because
  costs hurt holding more than they hurt exiting early. So the oracle ceiling
  is real, but it's modest.
- In constant-vol-gap worlds, causal exit rules should lose, and they do. If
  σ_real > σ_IV the whole time, every extra day of holding has positive
  expected P&L, so any early exit gives up edge (capture ≤ 0). The oracle still
  gains there, but that gain is pure ex-post path selection. This is the
  paper's central claim falling apart once you require causality.
- Where the edge decays mid-life, causal rules do work. In regime-shift
  scenarios, where realized vol collapses halfway through, a simple
  EWMA-forecast exit rule captures 72% of the oracle's edge (90% cluster
  bootstrap CI: 66-77%), net of full transaction costs.

The reframing I'd defend: exit timing isn't free money that falls out of
convexity bookkeeping. It's a bet on a volatility regime changing, and it
should be measured as one.

## Quickstart

```bash
# install (needs uv; creates ./.venv from the lockfile)
uv sync --dev
source .venv/bin/activate

# check the math before trusting anything (~20 s)
pytest

# see the M1 gate directly: discrete hedging converging to the identity
python -m gamma_exit.validation.harness

# full backtest: 5 regimes x 10 pre-registered policies x 3 cost levels
python -m gamma_exit.backtest.runner --positions-per-scenario 30

# regenerate the figures above
python -m gamma_exit.analytics.figures

# reproduce the headline numbers quoted in this README
python scripts/headline_numbers.py

# live data path (needs network; run during US market hours)
python -m gamma_exit.data.snapshot
```

Everything takes `--config` (default `configs/baseline.yaml`). The YAML holds
seeds, rates, universe, quote filters, cost levels, and the pre-registered
policy grid. Code carries no copies of those numbers, and every results file is
written alongside its config and git commit.

## How it's validated

I didn't trust the engine until it passed three gates. All three run in CI.

Gate 1, the identity. On synthetic GBM paths, discrete delta-hedge P&L has to
converge pathwise to the closed form

```
X_T = ∫ e^{r(T−u)} · ½ Γ S² (σ_real² − σ_IV²) du
```

as rehedge frequency rises (RMS residual ~ 1/√n), with no drift bias, for
calls, puts, and dividend-paying underlyings:

![Identity convergence](docs/figures/pnl_identity_convergence.png)

Gate 2, equivalence. The real-data replay pipeline (quote loader, per-day IV
solve, daily delta-hedge, shared accounting core) is fed a synthetic option
chain in canonical vendor format, and has to reproduce the synthetic engine's
P&L path to IV-solver tolerance. Real-data results then inherit Gate 1, and the
only new thing you have to trust is the quotes.

Gate 3, system invariants. In a fair world (μ = r, σ_real = σ_IV) no causal
policy may show significant P&L, while the oracle has to profit from noise,
which is the reason it's quarantined. In a world built so the vol edge dies
mid-life, the causal vol watcher has to beat holding. That pins the pipeline
from both sides. One of these probes false-alarmed at n=24 while I was
building it. I chased it down: optional stopping holds and the replication at
n=100 gives t = −0.02. The investigation is written up in the test itself.

Pricing and Greeks are checked against QuantLib to 1e-8 or tighter, and the
whole suite (172 tests) runs from a clean `uv sync` on CI.

## Rules the code actually enforces

| Rule | Where |
|---|---|
| Hedge P&L must reconcile with the ½ΓS²(σr²−σIV²) identity | `pnl/engine.py`, `tests/test_pnl_identity.py` |
| No drift bias (the paper's μ-term rejected empirically) | `tests/test_pnl_identity.py::TestNoDriftBias` |
| Causality by construction: policies get a frozen per-day state, never a frame | `strategy/base.py`, `tests/test_no_lookahead.py` |
| The oracle is not a strategy: separate module, incompatible interface | `strategy/oracle.py` |
| Realized vol (ex-post) vs forecast vol (causal) never crossed | `vol/realized.py` vs `vol/forecast.py` |
| Every backtest runs at zero / half / full spread; all three reported | `configs/baseline.yaml`, `tests/test_cost_model.py` |
| Mid-or-drop: nothing trades on a one-sided or stale quote | `data/schema.py`, `pnl/replay.py` |
| Raw pulls are immutable; bad ones get quarantined, never deleted | `data/cache.py` |
| One time basis: trading-day years, converted only at the data boundary | `conventions.py` |
| Policy parameters pre-registered in config, never tuned on results | `configs/baseline.yaml` |

## Layout

```
src/gamma_exit/
├── conventions.py  the ONE time basis: trading-day years, 252/yr
├── config.py       typed loader for configs/*.yaml (single source of truth)
├── plotstyle.py    shared chart tokens + fixed policy colors
├── pricing/        BS price, IV solver, Greeks (QuantLib-validated)
├── pnl/            the self-financing accounting core + synthetic & replay adapters
├── vol/            realized (EX-POST ONLY) vs forecast (CAUSAL ONLY)
├── data/           canonical schema, write-once cache, yfinance provider
├── strategy/       PositionState/ExitPolicy, benchmarks, causal rules, ORACLE (quarantined)
├── backtest/       synthetic scenario source + walk-forward runner
├── analytics/      capture-fraction metrics + bootstrap CIs, regimes, figures
└── validation/     the Gate-1 reconciliation harness
tests/              172 tests incl. the three gates and system invariants
scripts/            daily snapshot job, headline-number reproduction
docs/               design notes + the figures embedded above
```

## Limitations

- Synthetic worlds only, so far. yfinance has no historical option chains, so
  the real-data replay is validated but waiting on a historical source
  (OptionMetrics via WRDS, or ThetaData; see `TODOS.md`). The daily snapshot
  job is already accumulating a free validation slice.
- Same-close execution. Decisions read the close and trade the close. That's
  optimistic, but every policy and the oracle share it equally, so capture
  fractions stay internally fair.
- Stale days are marked to model (BS at last solved IV, flagged). Nothing
  trades on them, and a persistently large attribution `residual` on real data
  should be read as a data-quality alarm rather than ignored.
- Interest accrues on the trading clock. That's a deliberate approximation
  worth ~1.5% of r on the discount exponent, which is sub-bp in price at
  r ≤ 5%. `conventions.py` documents when to revisit it.
- Expiry is cash settlement at that day's spot. Real SPX monthlies are
  AM-settled, so prefer PM-settled weeklies once the real data arrives.
- Per-share units (one option on one share). Contract multipliers and sizing
  belong to a portfolio layer, which doesn't exist yet.
- The IV solver is a plain bracketed Brent. Transparent but slow. If full-chain
  recomputation ever becomes the bottleneck it gets swapped for Jäckel's "Let's
  Be Rational" (`TODOS.md`).

## References

- Ramkumar, D. (2025). *The Gamma Scalping-Theta Decay Trade-Off as a Basis for
  American Option Valuation and Optimal Exercise Timing.* The framework under
  test here; `docs/DESIGN.md` covers what this project keeps, corrects, and
  reframes.
- El Karoui, N., Jeanblanc-Picqué, M., & Shreve, S. (1998). *Robustness of the
  Black and Scholes formula.* The hedging-at-the-wrong-vol identity the whole
  engine is validated against.

MIT license. If you spot an accounting error, please open an issue.
