# Design notes

What this project keeps from the source paper, what it corrects, and the
architecture decisions that keep the research honest. Written for future-me
and for anyone auditing a result.

## The theory, and where I stand on it

The engine of everything is the gamma–vega identity. Hedge a long option at
implied vol σ_IV while the world realizes σ_real, keep the book
self-financing, and the P&L is — pathwise, not just in expectation —

```
X_T = ∫₀ᵀ e^{r(T−u)} · ½ Γ_IV(u, S_u) · S_u² · (σ_real² − σ_IV²) du
```

(El Karoui–Jeanblanc–Shreve 1998 territory.) Positive gamma plus realized
above implied means rehedging prints money; theta is the rent you pay for
the privilege. Ramkumar (2025) builds an "optimal exercise" story on top of
this trade-off. Two parts of that paper don't survive scrutiny:

1. **The drift term.** The paper's excess-return expression includes
   `−(μ−r)S²Γ`, coming from a `−S·dΔ` "rebalancing cost" that isn't how a
   self-financing account works. In the standard accounting the drift washes
   out. I treated this as a hypothesis rather than an error claim and tested
   it: at μ = r + 15% with zero vol gap, mean hedged P&L is statistically
   zero while the paper's term predicts a large bias
   (`tests/test_pnl_identity.py::TestNoDriftBias`). The term is refuted;
   Figure 1 shows it visually.

2. **"Optimal stopping" is an oracle.** Exiting at the ex-post argmax of
   cumulative P&L requires the whole future path. The paper itself admits
   this in passing, then reports it as the winning strategy anyway. Here it
   is only ever a ceiling, and the research question becomes: *what fraction
   of that ceiling can rules with no future information capture, net of
   costs?* That's the capture-fraction metric everything reports.

Also renamed throughout: this is **liquidation timing for a volatility
trade**, not option exercise. Early exercise of an American call on a
non-dividend underlying is never optimal; closing a hedged position is a
different (and actually interesting) decision.

## Architecture decisions

**One accounting core.** `pnl/engine.py::replay_hedged_position` is the only
loop that turns positions into P&L. It's mark-agnostic: synthetic mode feeds
it Black–Scholes marks, replay mode feeds it observed mids. The validation
gates run against the exact code path that produces real-data results —
there is no "test engine" and "real engine" pair to drift apart.

**One time basis.** Trading-day years, 252/yr, everywhere
(`conventions.py`). Realized variance accrues when markets trade; weekends
add theta cost but no gamma income, which is precisely the trade-off under
study. Market expiries are calendar dates, so conversion happens once, at
the data boundary, with business-day counting. Mixing 252-basis vols with
365-basis dt would misattribute gamma vs theta by up to 45% on the variance
scale — the kind of bug that never crashes and quietly poisons every result.
Interest technically accrues on calendar time; keeping r on the trading
clock costs ~1.5% *of r* on the discount exponent, which I accepted and
documented rather than carrying two clocks through every function.

**Causality by construction, not by discipline.** Exit policies don't
receive data frames. The runner builds one frozen `PositionState` per day —
past P&L, today's Greeks, the causal forecast — and a policy is a pure
function of it. Look-ahead isn't a bug a test catches; the future simply
isn't in scope. One truncation-invariance test verifies the constructor, and
every future policy inherits the guarantee for free.

**The oracle is quarantined.** `strategy/oracle.py` consumes a full
exit-value array — an input type the runner never hands to policies — is not
an `ExitPolicy`, is not constructible from the registry, and its results are
labeled `oracle` everywhere. Tests assert all of that, plus pathwise
dominance (a ceiling that isn't a ceiling would mean broken exit
accounting).

**Two vol functions, never crossed.** `vol/realized.py` is ex-post truth for
the oracle and attribution; `vol/forecast.py` (EWMA) is the only volatility
a causal policy may consume. The EWMA's prefix-invariance — appending future
rows cannot change the value at t — is itself a test.

**Pre-registration.** Entry protocols and the exit-policy parameter grid
live in `configs/baseline.yaml` and were fixed before any real-data run. The
whole grid gets reported. This exists because the alternative — tuning
thresholds after seeing results — is how backtests lie to their authors.

**Immutable raw data.** Provider pulls are written once to Parquet; a second
write to the same key refuses. Bad pulls get moved to a quarantine directory
with a reason file — history is never silently rewritten. DuckDB queries the
cache in place.

**Costs are first-class.** Every backtest runs at zero, half, and full
spread. Option legs pay a fraction of the *observed* half-spread; hedge legs
pay per share. A result that only survives at zero cost gets reported as
exactly that.

## Validation gates (all in CI)

1. **Identity gate** — discrete hedge P&L converges pathwise to the identity
   (RMS ~ 1/√n in rehedge frequency), drift-invariant, calls/puts/dividends.
2. **Equivalence gate** — the full replay pipeline on a synthetic chain in
   canonical vendor format reproduces the synthetic engine to solver
   tolerance, so replay inherits gate 1.
3. **Invariant gate** — martingale world: no causal rule may show edge, the
   oracle must profit from noise. Decaying-edge world: the causal vol
   watcher must beat holding. Plus determinism, oracle dominance, and
   degenerate-parameter identities (FixedTime(1.0) ≡ hold).

Greeks reconcile against QuantLib on a parameter grid; the IV solver
round-trips including vega-degenerate corners.

## Build order (how it actually went)

1. Pricing + synthetic engine + identity gate — no real data until this
   converged.
2. Data layer: canonical schema, write-once cache, yfinance provider (free
   snapshots; no historical chains there), config loader, time-basis module.
3. Replay adapter + daily attribution (gamma/theta/vega/carry/cost/net),
   gated by equivalence.
4. Policies + quarantined oracle + no-look-ahead tests.
5. Runner (position × protocol × policy × cost), capture metrics with
   entry-date cluster-bootstrap CIs, regime tags.
6. Paper figures rebuilt on the validated engine; invariant probes; live
   data-path verification.

Next: historical chains (OptionMetrics/WRDS or ThetaData — `TODOS.md`), then
the same pipeline on real SPX positions, where the regime-shift capture
result faces its real test.
