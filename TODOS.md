# TODOS

Deferred work with full context. Each entry records what/why/state so it can be
picked up cold months later. Added by /plan-eng-review on 2026-07-02.

## TD-1: Historical-data provider reader (ThetaData or OptionMetrics/WRDS)

- **What:** Implement the concrete `Provider` subclass for whichever source the
  access check lands on, normalizing to the canonical schema.
- **Why:** M3 replay is impossible without historical option chains; yfinance has
  none. This is the deferred half of the scope decision D3 (2026-07-02 eng review).
- **Context:** Provider ABC at `src/gamma_exit/data/providers/base.py`;
  `yfinance_provider.py` is the normalization template; cache keys are
  provider-scoped so vendors coexist. OptionMetrics comes as a WRDS export
  (reader over parquet/csv); ThetaData has a Python SDK.
- **Blocked by:** Data-access verification (decision 11A: check WRDS via
  university + price ThetaData this week).

## TD-2: Swap IV solver to py_vollib (Jäckel "Let's Be Rational")

- **What:** Replace the brentq loop in `pricing/implied_vol.py`; certify against
  the existing round-trip + QuantLib tests; keep brentq as reference in tests.
- **Why:** ~100× throughput for M3+ full-chain IV recomputation.
- **Trigger (decision 10A):** full-chain IV recompute exceeds ~60s in M3 profiling.
  Do not swap before the trigger fires.
- **Watch out:** re-verify NaN/vega-degenerate semantics match the current solver.

## TD-3: Inference plan for M5/M6 results

- **What:** Block-bootstrap confidence intervals on capture fraction (calendar-time
  blocks), cluster by entry date, and a pre-registered policy/threshold grid.
- **Why:** Overlapping holding windows share one realized-vol path — effective
  sample size is the number of non-overlapping windows (~10–30), not
  positions × policies. Without CIs + pre-commitment the headline numbers are
  not defensible. (Outside-voice finding 7, 2026-07-02 review.)
- **Context:** Builds on the 12A bootstrap-baseline machinery in `analytics/`;
  the grid pre-commitment belongs in `configs/` BEFORE the first full M5 run.
- **Depends on:** 12A implementation, M5 runner.

## TD-4: Decouple simulation grid from rehedge grid in the synthetic engine

- **What:** Simulate on a fine grid, rehedge on a coarser one (currently one knob).
- **Why:** Separates "how finely the world moves" from "how often the trader
  hedges" — enables daily-hedging-vs-intraday-paths studies and tightens the
  identity-integral approximation independently of hedge frequency. This was
  caveat #2 in the Milestone 1 notes.
- **Context:** `simulate_gbm_paths` is already exact at any resolution; the change
  is letting the accounting core skip rebalance steps while still marking daily.
- **Depends on:** the 1A shared-accounting-core refactor (natural time: right after).
