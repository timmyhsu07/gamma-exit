# TODOS

Deferred work with enough context to pick it up cold months from now.

## TD-1: Historical-data provider reader (ThetaData or OptionMetrics/WRDS)

- **What:** Implement the concrete `Provider` subclass for whichever source I
  end up with, normalizing into the canonical schema.
- **Why:** This is THE blocker for real-data results. yfinance has no
  historical chains, so replay currently runs only on synthetic worlds and
  freshly accumulated snapshots.
- **Context:** Provider ABC at `src/gamma_exit/data/providers/base.py`;
  `yfinance_provider.py` is the normalization template; cache keys are
  provider-scoped so vendors can coexist. OptionMetrics arrives as a WRDS
  export (reader over parquet/csv); ThetaData has a Python SDK. I
  deliberately did not write readers against imagined column schemas; that
  work happens once a real export is in hand to test against.
- **Blocked by:** WRDS access check via university, or paying for ThetaData.

## TD-2: Swap IV solver to py_vollib (Jäckel "Let's Be Rational")

- **What:** Replace the brentq loop in `pricing/implied_vol.py`; certify
  against the existing round-trip + QuantLib tests; keep brentq as the
  reference implementation in tests.
- **Why:** ~100x throughput for full-chain IV recomputation on historical
  data.
- **Trigger:** full-chain IV recompute exceeding ~60 s in profiling. Do not
  swap before then, since the transparent solver is easier to trust while the
  pipeline itself is still being audited.
- **Watch out:** re-verify NaN/vega-degenerate semantics match the current
  solver exactly.

## TD-3: Tighter inference for the real-data results

- **What:** Extend the existing entry-date cluster bootstrap: block bootstrap
  over calendar time, and report the pre-registered grid jointly rather than
  per-policy.
- **Why:** Overlapping holding windows share one realized-vol path, so the
  effective sample size is the number of distinct entry windows, not the
  number of positions. The cluster bootstrap in `analytics/metrics.py`
  handles the first-order problem; real-data results will get audited harder.
- **Depends on:** TD-1.

## TD-4: Decouple the simulation grid from the rehedge grid

- **What:** Simulate on a fine grid, rehedge on a coarser one (currently one
  knob controls both).
- **Why:** Separates "how finely the world moves" from "how often the trader
  hedges". Enables daily-hedging-on-intraday-paths studies and tightens the
  identity-integral approximation independently of hedge frequency.
- **Context:** `simulate_gbm_paths` is already exact at any resolution; the
  change is letting the accounting core skip rebalance steps while still
  marking daily.

## TD-5: Regime-switching scenario in the default set

- **What:** Promote the decaying-vol-edge world (currently hand-built in
  `tests/test_invariants.py` and `scripts/headline_numbers.py`) into a
  first-class `Scenario`, so the default backtest includes a world where
  causal exits can win.
- **Why:** The constant-gap scenarios show causal rules losing (correctly);
  the regime-shift world is where the actual research question lives.
