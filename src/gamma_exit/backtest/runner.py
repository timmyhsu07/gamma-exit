"""Milestone 5: run every position through every exit policy at every cost
level, with the no-look-ahead guarantee enforced structurally.

Per position the flow is:

    quotes ──► replay_position (hold-to-expiry, validated core)
                    │
                    ├─► exit_values()          realizable P&L per day
                    │        │
                    │        ├─► oracle_exit()   QUARANTINED ceiling (sees all)
                    │        │
    pre-entry ──► build_states()  one frozen PositionState per day (≤ t only)
    bars              │
                      └─► policy.decide(state) walked day by day; EXIT
                          executes at the first tradable day ≥ decision day

One row per (position × entry protocol × policy × cost level) lands in the
results frame; `run()` writes it as parquet plus a JSON sidecar carrying the
config dump, git commit, and timestamp (reproducibility hard rule).

Causality invariants (tests/test_no_lookahead.py):
- PositionState at day t built from a truncated replay is identical to one
  built from the full replay -- the state cannot encode the future;
- the causal forecast series is EWMA over closes up to each day only;
- the oracle result flows through a separate, loudly-named path.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from gamma_exit.backtest.synthetic import Candidate, SyntheticSource
from gamma_exit.config import DEFAULT_CONFIG, Config, CostLevel, load_config
from gamma_exit.conventions import years_to_expiry
from gamma_exit.pnl.replay import ReplayResult, replay_position
from gamma_exit.pnl.replay import exit_values as compute_exit_values
from gamma_exit.pricing.greeks import gamma as bs_gamma
from gamma_exit.pricing.greeks import theta as bs_theta
from gamma_exit.pricing.greeks import vega as bs_vega
from gamma_exit.strategy import Decision, ExitPolicy, PositionState, make_policies
from gamma_exit.strategy.oracle import oracle_exit
from gamma_exit.vol.forecast import ewma_vol

ORACLE_NAME = "oracle"  # ceiling label; never a tradable policy


def causal_forecast_series(
    pre_history: pd.Series, position_spots: pd.Series, decay: float
) -> pd.Series:
    """EWMA forecast vol aligned to position dates, causal by construction:
    the value at date d uses closes up to and including d (prefix invariance
    of ewm(adjust=False) is pinned by tests/test_vol.py)."""
    closes = pd.concat([pre_history, position_spots])
    return ewma_vol(closes, lam=decay).loc[position_spots.index]


def build_states(
    result: ReplayResult,
    r: float,
    q: float,
    forecast: pd.Series,
) -> list[PositionState]:
    """One frozen PositionState per day, from ≤ t data only.

    Every field is a scalar copied out of row t (or a running max up to t);
    nothing derived from rows > t enters, which the truncation test verifies
    object-by-object.
    """
    d = result.daily
    s = result.summary
    n = len(d)
    spot = d["spot"].to_numpy()
    # Greeks at each day's own state (same inputs the replay hedged with)
    tte_years = np.array([years_to_expiry(s["expiry"], dt) for dt in d["date"]])
    iv = d["iv"].to_numpy()
    gammas = bs_gamma(spot, s["strike"], tte_years, r, iv, q, s["kind"])
    thetas = bs_theta(spot, s["strike"], tte_years, r, iv, q, s["kind"])
    vegas = bs_vega(spot, s["strike"], tte_years, r, iv, q, s["kind"])

    cum = d["cum_net"].to_numpy()
    peak = np.maximum.accumulate(cum)
    fc = forecast.reindex(d["date"]).to_numpy(dtype=float)

    states = []
    for t in range(n):
        states.append(
            PositionState(
                day_index=t,
                total_days=n - 1,
                date=d["date"].iloc[t],
                spot=float(spot[t]),
                strike=float(s["strike"]),
                kind=s["kind"],
                expiry=s["expiry"],
                tte=float(tte_years[t]),
                mark=float(d["mark"].iloc[t]),
                stale=bool(d["stale"].iloc[t]),
                tradable=bool(d["tradable"].iloc[t]),
                iv=float(iv[t]),
                entry_iv=float(s["entry_iv"]),
                entry_mark=float(s["entry_mark"]),
                delta=float(d["delta"].iloc[t]),
                gamma=float(gammas[t]),
                theta=float(thetas[t]),
                vega=float(vegas[t]),
                cum_pnl=float(cum[t]),
                peak_pnl=float(peak[t]),
                forecast_vol=float(fc[t]),
                r=r,
                q=q,
            )
        )
    return states


def policy_exit_day(
    policy: ExitPolicy, states: list[PositionState], executable: np.ndarray
) -> int:
    """First day the policy says EXIT, pushed to the first executable day.

    Never earlier than day 1 (entry close is not an exit), never later than
    the final day (settlement / forced close), which is always executable in
    a hold-to-expiry replay.
    """
    n = len(states)
    decision_day = n - 1
    for st in states[1:]:
        if policy.decide(st) is Decision.EXIT:
            decision_day = st.day_index
            break
    later = np.where(executable[decision_day:])[0]
    return int(decision_day + later[0]) if later.size else n - 1


def run_candidate(
    cand: Candidate, cfg: Config, cost: CostLevel, policies: list[ExitPolicy]
) -> list[dict]:
    """All policy rows (+ the oracle ceiling row) for one position at one
    cost level."""
    r, q = cfg.rates.risk_free, cfg.rates.dividend_yield
    result = replay_position(
        cand.quotes,
        cand.spec,
        r,
        q=q,
        share_cost_per_share=cost.share_cost_per_share,
        option_spread_frac=cost.option_spread_frac,
    )
    values, executable = compute_exit_values(result)
    spots = pd.Series(
        result.daily["spot"].to_numpy(), index=result.daily["date"].to_numpy()
    )
    forecast = causal_forecast_series(cand.pre_history, spots, cfg.vol.forecast.decay)
    states = build_states(result, r, q, forecast)

    n = len(states)
    entry_fc = float(forecast.iloc[0])
    base = {
        "position_id": f"{cand.scenario}/{cand.seed}",
        "scenario": cand.scenario,
        "underlying": cand.spec.underlying,
        "kind": cand.spec.kind,
        "strike": cand.spec.strike,
        "expiry": cand.spec.expiry,
        "entry_date": cand.spec.entry_date,
        "moneyness": result.summary["entry_spot"] / cand.spec.strike,
        "dte_days": n - 1,
        "entry_iv": result.summary["entry_iv"],
        "entry_forecast_vol": entry_fc,
        "realized_vol_window": result.summary["realized_vol_window"],
        "window_return": float(spots.iloc[-1] / spots.iloc[0] - 1.0),
        "stale_days": result.summary["stale_days"],
        "cost_level": cost.name,
    }

    rows = []
    for pol in policies:
        e = policy_exit_day(pol, states, executable)
        pnl = float(result.pnl) if e == n - 1 else float(values[e])
        rows.append(
            {**base, "policy": pol.name, "exit_day": e, "exit_frac": e / (n - 1), "pnl": pnl}
        )
    # --- quarantined ceiling: full-path argmax, clearly labeled ------------
    values_for_oracle = values.copy()
    values_for_oracle[-1] = result.pnl  # settlement is always available
    o_day, o_pnl = oracle_exit(values_for_oracle, executable)
    rows.append(
        {
            **base,
            "policy": ORACLE_NAME,
            "exit_day": o_day,
            "exit_frac": o_day / (n - 1),
            "pnl": o_pnl,
        }
    )
    return rows


def apply_entry_protocols(rows: list[dict], cfg: Config) -> pd.DataFrame:
    """Tag every result row with each pre-registered entry protocol it passes:
    'unconditional' keeps everything; 'forecast_rv_gt_iv'
    keeps positions whose CAUSAL entry forecast exceeded the entry IV."""
    df = pd.DataFrame(rows)
    frames = []
    for protocol in cfg.entries:
        if protocol == "unconditional":
            sub = df.copy()
        elif protocol == "forecast_rv_gt_iv":
            sub = df[df["entry_forecast_vol"] > df["entry_iv"]].copy()
        else:
            raise ValueError(f"unknown entry protocol {protocol!r}")
        sub["entry_protocol"] = protocol
        frames.append(sub)
    return pd.concat(frames, ignore_index=True)


def git_commit_hash() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
    except Exception:
        return "unknown"


def run(cfg: Config, source: SyntheticSource, out_dir: str | Path | None = None) -> pd.DataFrame:
    """Every candidate x cost level x policy (+ oracle), one results frame."""
    policies = make_policies(cfg.policies)
    rows: list[dict] = []
    for cand in source.positions():
        for cost in cfg.costs.levels:
            rows.extend(run_candidate(cand, cfg, cost, policies))
    results = apply_entry_protocols(rows, cfg)

    if out_dir is not None:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        results.to_parquet(out / f"results_{stamp}.parquet", index=False)
        meta = {
            "created_utc": stamp,
            "git_commit": git_commit_hash(),
            "config": cfg.model_dump(mode="json"),
            "source": {
                "type": type(source).__name__,
                "n_per_scenario": source.n_per_scenario,
                "scenarios": [asdict(s) for s in source.scenarios],
                "base_seed": source.base_seed,
            },
            "n_rows": int(len(results)),
        }
        (out / f"results_{stamp}.meta.json").write_text(json.dumps(meta, indent=2, default=str))
        print(f"results -> {out / f'results_{stamp}.parquet'}  ({len(results)} rows)")
    return results


def main() -> None:
    from gamma_exit.analytics.metrics import summarize

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=str(DEFAULT_CONFIG))
    ap.add_argument("--positions-per-scenario", type=int, default=40)
    ap.add_argument("--out", default="results")
    args = ap.parse_args()

    cfg = load_config(args.config)
    source = SyntheticSource(
        n_per_scenario=args.positions_per_scenario,
        r=cfg.rates.risk_free,
        q=cfg.rates.dividend_yield,
        base_seed=cfg.experiment.seed,
    )
    results = run(cfg, source, out_dir=args.out)

    for protocol in cfg.entries:
        sub = results[results["entry_protocol"] == protocol]
        if sub.empty:
            print(f"\n=== entry protocol: {protocol} -- no positions passed ===")
            continue
        print(f"\n=== entry protocol: {protocol}  ({sub['position_id'].nunique()} positions) ===")
        with pd.option_context("display.width", 140, "display.precision", 3):
            print(summarize(sub).to_string())


if __name__ == "__main__":
    main()
