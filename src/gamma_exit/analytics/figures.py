"""Milestone 6: reproduce the source paper's Figures 1-3 with OUR validated
accounting -- and show where its story survives transaction costs.

Three figures, written to reports/:

fig1_surfaces.png   Paper Fig 1 rebuilt as heatmaps (single axis, diverging
                    color, no 3D theatrics): mean delta-hedged P&L over the
                    (drift spread x vol spread) plane, per moneyness x tenor.
                    Every cell is the VALIDATED engine run on GBM paths, not
                    a formula plot -- so the paper's claimed drift dependence
                    can be seen to vanish, and the vol-gap axis to carry
                    everything.

fig2_positions.png  Paper Fig 2 style: per-position panels (daily net /
                    cumulative net with the quarantined oracle exit marked /
                    spot path vs strike), one row per sample position.

fig3_summary.png    Paper Fig 3 style, upgraded to the study's real question:
                    mean P&L by scenario x policy, CAPTURE FRACTION by policy
                    x cost level (the headline panel), stop-time
                    distributions, win rates.

Run:  python -m gamma_exit.analytics.figures [--results results/xxx.parquet]
      (no --results: runs a fresh synthetic backtest first)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import TwoSlopeNorm

from gamma_exit.backtest.synthetic import SyntheticSource
from gamma_exit.config import DEFAULT_CONFIG, Config, load_config
from gamma_exit.plotstyle import (
    BASELINE,
    INK,
    MUTED,
    SURFACE,
    diverging_cmap,
    policy_color,
    style_axes,
)
from gamma_exit.pnl.engine import delta_hedge_synthetic, simulate_gbm_paths
from gamma_exit.pnl.replay import exit_values, replay_position
from gamma_exit.strategy.oracle import oracle_exit

DPI = 150


# --------------------------------------------------------------------------
# Figure 1: net P&L surfaces over (drift spread, vol spread)
# --------------------------------------------------------------------------
def fig1_surfaces(
    out_path: Path,
    r: float = 0.02,
    n_paths: int = 150,
    grid_n: int = 7,
    seed: int = 20260702,
) -> None:
    moneyness_levels = (0.90, 1.00, 1.10)  # S0/K
    tenors = (0.25, 0.5, 1.0)
    sigma_iv = 0.20
    vol_spreads = np.linspace(-0.08, 0.08, grid_n)
    drift_spreads = np.linspace(-0.20, 0.20, grid_n)

    fig, axes = plt.subplots(3, 3, figsize=(11, 9.5), dpi=DPI)
    fig.patch.set_facecolor(SURFACE)
    panels = []
    for mny in moneyness_levels:
        for t_years in tenors:
            n_steps = max(int(t_years * 126), 32)  # ~semi-daily rehedge
            z = np.empty((grid_n, grid_n))
            for i, dmu in enumerate(drift_spreads):
                for j, dsig in enumerate(vol_spreads):
                    paths = simulate_gbm_paths(
                        100.0, r + dmu, sigma_iv + dsig, t_years, n_steps, n_paths, seed
                    )
                    res = delta_hedge_synthetic(
                        paths, 100.0 / mny, t_years, r, sigma_iv
                    )
                    z[i, j] = res.pnl.mean()
            panels.append((mny, t_years, z))

    vmax = max(abs(z).max() for _, _, z in panels) or 1.0
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
    for idx, (ax, (mny, t_years, z)) in enumerate(zip(axes.flat, panels)):
        im = ax.imshow(
            z,
            origin="lower",
            aspect="auto",
            cmap=diverging_cmap(),
            norm=norm,
            extent=(vol_spreads[0], vol_spreads[-1], drift_spreads[0], drift_spreads[-1]),
        )
        style_axes(ax)
        ax.grid(False)
        ax.set_title(f"S/K={mny:.2f}   T={t_years:g}y", color=INK, fontsize=9, loc="left")
        if idx >= 6:  # outer-edge labels only: bottom row ...
            ax.set_xlabel("vol spread  σ_real − σ_IV", color=MUTED, fontsize=8)
        if idx % 3 == 0:  # ... and left column
            ax.set_ylabel("drift spread  μ − r", color=MUTED, fontsize=8)
    cbar = fig.colorbar(im, ax=axes, shrink=0.85, pad=0.02)
    cbar.set_label("mean hedged P&L ($/share)", color=MUTED, fontsize=9)
    cbar.ax.tick_params(colors=MUTED, labelsize=8)
    cbar.outline.set_edgecolor(BASELINE)  # type: ignore[union-attr,operator]
    fig.suptitle(
        "Delta-hedged P&L is a bet on the VOL GAP — drift barely matters\n"
        "(validated engine on GBM paths; the paper's −(μ−r)S²Γ term would tilt "
        "every panel toward the drift axis)",
        color=INK,
        fontsize=11,
        x=0.06,
        ha="left",
    )
    fig.savefig(out_path, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    print(f"fig1 -> {out_path}")


# --------------------------------------------------------------------------
# Figure 2: per-position daily / cumulative / spot panels
# --------------------------------------------------------------------------
def fig2_positions(out_path: Path, cfg: Config, n_positions: int = 5) -> None:
    source = SyntheticSource(
        n_per_scenario=1, r=cfg.rates.risk_free, q=cfg.rates.dividend_yield,
        base_seed=cfg.experiment.seed,
    )
    cands = source.positions()[:n_positions]

    fig, axes = plt.subplots(len(cands), 3, figsize=(12, 2.4 * len(cands)), dpi=DPI)
    fig.patch.set_facecolor(SURFACE)
    blue, red = policy_color("hold_to_expiry"), policy_color("oracle")
    for row, cand in zip(np.atleast_2d(axes), cands):
        res = replay_position(
            cand.quotes, cand.spec, cfg.rates.risk_free, q=cfg.rates.dividend_yield
        )
        vals, execable = exit_values(res)
        vals[-1] = res.pnl
        o_day, o_pnl = oracle_exit(vals, execable)
        d = res.daily
        x = np.arange(len(d))

        ax = row[0]
        ax.bar(x[1:], d["net"].iloc[1:], width=1.0, color=blue, linewidth=0)
        ax.axhline(0, color=BASELINE, lw=1)
        style_axes(ax)
        ax.set_ylabel(f"{cand.scenario}\ndaily net $", color=INK, fontsize=8)

        ax = row[1]
        ax.plot(x, d["cum_net"], color=blue, lw=2)
        ax.axvline(o_day, color=red, lw=1.2, ls="--")
        ax.plot([o_day], [o_pnl], "o", color=red, ms=6)
        ax.annotate(
            f"oracle exit (non-tradable)  {o_pnl:+.2f}",
            (o_day, o_pnl), xytext=(6, 6), textcoords="offset points",
            color=red, fontsize=7.5,
        )
        ax.axhline(0, color=BASELINE, lw=1)
        style_axes(ax)
        ax.set_ylabel("cum net $", color=MUTED, fontsize=8)

        ax = row[2]
        ax.plot(x, d["spot"], color=blue, lw=2)
        ax.axhline(cand.spec.strike, color=MUTED, lw=1.2, ls=":")
        ax.annotate(
            f"K={cand.spec.strike:.0f}", (x[-1], cand.spec.strike),
            xytext=(-4, 5), textcoords="offset points", ha="right",
            color=MUTED, fontsize=7.5,
        )
        ax.axvline(o_day, color=red, lw=1.2, ls="--")
        style_axes(ax)
        ax.set_ylabel("spot", color=MUTED, fontsize=8)
    for ax in np.atleast_2d(axes)[-1]:
        ax.set_xlabel("trading day", color=MUTED, fontsize=8)
    fig.suptitle(
        "One position per scenario: daily net, cumulative net with the "
        "quarantined oracle exit, spot vs strike",
        color=INK, fontsize=11, x=0.06, ha="left",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out_path, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    print(f"fig2 -> {out_path}")


# --------------------------------------------------------------------------
# Figure 3: cross-sectional summary of the backtest results
# --------------------------------------------------------------------------
def fig3_summary(out_path: Path, results: pd.DataFrame) -> None:
    from gamma_exit.analytics.metrics import capture_fraction

    sub = results[results["entry_protocol"] == "unconditional"]
    policies = [p for p in sub["policy"].unique()]
    cost_levels = list(dict.fromkeys(sub["cost_level"]))
    # one default variant per family keeps panel (a) readable; the full
    # pre-registered grid still appears in panels (b)-(d)
    defaults = [
        p for p in policies
        if p in ("hold_to_expiry", "oracle")
        or p.endswith(("_0.79", "_1.00", "_0.50")) and not p.startswith("fixed_time_0.50")
    ]

    fig, axes = plt.subplots(2, 2, figsize=(13, 9), dpi=DPI)
    fig.patch.set_facecolor(SURFACE)

    # (a) mean P&L by scenario x policy (default variants), zero-cost
    ax = axes[0, 0]
    zero = sub[sub["cost_level"] == cost_levels[0]]
    scen = zero.pivot_table(index="scenario", columns="policy", values="pnl")
    xs = np.arange(len(scen.index))
    width = 0.8 / len(defaults)
    for k, pol in enumerate(defaults):
        if pol in scen.columns:
            ax.bar(
                xs + k * width, scen[pol], width=width * 0.92,
                color=policy_color(pol), linewidth=0, label=pol,
            )
    ax.set_xticks(xs + 0.4 - width / 2, scen.index, fontsize=8)
    ax.axhline(0, color=BASELINE, lw=1)
    style_axes(ax)
    ax.set_title(
        "mean P&L by scenario (zero cost, default variants)",
        color=INK, fontsize=10, loc="left",
    )
    ax.legend(frameon=False, fontsize=7, ncols=2, labelcolor=INK)

    # (b) THE panel: capture fraction by policy x cost level
    ax = axes[0, 1]
    causal = [p for p in policies if p not in ("hold_to_expiry", "oracle")]
    xs = np.arange(len(causal))
    width = 0.8 / len(cost_levels)
    from gamma_exit.plotstyle import COST_LEVEL_COLORS

    for k, cl in enumerate(cost_levels):
        caps = [capture_fraction(sub[sub["cost_level"] == cl], p) for p in causal]
        ax.bar(
            xs + k * width, caps, width=width * 0.92,
            color=COST_LEVEL_COLORS.get(cl, MUTED), linewidth=0, label=cl,
        )
    ax.axhline(0, color=BASELINE, lw=1)
    ax.axhline(1.0, color=policy_color("oracle"), lw=1.2, ls="--")
    ax.annotate("oracle ceiling = 1.0", (0.02, 1.02), xycoords=("axes fraction", "data"),
                color=policy_color("oracle"), fontsize=8)
    ax.set_xticks(xs + 0.4 - width / 2, causal, fontsize=7, rotation=30, ha="right")
    style_axes(ax)
    ax.set_title(
        "capture of the oracle's edge over hold  (policy − hold)/(oracle − hold)",
        color=INK, fontsize=10, loc="left",
    )
    ax.legend(frameon=False, fontsize=7.5, labelcolor=INK)

    # (c) stop-time distribution per policy (zero cost)
    ax = axes[1, 0]
    data = [zero[zero["policy"] == p]["exit_frac"].to_numpy() for p in policies]
    bp = ax.boxplot(
        data, orientation="vertical", patch_artist=True, widths=0.55,
        medianprops={"color": INK, "lw": 1.2},
        flierprops={"marker": ".", "markersize": 3, "markerfacecolor": MUTED,
                    "markeredgecolor": MUTED},
    )
    for patch, pol in zip(bp["boxes"], policies):
        patch.set_facecolor(policy_color(pol))
        patch.set_alpha(0.75)
        patch.set_linewidth(0)
    for part in ("whiskers", "caps"):
        for line in bp[part]:
            line.set_color(BASELINE)
    ax.set_xticks(
        np.arange(1, len(policies) + 1), policies, fontsize=7, rotation=30, ha="right"
    )
    ax.set_ylabel("exit time (fraction of lifetime)", color=MUTED, fontsize=8.5)
    style_axes(ax)
    ax.set_title("stop-time distributions (zero cost)", color=INK, fontsize=10, loc="left")

    # (d) win rate by policy x cost level
    ax = axes[1, 1]
    xs = np.arange(len(policies))
    for k, cl in enumerate(cost_levels):
        wr = [
            (sub[(sub["cost_level"] == cl) & (sub["policy"] == p)]["pnl"] > 0).mean()
            for p in policies
        ]
        ax.bar(
            xs + k * width, wr, width=width * 0.92,
            color=COST_LEVEL_COLORS.get(cl, MUTED), linewidth=0, label=cl,
        )
    ax.set_xticks(xs + 0.4 - width / 2, policies, fontsize=7, rotation=30, ha="right")
    ax.set_ylim(0, 1.05)
    style_axes(ax)
    ax.set_title("win rate", color=INK, fontsize=10, loc="left")
    ax.legend(frameon=False, fontsize=7.5, labelcolor=INK)

    fig.suptitle(
        "Exit-rule cross-section: the oracle is a ceiling, not a strategy — "
        "the research question is how much of it CAUSAL rules keep, per cost level",
        color=INK, fontsize=11, x=0.06, ha="left",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out_path, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    print(f"fig3 -> {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=str(DEFAULT_CONFIG))
    ap.add_argument("--results", default=None, help="parquet from backtest.runner")
    ap.add_argument("--out-dir", default="reports")
    ap.add_argument("--positions-per-scenario", type=int, default=25)
    ap.add_argument("--skip-fig1", action="store_true", help="fig1 is the slow one")
    args = ap.parse_args()

    cfg = load_config(args.config)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    if args.results:
        results = pd.read_parquet(args.results)
    else:
        from gamma_exit.backtest.runner import run

        source = SyntheticSource(
            n_per_scenario=args.positions_per_scenario,
            r=cfg.rates.risk_free, q=cfg.rates.dividend_yield,
            base_seed=cfg.experiment.seed,
        )
        results = run(cfg, source, out_dir="results")

    if not args.skip_fig1:
        fig1_surfaces(out / "fig1_surfaces.png", r=cfg.rates.risk_free,
                      seed=cfg.experiment.seed)
    fig2_positions(out / "fig2_positions.png", cfg)
    fig3_summary(out / "fig3_summary.png", results)


if __name__ == "__main__":
    main()
