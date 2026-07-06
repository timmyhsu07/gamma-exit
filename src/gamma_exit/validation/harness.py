"""Milestone 1 validation harness.

Runs the synthetic GBM delta-hedge replay across rehedge frequencies and
reports convergence of the discrete-hedge P&L to the pathwise identity

    X_T = int_0^T e^{r(T-u)} 1/2 Gamma_iv S_u^2 (sigma_real^2 - sigma_iv^2) du

plus a drift sweep showing mean P&L is invariant to mu (no -(mu-r) S^2 Gamma
term survives self-financing accounting).

Scenario parameters and the seed come from the experiment config (decision
2A); rehedge frequencies are a harness detail, not experiment config.

Run:  python -m gamma_exit.validation.harness [--config configs/baseline.yaml]
      [--paths N] [--no-plot]
Writes reports/pnl_identity_convergence.png unless --no-plot.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from gamma_exit.config import DEFAULT_CONFIG, Config, load_config
from gamma_exit.pnl.engine import (
    delta_hedge_synthetic,
    gamma_pnl_identity_integral,
    simulate_gbm_paths,
)

FREQS = (12, 52, 252, 1008, 2520)

# chart tokens from the reference dataviz palette (light mode)
INK = "#0b0b0b"
MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"
SURFACE = "#fcfcfb"
SERIES_1 = "#2a78d6"  # blue
SERIES_2 = "#1baf7a"  # aqua


def _run(cfg: Config, mu: float, n_steps: int, n_paths: int):
    v = cfg.validation
    paths = simulate_gbm_paths(
        v.s0, mu, v.sigma_real, v.t_years, n_steps, n_paths, cfg.experiment.seed
    )
    res = delta_hedge_synthetic(paths, v.strike, v.t_years, v.r, v.sigma_iv)
    identity = gamma_pnl_identity_integral(
        paths, v.strike, v.t_years, v.r, v.sigma_iv, v.sigma_real
    )
    return res, identity


def convergence_study(cfg: Config, n_paths: int, freqs: tuple[int, ...] = FREQS) -> list[dict]:
    rows = []
    for n_steps in freqs:
        res, identity = _run(cfg, cfg.validation.mu, n_steps, n_paths)
        resid = res.pnl - identity
        rows.append(
            {
                "rehedges_per_year": n_steps,
                "mean_pnl": float(res.pnl.mean()),
                "mean_identity": float(identity.mean()),
                "mean_abs_gap": abs(float(resid.mean())),
                "rms_pathwise_resid": float(np.sqrt(np.mean(resid**2))),
                "se_mean": float(resid.std(ddof=1) / np.sqrt(n_paths)),
            }
        )
    return rows


def drift_sweep(cfg: Config, n_paths: int, n_steps: int = 252) -> list[dict]:
    r = cfg.validation.r
    rows = []
    for mu in (r, r + 0.06, r + 0.15):
        res, identity = _run(cfg, mu, n_steps, n_paths)
        resid = res.pnl - identity
        rows.append(
            {
                "mu_minus_r": mu - r,
                "mean_pnl": float(res.pnl.mean()),
                "mean_identity": float(identity.mean()),
                "mean_resid": float(resid.mean()),
                "se_resid": float(resid.std(ddof=1) / np.sqrt(n_paths)),
            }
        )
    return rows


def print_tables(cfg: Config, conv: list[dict], drift: list[dict]) -> None:
    v = cfg.validation
    print(
        f"\nConvergence to identity  (s0={v.s0} K={v.strike} T={v.t_years} r={v.r} "
        f"mu={v.mu} sigma_real={v.sigma_real} sigma_iv={v.sigma_iv})"
    )
    hdr = (
        f"{'rehedges/yr':>11} {'mean P&L':>10} {'mean identity':>14} "
        f"{'|gap|':>8} {'RMS resid':>10} {'SE(mean)':>9}"
    )
    print(hdr)
    print("-" * len(hdr))
    for r_ in conv:
        print(
            f"{r_['rehedges_per_year']:>11d} {r_['mean_pnl']:>10.4f} "
            f"{r_['mean_identity']:>14.4f} {r_['mean_abs_gap']:>8.4f} "
            f"{r_['rms_pathwise_resid']:>10.4f} {r_['se_mean']:>9.4f}"
        )

    print("\nDrift invariance at 252 rehedges/yr (resid = P&L - identity; no mu bias)")
    hdr = f"{'mu - r':>8} {'mean P&L':>10} {'mean identity':>14} {'mean resid':>11} {'SE':>8}"
    print(hdr)
    print("-" * len(hdr))
    for r_ in drift:
        print(
            f"{r_['mu_minus_r']:>8.2f} {r_['mean_pnl']:>10.4f} "
            f"{r_['mean_identity']:>14.4f} {r_['mean_resid']:>11.4f} {r_['se_resid']:>8.4f}"
        )


def plot_convergence(conv: list[dict], out_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = np.array([r_["rehedges_per_year"] for r_ in conv], dtype=float)
    rms = np.array([r_["rms_pathwise_resid"] for r_ in conv])
    gap = np.array([r_["mean_abs_gap"] for r_ in conv])

    fig, ax = plt.subplots(figsize=(7.2, 4.6), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)

    # sqrt(dt) reference slope, anchored at the first RMS point (annotation, not a series)
    ref = rms[0] * np.sqrt(n[0] / n)
    ax.plot(n, ref, ls="--", lw=1.2, color=MUTED, zorder=1)
    ax.annotate("~ 1/sqrt(rehedges)", (n[-2], ref[-2] * 1.35), color=MUTED, fontsize=8)

    ax.plot(n, rms, lw=2, color=SERIES_1, marker="o", ms=6, zorder=3, label="RMS pathwise residual")
    ax.plot(n, gap, lw=2, color=SERIES_2, marker="o", ms=6, zorder=3, label="|mean P&L − mean identity|")

    # selective direct labels at the right end (relief rule for the aqua series)
    ax.annotate("RMS pathwise residual", (n[-1], rms[-1]), xytext=(-8, 10),
                textcoords="offset points", ha="right", color=INK, fontsize=8.5)
    ax.annotate("|mean gap|", (n[-1], gap[-1]), xytext=(-8, 10),
                textcoords="offset points", ha="right", color=INK, fontsize=8.5)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xticks(list(n))
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.set_xlabel("rehedges per year", color=MUTED)
    ax.set_ylabel("dollars (per 1 option on 100 shares eq.)", color=MUTED)
    ax.set_title(
        "Discrete delta-hedge P&L converges to the gamma–vega identity",
        color=INK, fontsize=11, loc="left",
    )
    ax.grid(True, which="major", color=GRID, lw=0.8)
    ax.tick_params(colors=MUTED, labelsize=8.5)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(BASELINE)
    leg = ax.legend(frameon=False, fontsize=8.5, loc="lower left")
    for txt in leg.get_texts():
        txt.set_color(INK)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, facecolor=SURFACE)
    print(f"\nplot -> {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=str(DEFAULT_CONFIG))
    ap.add_argument("--paths", type=int, default=None, help="override config n_paths")
    ap.add_argument("--no-plot", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config)
    n_paths = args.paths or cfg.validation.n_paths

    conv = convergence_study(cfg, n_paths)
    drift = drift_sweep(cfg, n_paths)
    print_tables(cfg, conv, drift)
    if not args.no_plot:
        plot_convergence(conv, Path("reports/pnl_identity_convergence.png"))


if __name__ == "__main__":
    main()
