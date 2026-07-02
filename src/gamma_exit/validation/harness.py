"""Milestone 1 validation harness.

Runs the synthetic GBM delta-hedge replay across rehedge frequencies and
reports convergence of the discrete-hedge P&L to the pathwise identity

    X_T = int_0^T e^{r(T-u)} 1/2 Gamma_iv S_u^2 (sigma_real^2 - sigma_iv^2) du

plus a drift sweep showing mean P&L is invariant to mu (no -(mu-r) S^2 Gamma
term survives self-financing accounting).

Run:  python -m gamma_exit.validation.harness [--paths 5000] [--no-plot]
Writes reports/pnl_identity_convergence.png unless --no-plot.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from gamma_exit.pnl.engine import delta_hedge_synthetic, simulate_gbm_paths

S0, K, T, R = 100.0, 100.0, 1.0, 0.02
MU, SIGMA_REAL, SIGMA_IV = 0.08, 0.30, 0.20
FREQS = (12, 52, 252, 1008, 2520)
SEED = 20260702

# chart tokens from the reference dataviz palette (light mode)
INK = "#0b0b0b"
MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"
SURFACE = "#fcfcfb"
SERIES_1 = "#2a78d6"  # blue
SERIES_2 = "#1baf7a"  # aqua


def convergence_study(n_paths: int) -> list[dict]:
    rows = []
    for n_steps in FREQS:
        paths = simulate_gbm_paths(S0, MU, SIGMA_REAL, T, n_steps, n_paths, SEED)
        res = delta_hedge_synthetic(paths, K, T, R, SIGMA_IV, SIGMA_REAL)
        resid = res.pnl - res.identity_integral
        rows.append(
            {
                "rehedges_per_year": n_steps,
                "mean_pnl": float(res.pnl.mean()),
                "mean_identity": float(res.identity_integral.mean()),
                "mean_abs_gap": abs(float(resid.mean())),
                "rms_pathwise_resid": float(np.sqrt(np.mean(resid**2))),
                "se_mean": float(resid.std(ddof=1) / np.sqrt(n_paths)),
            }
        )
    return rows


def drift_sweep(n_paths: int, n_steps: int = 252) -> list[dict]:
    rows = []
    for mu in (R, R + 0.06, R + 0.15):
        paths = simulate_gbm_paths(S0, mu, SIGMA_REAL, T, n_steps, n_paths, SEED)
        res = delta_hedge_synthetic(paths, K, T, R, SIGMA_IV, SIGMA_REAL)
        resid = res.pnl - res.identity_integral
        rows.append(
            {
                "mu_minus_r": mu - R,
                "mean_pnl": float(res.pnl.mean()),
                "mean_identity": float(res.identity_integral.mean()),
                "mean_resid": float(resid.mean()),
                "se_resid": float(resid.std(ddof=1) / np.sqrt(n_paths)),
            }
        )
    return rows


def print_tables(conv: list[dict], drift: list[dict]) -> None:
    print(
        f"\nConvergence to identity  (S0={S0} K={K} T={T} r={R} mu={MU} "
        f"sigma_real={SIGMA_REAL} sigma_iv={SIGMA_IV})"
    )
    hdr = f"{'rehedges/yr':>11} {'mean P&L':>10} {'mean identity':>14} {'|gap|':>8} {'RMS resid':>10} {'SE(mean)':>9}"
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
    ap.add_argument("--paths", type=int, default=5000)
    ap.add_argument("--no-plot", action="store_true")
    args = ap.parse_args()

    conv = convergence_study(args.paths)
    drift = drift_sweep(args.paths)
    print_tables(conv, drift)
    if not args.no_plot:
        plot_convergence(conv, Path("reports/pnl_identity_convergence.png"))


if __name__ == "__main__":
    main()
