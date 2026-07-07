"""Shared chart tokens + helpers (dataviz reference palette, light mode).

Categorical slots keep the palette's validated ordering (the ordering IS the
CVD-safety mechanism); color follows the ENTITY, so each policy owns a fixed
slot no matter which subset a figure shows. Oracle wears red on purpose: the
loud "this is the non-tradable ceiling" color.
"""

from __future__ import annotations

INK = "#0b0b0b"
MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"
SURFACE = "#fcfcfb"

SERIES = {  # categorical slots, validated order
    1: "#2a78d6",  # blue
    2: "#1baf7a",  # aqua
    3: "#eda100",  # yellow
    4: "#008300",  # green
    5: "#4a3aa7",  # violet
    6: "#e34948",  # red
}

# policy FAMILY -> color (fixed; parameter variants of one family share its
# hue -- the entity is the rule, the parameter is a variation of it)
POLICY_FAMILY_COLORS = {
    "hold_to_expiry": SERIES[1],
    "fixed_time": SERIES[2],
    "theta_gamma": SERIES[3],
    "trailing_stop": SERIES[4],
    "vol_regime": SERIES[5],
    "oracle": SERIES[6],
}

# ordinal ramp for the three cost levels (sequential blue, steps 250/450/650)
COST_LEVEL_COLORS = {"zero": "#86b6ef", "half_spread": "#2a78d6", "full_spread": "#104281"}


def policy_color(name: str) -> str:
    for family, color in POLICY_FAMILY_COLORS.items():
        if name.startswith(family):
            return color
    return MUTED


def style_axes(ax) -> None:
    """Recessive grid/axes, ink ticks -- the shared look of every figure."""
    ax.set_facecolor(SURFACE)
    ax.grid(True, which="major", color=GRID, lw=0.8)
    ax.tick_params(colors=MUTED, labelsize=8.5)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(BASELINE)


def diverging_cmap():
    """Loss-red -> neutral-gray -> profit-blue, for polarity (P&L) heatmaps."""
    from matplotlib.colors import LinearSegmentedColormap

    return LinearSegmentedColormap.from_list(
        "pnl_diverging", ["#e34948", "#efeeea", "#2a78d6"]
    )
