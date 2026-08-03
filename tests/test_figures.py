"""Smoke tests: every figure renders to a non-trivial PNG on tiny inputs."""

from gamma_exit.analytics.figures import fig1_surfaces, fig2_positions, fig3_summary
from gamma_exit.backtest.runner import run
from gamma_exit.backtest.synthetic import Scenario, SyntheticSource
from gamma_exit.config import load_config


def test_all_figures_render(tmp_path):
    cfg = load_config()
    source = SyntheticSource(
        scenarios=(Scenario("baseline", 0.06, 0.26, 0.20, n_days=21),),
        n_per_scenario=3,
        base_seed=2,
    )
    results = run(cfg, source)

    fig1_surfaces(tmp_path / "f1.png", n_paths=8, grid_n=3, seed=1)
    fig2_positions(tmp_path / "f2.png", cfg, n_positions=2)
    fig3_summary(tmp_path / "f3.png", results)

    for name in ("f1.png", "f2.png", "f3.png"):
        assert (tmp_path / name).stat().st_size > 20_000, name
