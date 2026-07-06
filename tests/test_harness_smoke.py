"""Smoke test: the Milestone 1 harness CLI must run end-to-end (decision 8A)."""

import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def test_harness_cli_runs_clean():
    out = subprocess.run(
        [
            sys.executable,
            "-m",
            "gamma_exit.validation.harness",
            "--config",
            str(REPO / "configs" / "baseline.yaml"),
            "--paths",
            "120",
            "--no-plot",
        ],
        cwd=REPO,
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert out.returncode == 0, out.stderr
    assert "Convergence to identity" in out.stdout
    assert "Drift invariance" in out.stdout
