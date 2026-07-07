"""Test alias for the synthetic chain generator.

The generator was promoted to src/gamma_exit/backtest/synthetic.py when the
M5 runner started needing it; this module keeps the test imports stable and
guarantees tests exercise the exact generator production code uses.
"""

from gamma_exit.backtest.synthetic import synthetic_quotes  # noqa: F401
