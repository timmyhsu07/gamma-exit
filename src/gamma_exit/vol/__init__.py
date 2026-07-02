"""Volatility estimators.

HARD RULE (PROJECT_BRIEF.md section 2): realized.py is EX-POST truth, usable
only by the oracle and for P&L attribution. forecast.py is the ONLY vol a
causal exit policy may consume. Never substitute one for the other.
"""
