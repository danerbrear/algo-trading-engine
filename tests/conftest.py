"""Shared pytest configuration for algo-trading-engine."""

from __future__ import annotations

import os

# Non-interactive backend before any test imports matplotlib.
os.environ.setdefault("MPLBACKEND", "Agg")
