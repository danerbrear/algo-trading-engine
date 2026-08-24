"""Build equity-curve DataFrames from backtest closed positions."""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any, Iterable

import pandas as pd

from algo_trading_engine.plotting.spec import EQUITY_CURVE_NAME


def build_equity_curve_dataframe(
    closed_positions: Iterable[dict[str, Any]],
    initial_capital: float,
) -> pd.DataFrame:
    """
    Build ``(timestamp, equity)`` rows from closed position records.

    Mirrors ``calculate_equity_curve`` in prediction/plot_equity_curve.py but
    uses backtest ``closed_positions`` dicts (``exit_date``, ``return_dollars``).
    """
    sorted_positions = sorted(closed_positions, key=lambda p: p["exit_date"])
    if not sorted_positions:
        return pd.DataFrame(columns=["timestamp", "equity"])

    rows: list[dict[str, Any]] = []
    start_date = sorted_positions[0]["exit_date"] - timedelta(days=1)
    rows.append({"timestamp": start_date, "equity": float(initial_capital)})

    capital = float(initial_capital)
    for position in sorted_positions:
        capital += float(position["return_dollars"])
        rows.append({"timestamp": position["exit_date"], "equity": capital})

    frame = pd.DataFrame(rows)
    frame["timestamp"] = pd.to_datetime(frame["timestamp"])
    return frame


def emit_equity_curve(
    closed_positions: Iterable[dict[str, Any]],
    initial_capital: float,
    *,
    show: bool = True,
) -> pd.DataFrame:
    """Build and emit the standard equity curve plot."""
    from algo_trading_engine.plotting.emit import emit_plot  # pylint: disable=import-outside-toplevel

    frame = build_equity_curve_dataframe(closed_positions, initial_capital)
    if frame.empty:
        return frame
    emit_plot(
        frame,
        name=EQUITY_CURVE_NAME,
        title="Equity Curve",
        y_label="Equity ($)",
        show=show,
    )
    return frame
