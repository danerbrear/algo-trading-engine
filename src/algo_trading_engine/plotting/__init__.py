"""DataFrame-first plotting API for engine and child repos."""

from algo_trading_engine.plotting.emit import emit_plot
from algo_trading_engine.plotting.equity import build_equity_curve_dataframe, emit_equity_curve
from algo_trading_engine.plotting.serialize import (
    coerce_plot_frame,
    payload_markers,
    payload_to_plot_frame,
    plot_payload,
)
from algo_trading_engine.plotting.spec import EQUITY_CURVE_NAME, PlotSpec, build_plot_spec

__all__ = [
    "EQUITY_CURVE_NAME",
    "PlotSpec",
    "build_equity_curve_dataframe",
    "build_plot_spec",
    "coerce_plot_frame",
    "emit_equity_curve",
    "emit_plot",
    "payload_markers",
    "payload_to_plot_frame",
    "plot_payload",
]
