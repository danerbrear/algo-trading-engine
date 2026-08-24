"""Tests for dataframe-first plotting module."""

from __future__ import annotations

import os
from datetime import datetime, timedelta
from io import StringIO
from unittest.mock import Mock, patch

import pandas as pd
import pytest

from algo_trading_engine.common.logger import configure_logger, remove_logger_sink
from algo_trading_engine.common.run_observer import JsonLinesRunObserver
from algo_trading_engine.plotting import (
    EQUITY_CURVE_NAME,
    build_equity_curve_dataframe,
    emit_plot,
    payload_to_plot_frame,
    plot_payload,
)
from algo_trading_engine.plotting.spec import build_plot_spec


@pytest.fixture(autouse=True)
def reset_logger():
    remove_logger_sink()
    yield
    remove_logger_sink()


def test_build_equity_curve_dataframe():
    positions = [
        {"exit_date": datetime(2024, 1, 10), "return_dollars": 100.0},
        {"exit_date": datetime(2024, 2, 1), "return_dollars": -50.0},
    ]
    frame = build_equity_curve_dataframe(positions, initial_capital=3000.0)
    assert list(frame.columns) == ["timestamp", "equity"]
    assert len(frame) == 3
    assert frame.iloc[0]["equity"] == 3000.0
    assert frame.iloc[-1]["equity"] == 3050.0


def test_plot_payload_equity_curve_round_trip():
    original = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2024-01-01", "2024-02-01"]),
            "equity": [3000.0, 3100.0],
        }
    )
    spec = build_plot_spec(original, name=EQUITY_CURVE_NAME)
    restored = payload_to_plot_frame(plot_payload(spec))
    pd.testing.assert_frame_equal(
        original.reset_index(drop=True),
        restored.reset_index(drop=True),
    )


def test_emit_plot_routes_to_observer(tmp_path):
    stream = StringIO()
    observer = JsonLinesRunObserver("run-plot", stream=stream)
    configure_logger("backtest", log_dir=str(tmp_path), observer=observer)

    frame = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2024-01-01"]),
            "equity": [3000.0],
        }
    )
    with patch("algo_trading_engine.plotting.emit.show_plot") as mock_show:
        emit_plot(frame, name=EQUITY_CURVE_NAME, show=True)
        mock_show.assert_not_called()

    payload_line = stream.getvalue().strip().splitlines()[-1]
    payload = __import__("json").loads(payload_line)
    assert payload["type"] == "dataframe"
    assert payload["payload"]["name"] == EQUITY_CURVE_NAME


def test_emit_plot_uses_matplotlib_when_no_observer():
    frame = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2024-01-01"]),
            "Close": [100.0],
        }
    )
    with patch("algo_trading_engine.plotting.emit._interactive_show_enabled", return_value=True):
        with patch("algo_trading_engine.plotting.emit.show_plot") as mock_show:
            emit_plot(frame, name="price_chart", show=True)
            mock_show.assert_called_once()


def test_emit_plot_skips_interactive_show_under_pytest():
    frame = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2024-01-01"]),
            "Close": [100.0],
        }
    )
    with patch.dict(os.environ, {"PYTEST_CURRENT_TEST": "tests/plotting/test_plotting.py::test"}):
        with patch("algo_trading_engine.plotting.emit._matplotlib_available", return_value=True):
            with patch("algo_trading_engine.plotting.emit.show_plot") as mock_show:
                emit_plot(frame, name="price_chart", show=True)
                mock_show.assert_not_called()


def test_emit_plot_noop_without_observer_or_matplotlib():
    frame = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2024-01-01"]),
            "Close": [100.0],
        }
    )
    with patch("algo_trading_engine.plotting.emit._matplotlib_available", return_value=False):
        with patch("algo_trading_engine.plotting.emit.show_plot") as mock_show:
            emit_plot(frame, name="price_chart", show=True)
            mock_show.assert_not_called()
