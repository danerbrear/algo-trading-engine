"""Tests for build/show plotting API."""

from __future__ import annotations

import os
import sys
from io import StringIO
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

import algo_trading_engine.plotting as plotting_pkg
from algo_trading_engine.logging import configure_logger, remove_logger_sink
from algo_trading_engine.gui import JsonLinesRunObserver
from algo_trading_engine.plotting import (
    HeatmapSpec,
    PlotConfig,
    build_plot_spec,
    show_plot,
)
from algo_trading_engine.plotting.serialize import payload_to_plot_frame, plot_payload
from algo_trading_engine.plotting.spec import EQUITY_CURVE_NAME


@pytest.fixture(autouse=True)
def reset_logger():
    remove_logger_sink()
    yield
    remove_logger_sink()


def test_public_all():
    assert plotting_pkg.__all__ == [
        "PlotSpec",
        "HeatmapSpec",
        "PlotConfig",
        "build_plot_spec",
        "show_plot",
    ]


def test_build_plot_spec_does_not_import_matplotlib():
    before = "matplotlib" in sys.modules
    build_plot_spec(
        pd.DataFrame({"timestamp": pd.to_datetime(["2024-01-01"]), "Close": [1.0]}),
        name="price",
    )
    if not before:
        assert "matplotlib" not in sys.modules


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


def test_show_plot_routes_to_observer(tmp_path):
    stream = StringIO()
    observer = JsonLinesRunObserver("run-plot", stream=stream)
    configure_logger("backtest", log_dir=str(tmp_path), observer=observer)

    frame = pd.DataFrame({"timestamp": pd.to_datetime(["2024-01-01"]), "equity": [3000.0]})
    spec = build_plot_spec(frame, name=EQUITY_CURVE_NAME)
    with patch("algo_trading_engine.plotting.show.render_spec_grid") as mock_grid:
        show_plot(spec, config=None)
        mock_grid.assert_not_called()

    payload_line = stream.getvalue().strip().splitlines()[-1]
    payload = __import__("json").loads(payload_line)
    assert payload["type"] == "dataframe"
    assert payload["payload"]["name"] == EQUITY_CURVE_NAME


def test_show_plot_skips_when_disabled():
    spec = build_plot_spec(
        pd.DataFrame({"index": [0, 1], "Close": [1.0, 2.0]}),
        name="price",
        x="index",
    )
    with patch(
        "algo_trading_engine.plotting.matplotlib_backend.render_lines_figure"
    ) as mock_render:
        show_plot(spec, config=PlotConfig(enabled=False))
        mock_render.assert_not_called()


def test_show_plot_renders_when_enabled():
    spec = build_plot_spec(
        pd.DataFrame({"index": [0, 1], "Close": [1.0, 2.0]}),
        name="price",
        x="index",
    )
    with patch(
        "algo_trading_engine.plotting.matplotlib_backend.render_lines_figure"
    ) as mock_render:
        show_plot(spec, config=PlotConfig(enabled=True, show=False))
        mock_render.assert_called_once()


def test_show_plot_heatmap_when_enabled():
    spec = HeatmapSpec(
        matrix=[[1, 0], [0, 1]],
        row_labels=("a", "b"),
        col_labels=("a", "b"),
        name="heatmap",
    )
    with patch(
        "algo_trading_engine.plotting.matplotlib_backend.render_heatmap_figure"
    ) as mock_render:
        show_plot(spec, config=PlotConfig(enabled=True, show=False))
        mock_render.assert_called_once()


def test_show_plot_import_error_hint():
    spec = build_plot_spec(
        pd.DataFrame({"index": [0], "Close": [1.0]}),
        name="price",
        x="index",
    )

    def _raise_plot_install_error():
        raise ImportError(
            "Plotting requires matplotlib. Install with: pip install algo-trading-engine[plot]"
        )

    with patch(
        "algo_trading_engine.plotting.matplotlib_backend._import_pyplot",
        side_effect=_raise_plot_install_error,
    ):
        with pytest.raises(ImportError, match="algo-trading-engine\\[plot\\]"):
            show_plot(spec, config=PlotConfig(enabled=True))


@pytest.mark.parametrize("backend", ["agg"])
def test_backend_smoke_render_lines(tmp_path, backend):
    os.environ["MPLBACKEND"] = backend
    spec = build_plot_spec(
        pd.DataFrame(
            {
                "timestamp": pd.to_datetime(["2024-01-01", "2024-02-01"]),
                "equity": [100.0, 110.0],
            }
        ),
        name="smoke_lines",
    )
    output = tmp_path / "lines.png"
    show_plot(spec, config=PlotConfig(enabled=True, show=False), save_path=output)
    assert output.exists()


@pytest.mark.parametrize("backend", ["agg"])
def test_backend_smoke_render_grid(tmp_path, backend):
    os.environ["MPLBACKEND"] = backend
    specs = [
        build_plot_spec(
            pd.DataFrame({"epoch": [0, 1], "loss": [1.0, 0.5]}),
            name="loss",
            x="epoch",
            title="Loss",
        ),
        build_plot_spec(
            pd.DataFrame({"epoch": [0, 1], "accuracy": [0.5, 0.8]}),
            name="accuracy",
            x="epoch",
            title="Accuracy",
        ),
    ]
    output = tmp_path / "grid.png"
    show_plot(specs, config=PlotConfig(enabled=True, show=False), save_path=output, ncols=2)
    assert output.exists()


@pytest.mark.parametrize("backend", ["agg"])
def test_backend_smoke_render_heatmap(tmp_path, backend):
    os.environ["MPLBACKEND"] = backend
    spec = HeatmapSpec(
        matrix=[[1, 0], [0, 2]],
        row_labels=("t0", "t1"),
        col_labels=("p0", "p1"),
        name="smoke_heatmap",
    )
    output = tmp_path / "heatmap.png"
    show_plot(spec, config=PlotConfig(enabled=True, show=False), save_path=output)
    assert output.exists()
