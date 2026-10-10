"""Unit tests for RunObserver and JsonLinesRunObserver."""

from __future__ import annotations

import io
import json
import os
from dataclasses import dataclass, field

import pandas as pd
import pytest

from algo_trading_engine.gui import JsonLinesRunObserver, observer_from_env


@dataclass
class RecordingObserver:
    logs: list[tuple[str, str]] = field(default_factory=list)
    progress_events: list[tuple[int, int, str]] = field(default_factory=list)
    results: list[dict[str, str]] = field(default_factory=list)

    def log(self, level: str, message: str) -> None:
        self.logs.append((level, message))

    def progress(self, current: int, total: int, label: str = "") -> None:
        self.progress_events.append((current, total, label))

    def result(self, stats: dict[str, str]) -> None:
        self.results.append(stats)


class TestJsonLinesRunObserver:
    def test_log_emits_gui_envelope(self) -> None:
        stream = io.StringIO()
        observer = JsonLinesRunObserver("run-abc", stream=stream)
        observer.log("INFO", "hello world")
        message = json.loads(stream.getvalue().strip())
        assert message["v"] == 1
        assert message["type"] == "log"
        assert message["run_id"] == "run-abc"
        assert message["payload"]["line"] == "hello world"
        assert message["payload"]["stream"] == "stdout"

    def test_progress_emits_pct_and_label(self) -> None:
        stream = io.StringIO()
        observer = JsonLinesRunObserver("run-abc", stream=stream)
        observer.progress(50, 200, "Processing 2025-01-01 (25.0% bars)")
        message = json.loads(stream.getvalue().strip())
        assert message["type"] == "progress"
        assert message["payload"]["pct"] == 25
        assert message["payload"]["label"] == "Processing 2025-01-01 (25.0% bars)"

    def test_progress_defaults_label(self) -> None:
        stream = io.StringIO()
        observer = JsonLinesRunObserver("run-abc", stream=stream)
        observer.progress(3, 10)
        message = json.loads(stream.getvalue().strip())
        assert message["payload"]["pct"] == 30
        assert message["payload"]["label"] == "3 / 10"

    def test_result_emits_stats(self) -> None:
        stream = io.StringIO()
        observer = JsonLinesRunObserver("run-abc", stream=stream)
        stats = {"total_return": "$+100.00 (+3.33%)", "sharpe_ratio": "0.842"}
        observer.result(stats)
        message = json.loads(stream.getvalue().strip())
        assert message["type"] == "result"
        assert message["payload"] == stats

    def test_dataframe_emits_plot_payload(self) -> None:
        stream = io.StringIO()
        observer = JsonLinesRunObserver("run-abc", stream=stream)
        frame = pd.DataFrame(
            {
                "timestamp": pd.to_datetime(["2024-01-01", "2024-02-01"]),
                "equity": [3000.0, 3100.0],
            }
        )
        observer.dataframe("equity_curve", frame, meta={"title": "Equity Curve"})
        message = json.loads(stream.getvalue().strip())
        assert message["type"] == "dataframe"
        assert message["payload"]["name"] == "equity_curve"
        assert message["payload"]["title"] == "Equity Curve"
        assert message["payload"]["orient"] == "split"


class TestObserverFromEnv:
    def test_returns_none_when_unset(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("ALGO_GUI_RUN_ID", raising=False)
        assert observer_from_env() is None

    def test_returns_none_when_blank(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("ALGO_GUI_RUN_ID", "   ")
        assert observer_from_env() is None

    def test_returns_observer_when_set(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("ALGO_GUI_RUN_ID", "abc123")
        observer = observer_from_env()
        assert observer is not None
        assert isinstance(observer, JsonLinesRunObserver)
