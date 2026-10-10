"""Public gui package exports."""

from algo_trading_engine.gui import (
    JsonLinesRunObserver,
    RunObserver,
    observer_from_env,
)


def test_public_exports():
    assert JsonLinesRunObserver is not None
    assert RunObserver is not None
    assert callable(observer_from_env)


def test_observer_from_env_none_without_run_id(monkeypatch):
    monkeypatch.delenv("ALGO_GUI_RUN_ID", raising=False)
    assert observer_from_env() is None
