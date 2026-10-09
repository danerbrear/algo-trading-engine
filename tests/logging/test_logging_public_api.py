"""Public logging package exports."""

from algo_trading_engine.logging import (
    configure_logger,
    get_logger,
    observer_from_env,
    remove_logger_sink,
)


def test_public_exports_are_callable():
    assert callable(configure_logger)
    assert callable(get_logger)
    assert callable(remove_logger_sink)
    assert callable(observer_from_env)


def test_observer_from_env_none_without_run_id(monkeypatch):
    monkeypatch.delenv("ALGO_GUI_RUN_ID", raising=False)
    assert observer_from_env() is None
