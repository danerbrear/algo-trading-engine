"""Public logging package exports."""

from algo_trading_engine.logging import (
    configure_logger,
    get_logger,
    remove_logger_sink,
)


def test_public_exports_are_callable():
    assert callable(configure_logger)
    assert callable(get_logger)
    assert callable(remove_logger_sink)
