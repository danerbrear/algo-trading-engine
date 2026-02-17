"""
Unit tests for the singleton logger (Loguru) in common.logger.
"""

import pytest
import tempfile
from pathlib import Path

from algo_trading_engine.common.logger import (
    configure_logger,
    get_logger,
    log_and_echo,
    remove_logger_sink,
)


@pytest.fixture
def tmp_log_dir():
    """Temporary directory for log files; closes logger sink on teardown so Windows can delete the dir."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir
        remove_logger_sink()


class TestConfigureLogger:
    """Tests for configure_logger."""

    def test_creates_log_dir_and_backtest_log_file(self, tmp_log_dir):
        tmpdir = tmp_log_dir
        configure_logger("backtest", log_dir=tmpdir, log_level="info")
        log_file = Path(tmpdir) / "backtest.log"
        assert log_file.exists()
        get_logger().info("test message")
        assert "test message" in log_file.read_text()

    def test_creates_trade_log_file(self, tmp_log_dir):
        tmpdir = tmp_log_dir
        configure_logger("trade", log_dir=tmpdir, log_level="info")
        log_file = Path(tmpdir) / "trade.log"
        assert log_file.exists()
        get_logger().info("trade test")
        assert "trade test" in log_file.read_text()

    def test_log_level_debug(self, tmp_log_dir):
        tmpdir = tmp_log_dir
        configure_logger("backtest", log_dir=tmpdir, log_level="debug")
        get_logger().debug("debug message")
        log_file = Path(tmpdir) / "backtest.log"
        assert "debug message" in log_file.read_text()

    def test_log_level_warn(self, tmp_log_dir):
        tmpdir = tmp_log_dir
        configure_logger("backtest", log_dir=tmpdir, log_level="warn")
        get_logger().info("info message")
        get_logger().warning("warn message")
        log_file = Path(tmpdir) / "backtest.log"
        text = log_file.read_text()
        assert "warn message" in text
        assert "info message" not in text

    def test_log_level_case_insensitive(self, tmp_log_dir):
        tmpdir = tmp_log_dir
        configure_logger("backtest", log_dir=tmpdir, log_level="INFO")
        get_logger().info("ok")
        assert (Path(tmpdir) / "backtest.log").exists()

    def test_invalid_log_level_raises(self, tmp_log_dir):
        tmpdir = tmp_log_dir
        with pytest.raises(ValueError, match="log_level must be one of"):
            configure_logger("backtest", log_dir=tmpdir, log_level="invalid")

    def test_overwrites_file_on_reconfigure(self, tmp_log_dir):
        tmpdir = tmp_log_dir
        configure_logger("backtest", log_dir=tmpdir, log_level="info")
        get_logger().info("first")
        configure_logger("backtest", log_dir=tmpdir, log_level="info")
        get_logger().info("second")
        log_file = Path(tmpdir) / "backtest.log"
        text = log_file.read_text()
        assert "second" in text
        assert "first" not in text


class TestGetLogger:
    """Tests for get_logger."""

    def test_returns_logger(self):
        assert get_logger() is not None
        assert hasattr(get_logger(), "info")
        assert hasattr(get_logger(), "debug")
        assert hasattr(get_logger(), "warning")


class TestLogAndEcho:
    """Tests for log_and_echo."""

    def test_log_and_echo_writes_to_file(self, tmp_log_dir, capsys):
        tmpdir = tmp_log_dir
        configure_logger("backtest", log_dir=tmpdir, log_level="info")
        log_and_echo("recommendation message")
        log_file = Path(tmpdir) / "backtest.log"
        assert "recommendation message" in log_file.read_text()

    def test_log_and_echo_prints_to_stdout(self, tmp_log_dir, capsys):
        tmpdir = tmp_log_dir
        configure_logger("backtest", log_dir=tmpdir, log_level="info")
        log_and_echo("echo to user")
        captured = capsys.readouterr()
        assert "echo to user" in captured.out
