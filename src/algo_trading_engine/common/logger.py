"""
Singleton logging for backtest and trading flows using Loguru.

Configure once per run with configure_logger(run_type="backtest" | "trade");
all logs go to backtest.log or trade.log (overwritten each run). When
log_to_stdout=True, logs are sent to stdout (useful for AWS Lambda / CloudWatch)
instead of a file.
"""

import sys
from pathlib import Path
from typing import Literal

from loguru import logger

RunType = Literal["backtest", "trade"]
LogLevel = Literal["debug", "info", "warn"]

_LOG_DIR_DEFAULT = "logs"
_FILE_BY_RUN_TYPE: dict[RunType, str] = {
    "backtest": "backtest.log",
    "trade": "trade.log",
}
_LOGURU_LEVEL: dict[str, str] = {
    "debug": "DEBUG",
    "info": "INFO",
    "warn": "WARNING",
}
_sink_id: int | None = None


def configure_logger(
    run_type: RunType,
    log_dir: str = _LOG_DIR_DEFAULT,
    log_level: LogLevel = "info",
    log_to_stdout: bool = False,
) -> None:
    """
    Configure the singleton logger for this run. Only the first call takes
    effect; subsequent calls are silently ignored. This lets the outermost
    caller (e.g. a Lambda handler) lock in the configuration before inner
    code (e.g. PaperTradingEngine) calls configure_logger again.

    Call remove_logger_sink() first if you need to reconfigure.

    - When log_to_stdout is False (default), logs write to a file
      (backtest.log or trade.log), overwritten each run.
    - When log_to_stdout is True, logs write to stdout instead of a file.
      This is intended for AWS Lambda where stdout is captured by CloudWatch.

    Args:
        run_type: "backtest" or "trade" → determines log file name when writing to file.
        log_dir: Directory for log files (created if missing). Default "logs".
        log_level: One of "debug", "info", or "warn". Default "info".
        log_to_stdout: If True, log to stdout instead of a file.

    Raises:
        ValueError: If log_level is not "debug", "info", or "warn".
    """
    global _sink_id

    if _sink_id is not None:
        return

    normalized = log_level.lower().strip()
    if normalized not in _LOGURU_LEVEL:
        raise ValueError(
            f"log_level must be one of 'debug', 'info', 'warn'; got {log_level!r}"
        )
    level = _LOGURU_LEVEL[normalized]

    logger.remove(None)

    log_format = "{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} | {message}"

    if log_to_stdout:
        _sink_id = logger.add(
            sys.stdout,
            format=log_format,
            level=level,
        )
    else:
        path = Path(log_dir)
        path.mkdir(parents=True, exist_ok=True)
        log_file = path / _FILE_BY_RUN_TYPE[run_type]

        _sink_id = logger.add(
            str(log_file),
            format=log_format,
            level=level,
            mode="w",
        )


def remove_logger_sink() -> None:
    """
    Remove the current file sink and close the log file.

    Call this when the log file must be closed (e.g. in tests before removing
    a temp directory that contains the log file). Idempotent if no sink is set.
    """
    global _sink_id
    if _sink_id is not None:
        logger.remove(_sink_id)
        _sink_id = None


def get_logger():
    """
    Return the singleton Loguru logger.

    Call configure_logger(run_type=...) before the first use in a run so logs
    go to the correct file. If never configured, Loguru's default (stderr) may
    still be active.
    """
    return logger


def log_and_echo(message: str) -> None:
    """
    Log the message to the current log file and also print to stdout.

    Use only for recommendation-relevant messages in the trading flow so
    the user sees them while the full audit trail remains in trade.log.
    """
    logger.info(message)
    print(message)
