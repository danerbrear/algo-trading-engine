"""
Singleton logging for backtest and trading flows using Loguru.

Configure once per run with configure_logger(run_type="backtest" | "trade");
all logs go to backtest.log or trade.log (overwritten each run). No stdout
output from the logger; ProgressTracker / progress_print remain the stdout path.
"""

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
) -> None:
    """
    Configure the singleton logger for this run.

    - Removes any existing handler and adds a single file sink.
    - File is overwritten each run (mode="w").
    - Loguru's default stderr handler is removed so no logs go to stdout.
    - log_level controls log file verbosity only (not stdout).

    Args:
        run_type: "backtest" or "trade" → writes to backtest.log or trade.log.
        log_dir: Directory for log files (created if missing). Default "logs".
        log_level: One of "debug", "info", or "warn". Default "info".

    Raises:
        ValueError: If log_level is not "debug", "info", or "warn".
    """
    global _sink_id

    normalized = log_level.lower().strip()
    if normalized not in _LOGURU_LEVEL:
        raise ValueError(
            f"log_level must be one of 'debug', 'info', 'warn'; got {log_level!r}"
        )
    level = _LOGURU_LEVEL[normalized]

    # Remove all handlers (default stderr and any previous file sink); id 0 only exists before first configure
    logger.remove(None)

    path = Path(log_dir)
    path.mkdir(parents=True, exist_ok=True)
    log_file = path / _FILE_BY_RUN_TYPE[run_type]

    _sink_id = logger.add(
        str(log_file),
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} | {message}",
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
