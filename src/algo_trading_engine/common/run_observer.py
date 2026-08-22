"""
Optional run observer for streaming logs, progress, and results to a GUI host.

When ALGO_GUI_RUN_ID is set in the environment, strategy entrypoints construct
a JsonLinesRunObserver that writes JSON-line messages on stdout for the GUI
ProcessRunner to consume.
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from typing import Any, Protocol, TextIO

PROTOCOL_VERSION = 1


class RunObserver(Protocol):
    """Contract for streaming run events to an external host (e.g. GUI)."""

    def log(self, level: str, message: str) -> None:
        """Emit a log line."""

    def progress(self, current: int, total: int, label: str = "") -> None:
        """Emit progress as current/total with an optional label."""

    def result(self, stats: dict[str, str]) -> None:
        """Emit final result statistics."""


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _make_message(message_type: str, run_id: str, payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "v": PROTOCOL_VERSION,
        "type": message_type,
        "run_id": run_id,
        "ts": _utc_now_iso(),
        "payload": payload,
    }


class JsonLinesRunObserver:
    """Writes GUI protocol messages as newline-delimited JSON on a text stream."""

    def __init__(self, run_id: str, stream: TextIO | None = None) -> None:
        self._run_id = run_id
        self._stream = stream or sys.stdout

    def _send(self, message_type: str, payload: dict[str, Any]) -> None:
        line = json.dumps(_make_message(message_type, self._run_id, payload))
        self._stream.write(line)
        self._stream.write("\n")
        self._stream.flush()

    def log(self, level: str, message: str) -> None:
        del level  # GUI log pane shows message only; level is in the audit file
        self._send("log", {"line": message, "stream": "stdout"})

    def progress(self, current: int, total: int, label: str = "") -> None:
        if total <= 0:
            pct = 0
        else:
            pct = min(100, current * 100 // total)
        display_label = label or f"{current} / {total}"
        self._send("progress", {"pct": pct, "label": display_label})

    def result(self, stats: dict[str, str]) -> None:
        self._send("result", stats)


def observer_from_env() -> JsonLinesRunObserver | None:
    """Return a JsonLinesRunObserver when ALGO_GUI_RUN_ID is set, else None."""
    run_id = os.environ.get("ALGO_GUI_RUN_ID", "").strip()
    if not run_id:
        return None
    return JsonLinesRunObserver(run_id)
