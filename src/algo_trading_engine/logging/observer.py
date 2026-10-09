"""Environment-driven run observer construction for GUI hosts."""

from __future__ import annotations

import os

from algo_trading_engine._internal.common.run_observer import JsonLinesRunObserver


def observer_from_env() -> JsonLinesRunObserver | None:
    """Return a JsonLinesRunObserver when ALGO_GUI_RUN_ID is set, else None."""
    run_id = os.environ.get("ALGO_GUI_RUN_ID", "").strip()
    if not run_id:
        return None
    return JsonLinesRunObserver(run_id)
