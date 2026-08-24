"""Route plot specs to GUI observers or optional matplotlib display."""

from __future__ import annotations

import os
from typing import Any

import pandas as pd

from algo_trading_engine.common.logger import get_active_observer
from algo_trading_engine.plotting.matplotlib_backend import show_plot
from algo_trading_engine.plotting.spec import PlotSpec, build_plot_spec

_NON_INTERACTIVE_BACKENDS = frozenset({"agg", "svg", "pdf", "ps", "cairo", "template"})


def _observer_supports_dataframe(observer: object) -> bool:
    return callable(getattr(observer, "dataframe", None))


def emit_plot(
    frame: pd.DataFrame,
    *,
    name: str,
    title: str = "",
    y_label: str = "",
    right_axis: list[str] | tuple[str, ...] | None = None,
    markers: dict[str, pd.DataFrame] | None = None,
    meta: dict[str, Any] | None = None,
    show: bool = True,
) -> PlotSpec:
    """
    Emit a plot from a pandas DataFrame.

    Routing:
    1. Active RunObserver with ``dataframe`` → JSON wire (GUI subprocess).
    2. Else if ``show`` and matplotlib available → local window.
    3. Else no-op.
    """
    spec = build_plot_spec(
        frame,
        name=name,
        title=title,
        y_label=y_label,
        right_axis=right_axis,
        markers=markers,
        meta=meta,
    )

    observer = get_active_observer()
    if observer is not None and _observer_supports_dataframe(observer):
        observer.dataframe(spec.name, spec.frame, meta=_plot_meta(spec))
        return spec

    if _interactive_show_enabled(show):
        show_plot(spec)

    return spec


def _interactive_show_enabled(show: bool) -> bool:
    """True when a blocking local matplotlib window should open."""
    if not show or not _matplotlib_available():
        return False
    if "PYTEST_CURRENT_TEST" in os.environ:
        return False
    env_backend = os.environ.get("MPLBACKEND", "").lower()
    if env_backend in _NON_INTERACTIVE_BACKENDS:
        return False
    try:
        import matplotlib  # pylint: disable=import-outside-toplevel

        if matplotlib.get_backend().lower() in _NON_INTERACTIVE_BACKENDS:
            return False
    except ImportError:
        return False
    return True


def _plot_meta(spec: PlotSpec) -> dict[str, Any]:
    meta = dict(spec.meta)
    if spec.title:
        meta.setdefault("title", spec.title)
    if spec.y_label:
        meta.setdefault("y_label", spec.y_label)
    if spec.right_axis:
        meta.setdefault("right_axis", list(spec.right_axis))
    if spec.markers:
        meta.setdefault("markers", spec.markers)
    return meta


def _matplotlib_available() -> bool:
    try:
        import matplotlib  # pylint: disable=import-outside-toplevel, unused-import
        return True
    except ImportError:
        return False
