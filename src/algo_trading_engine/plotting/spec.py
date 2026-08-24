"""Plot specification: normalized DataFrame + metadata for serialization and rendering."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd

EQUITY_CURVE_NAME = "equity_curve"


@dataclass(frozen=True)
class PlotSpec:
    """Serializable plot description passed to observers or matplotlib backend."""

    name: str
    frame: pd.DataFrame
    title: str = ""
    y_label: str = ""
    right_axis: tuple[str, ...] = ()
    markers: dict[str, pd.DataFrame] = field(default_factory=dict)
    meta: dict[str, Any] = field(default_factory=dict)


def _ensure_timestamp_column(frame: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with a ``timestamp`` column (from index when needed)."""
    if frame.empty:
        return frame.copy()

    result = frame.copy()
    if "timestamp" not in result.columns:
        if not isinstance(result.index, pd.DatetimeIndex):
            raise ValueError("Plot DataFrame requires a DatetimeIndex or a 'timestamp' column")
        result = result.reset_index()
        index_name = result.columns[0]
        if index_name != "timestamp":
            result = result.rename(columns={index_name: "timestamp"})
    result["timestamp"] = pd.to_datetime(result["timestamp"])
    return result


def _normalize_equity_curve(frame: pd.DataFrame) -> pd.DataFrame:
    normalized = _ensure_timestamp_column(frame)
    if "equity" not in normalized.columns:
        raise ValueError("equity_curve requires an 'equity' column")
    return normalized.loc[:, ["timestamp", "equity"]].copy()


def build_plot_spec(
    frame: pd.DataFrame,
    *,
    name: str,
    title: str = "",
    y_label: str = "",
    right_axis: list[str] | tuple[str, ...] | None = None,
    markers: dict[str, pd.DataFrame] | None = None,
    meta: dict[str, Any] | None = None,
) -> PlotSpec:
    """Normalize user input into a PlotSpec."""
    if name == EQUITY_CURVE_NAME:
        normalized = _normalize_equity_curve(frame)
    else:
        normalized = _ensure_timestamp_column(frame)

    marker_frames: dict[str, pd.DataFrame] = {}
    for key, marker_frame in (markers or {}).items():
        if marker_frame is None or marker_frame.empty:
            continue
        marker_frames[key] = _ensure_timestamp_column(marker_frame)

    return PlotSpec(
        name=name,
        frame=normalized,
        title=title,
        y_label=y_label,
        right_axis=tuple(right_axis or ()),
        markers=marker_frames,
        meta=dict(meta or {}),
    )
