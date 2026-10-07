"""Plot specifications: normalized data + metadata for serialization and rendering."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

EQUITY_CURVE_NAME = "equity_curve"


def _ensure_x_column(frame: pd.DataFrame, x: str) -> pd.DataFrame:
    """Return a copy with column ``x`` present (from index when x is timestamp)."""
    if frame.empty:
        return frame.copy()

    result = frame.copy()
    if x not in result.columns:
        if x == "timestamp":
            if not isinstance(result.index, pd.DatetimeIndex):
                raise ValueError("Plot DataFrame requires a DatetimeIndex or a 'timestamp' column")
            result = result.reset_index()
            index_name = result.columns[0]
            if index_name != "timestamp":
                result = result.rename(columns={index_name: "timestamp"})
            result["timestamp"] = pd.to_datetime(result["timestamp"])
        else:
            raise ValueError(f"Plot DataFrame missing column: {x}")
    elif x == "timestamp":
        result[x] = pd.to_datetime(result[x])
    return result


def _normalize_equity_curve(frame: pd.DataFrame) -> pd.DataFrame:
    normalized = _ensure_x_column(frame, "timestamp")
    if "equity" not in normalized.columns:
        raise ValueError("equity_curve requires an 'equity' column")
    return normalized.loc[:, ["timestamp", "equity"]].copy()


@dataclass(frozen=True)
class PlotSpec:
    """Line chart over a shared x column."""

    frame: pd.DataFrame
    name: str = ""
    x: str = "timestamp"
    title: str = ""
    y_label: str = ""
    right_axis: tuple[str, ...] = ()
    markers: dict[str, pd.DataFrame] = field(default_factory=dict)
    y_ticks: tuple[tuple[float, str], ...] = ()
    meta: dict[str, Any] = field(default_factory=dict)

    def wire_compatible(self) -> bool:
        return self.x == "timestamp"

    def wire_meta(self) -> dict[str, Any]:
        meta = dict(self.meta)
        if self.title:
            meta.setdefault("title", self.title)
        if self.y_label:
            meta.setdefault("y_label", self.y_label)
        if self.right_axis:
            meta.setdefault("right_axis", list(self.right_axis))
        if self.markers:
            meta.setdefault("markers", self.markers)
        return meta

    def render(
        self,
        *,
        save_path: Path | str | None = None,
        show: bool = True,
    ) -> None:
        from algo_trading_engine.plotting import matplotlib_backend  # pylint: disable=import-outside-toplevel

        matplotlib_backend.render_lines_figure(self, save_path=save_path, show=show)


@dataclass(frozen=True)
class HeatmapSpec:
    """Matrix heatmap (e.g. confusion matrix)."""

    matrix: np.ndarray
    row_labels: tuple[str, ...]
    col_labels: tuple[str, ...]
    name: str = ""
    title: str = ""
    x_label: str = ""
    y_label: str = ""
    value_format: str = "d"

    def render(
        self,
        *,
        save_path: Path | str | None = None,
        show: bool = True,
    ) -> None:
        from algo_trading_engine.plotting import matplotlib_backend  # pylint: disable=import-outside-toplevel

        matplotlib_backend.render_heatmap_figure(self, save_path=save_path, show=show)


@dataclass(frozen=True)
class BarSpec:
    """Horizontal bar chart (package-internal)."""

    labels: tuple[str, ...]
    values: tuple[float, ...]
    name: str = ""
    title: str = ""
    x_label: str = ""

    def render(
        self,
        *,
        save_path: Path | str | None = None,
        show: bool = True,
    ) -> None:
        from algo_trading_engine.plotting import matplotlib_backend  # pylint: disable=import-outside-toplevel

        matplotlib_backend.render_bar_figure(self, save_path=save_path, show=show)


def build_plot_spec(
    frame: pd.DataFrame,
    *,
    name: str = "",
    x: str = "timestamp",
    title: str = "",
    y_label: str = "",
    right_axis: Sequence[str] | None = None,
    markers: dict[str, pd.DataFrame] | None = None,
    y_ticks: Sequence[tuple[float, str]] | None = None,
    meta: dict[str, Any] | None = None,
) -> PlotSpec:
    """Normalize user input into a PlotSpec (pure, no matplotlib)."""
    if name == EQUITY_CURVE_NAME:
        normalized = _normalize_equity_curve(frame)
        x = "timestamp"
    else:
        normalized = _ensure_x_column(frame, x)

    marker_frames: dict[str, pd.DataFrame] = {}
    for key, marker_frame in (markers or {}).items():
        if marker_frame is None or marker_frame.empty:
            continue
        marker_frames[key] = _ensure_x_column(marker_frame, x)

    return PlotSpec(
        name=name,
        frame=normalized,
        x=x,
        title=title,
        y_label=y_label,
        right_axis=tuple(right_axis or ()),
        markers=marker_frames,
        y_ticks=tuple(y_ticks or ()),
        meta=dict(meta or {}),
    )


def render_spec_grid(
    specs: Sequence[PlotSpec | HeatmapSpec | BarSpec],
    *,
    ncols: int,
    title: str = "",
    save_path: Path | str | None = None,
    show: bool = True,
) -> None:
    from algo_trading_engine.plotting import matplotlib_backend  # pylint: disable=import-outside-toplevel

    matplotlib_backend.render_grid(
        specs,
        ncols=ncols,
        title=title,
        save_path=save_path,
        show=show,
    )
