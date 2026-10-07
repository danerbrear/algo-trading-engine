"""Optional matplotlib renderer for local runs."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Sequence

import numpy as np

from algo_trading_engine.plotting.spec import BarSpec, HeatmapSpec, PlotSpec

_NON_INTERACTIVE_BACKENDS = frozenset({"agg", "svg", "pdf", "ps", "cairo", "template"})


def _import_pyplot():
    try:
        import matplotlib.pyplot as plt  # pylint: disable=import-outside-toplevel
        return plt
    except ImportError as exc:
        raise ImportError(
            "Plotting requires matplotlib. Install with: pip install algo-trading-engine[plot]"
        ) from exc


def _interactive_show_requested(show: bool) -> bool:
    if not show:
        return False
    if "PYTEST_CURRENT_TEST" in os.environ:
        return False
    env_backend = os.environ.get("MPLBACKEND", "").lower()
    if env_backend in _NON_INTERACTIVE_BACKENDS:
        return False
    plt = _import_pyplot()
    if plt.get_backend().lower() in _NON_INTERACTIVE_BACKENDS:
        return False
    return True


def _finish(fig, *, save_path: Path | str | None, show: bool) -> None:
    plt = _import_pyplot()
    if save_path is not None:
        path = Path(save_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=300, bbox_inches="tight")
    if _interactive_show_requested(show):
        plt.show()
    plt.close(fig)


def render_lines_on_axes(spec: PlotSpec, ax) -> None:
    """Draw a PlotSpec on an existing matplotlib axes."""
    plt = _import_pyplot()
    frame = spec.frame
    x_values = frame[spec.x]

    left_cols = [
        col
        for col in frame.columns
        if col not in {spec.x, *spec.right_axis}
    ]
    for col in left_cols:
        ax.plot(x_values, frame[col], label=col, linewidth=1.5)

    for key, marker_frame in spec.markers.items():
        if marker_frame.empty:
            continue
        value_col = next((c for c in marker_frame.columns if c != spec.x), None)
        if value_col is None:
            continue
        ax.scatter(
            marker_frame[spec.x],
            marker_frame[value_col],
            label=key,
            s=60,
            zorder=5,
        )

    if spec.y_label:
        ax.set_ylabel(spec.y_label)
    ax.grid(True, alpha=0.3)

    twin_axes = []
    if spec.right_axis:
        offset = 0
        for col in spec.right_axis:
            if col not in frame.columns:
                continue
            twin = ax.twinx()
            if offset:
                twin.spines["right"].set_position(("outward", 60 * offset))
            twin.plot(x_values, frame[col], label=col, linestyle="--", alpha=0.7)
            twin.set_ylabel(col)
            twin_axes.append(twin)
            offset += 1

    lines, labels = ax.get_legend_handles_labels()
    for twin in twin_axes:
        t_lines, t_labels = twin.get_legend_handles_labels()
        lines += t_lines
        labels += t_labels
    if lines:
        ax.legend(lines, labels, loc="upper left")

    if spec.y_ticks:
        tick_values, tick_labels = zip(*spec.y_ticks)
        ax.set_yticks(tick_values)
        ax.set_yticklabels(tick_labels)

    if spec.title:
        ax.set_title(spec.title, fontsize=12, fontweight="bold")

    if spec.x == "timestamp":
        import matplotlib.dates as mdates  # pylint: disable=import-outside-toplevel

        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right")


def render_lines_figure(
    spec: PlotSpec,
    *,
    save_path: Path | str | None = None,
    show: bool = True,
) -> None:
    plt = _import_pyplot()
    fig, ax = plt.subplots(figsize=(14, 7))
    render_lines_on_axes(spec, ax)
    if spec.title and not ax.get_title():
        ax.set_title(spec.title, fontsize=14, fontweight="bold")
    fig.tight_layout()
    _finish(fig, save_path=save_path, show=show)


def render_heatmap_figure(
    spec: HeatmapSpec,
    *,
    save_path: Path | str | None = None,
    show: bool = True,
) -> None:
    plt = _import_pyplot()
    fig, ax = plt.subplots(figsize=(10, 8))
    matrix = np.asarray(spec.matrix)
    im = ax.imshow(matrix, aspect="auto")
    ax.set_xticks(range(len(spec.col_labels)))
    ax.set_yticks(range(len(spec.row_labels)))
    ax.set_xticklabels(spec.col_labels, rotation=45, ha="right")
    ax.set_yticklabels(spec.row_labels)
    if spec.title:
        ax.set_title(spec.title, fontsize=14, fontweight="bold")
    if spec.x_label:
        ax.set_xlabel(spec.x_label)
    if spec.y_label:
        ax.set_ylabel(spec.y_label)
    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            value = matrix[row, col]
            if spec.value_format == "d":
                text = f"{int(value)}"
            else:
                text = format(value, spec.value_format)
            ax.text(col, row, text, ha="center", va="center", color="black")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    _finish(fig, save_path=save_path, show=show)


def render_bar_figure(
    spec: BarSpec,
    *,
    save_path: Path | str | None = None,
    show: bool = True,
) -> None:
    plt = _import_pyplot()
    fig, ax = plt.subplots(figsize=(12, 8))
    y_pos = np.arange(len(spec.labels))
    ax.barh(y_pos, list(spec.values))
    ax.set_yticks(y_pos)
    ax.set_yticklabels(list(spec.labels))
    ax.invert_yaxis()
    if spec.title:
        ax.set_title(spec.title, fontsize=14, fontweight="bold")
    if spec.x_label:
        ax.set_xlabel(spec.x_label)
    ax.grid(True, alpha=0.3, axis="x")
    fig.tight_layout()
    _finish(fig, save_path=save_path, show=show)


def render_grid(
    specs: Sequence[PlotSpec | HeatmapSpec | BarSpec],
    *,
    ncols: int,
    title: str = "",
    save_path: Path | str | None = None,
    show: bool = True,
) -> None:
    if not specs:
        return
    plt = _import_pyplot()
    ncols = max(1, ncols)
    nrows = (len(specs) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(15, 4 * nrows))
    if nrows == 1 and ncols == 1:
        axes_list = [axes]
    elif nrows == 1 or ncols == 1:
        axes_list = list(np.atleast_1d(axes).flat)
    else:
        axes_list = list(axes.flat)

    for index, spec in enumerate(specs):
        ax = axes_list[index]
        if isinstance(spec, PlotSpec):
            render_lines_on_axes(spec, ax)
        elif isinstance(spec, HeatmapSpec):
            matrix = np.asarray(spec.matrix)
            ax.imshow(matrix, aspect="auto")
            ax.set_xticks(range(len(spec.col_labels)))
            ax.set_yticks(range(len(spec.row_labels)))
            ax.set_xticklabels(spec.col_labels, rotation=45, ha="right")
            ax.set_yticklabels(spec.row_labels)
            if spec.title:
                ax.set_title(spec.title)
        elif isinstance(spec, BarSpec):
            y_pos = np.arange(len(spec.labels))
            ax.barh(y_pos, list(spec.values))
            ax.set_yticks(y_pos)
            ax.set_yticklabels(list(spec.labels))
            ax.invert_yaxis()
            if spec.title:
                ax.set_title(spec.title)

    for ax in axes_list[len(specs) :]:
        ax.set_visible(False)

    if title:
        fig.suptitle(title, fontsize=16)
    fig.tight_layout()
    _finish(fig, save_path=save_path, show=show)
