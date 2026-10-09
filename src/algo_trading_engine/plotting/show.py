"""Route built plot specs to GUI observers or local matplotlib rendering."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Union

from algo_trading_engine.logging.logger import get_active_observer
from algo_trading_engine.plotting.config import PlotConfig
from algo_trading_engine.plotting.spec import (
    BarSpec,
    HeatmapSpec,
    PlotSpec,
    render_spec_grid,
)

RenderableSpec = Union[PlotSpec, HeatmapSpec, BarSpec]
SpecInput = Union[RenderableSpec, Sequence[RenderableSpec]]


def _observer_supports_dataframe(observer: object) -> bool:
    return callable(getattr(observer, "dataframe", None))


def _resolve_save_path(
    config: PlotConfig | None,
    spec: RenderableSpec,
    save_path: Path | str | None,
) -> Path | None:
    if save_path is not None:
        return Path(save_path)
    if config is None or config.save_dir is None:
        return None
    name = spec.name or type(spec).__name__.lower()
    return config.save_dir / f"{name}.png"


def show_plot(
    spec_or_specs: SpecInput,
    *,
    config: PlotConfig | None,
    save_path: Path | str | None = None,
    ncols: int = 2,
    grid_title: str = "",
) -> None:
    """
    Display or emit plot data.

    1. Single wire-compatible PlotSpec + active observer → JSON to GUI (config ignored).
    2. Else if config is None or not enabled → no-op (no matplotlib import).
    3. Else render locally via matplotlib.
    """
    if isinstance(spec_or_specs, (PlotSpec, HeatmapSpec, BarSpec)):
        _show_single(spec_or_specs, config=config, save_path=save_path)
        return

    specs = list(spec_or_specs)
    if not specs:
        return
    if config is None or not config.enabled:
        return
    resolved = save_path
    if resolved is None and config.save_dir is not None:
        resolved = config.save_dir / "grid.png"
    show_window = config.show
    render_spec_grid(
        specs,
        ncols=ncols,
        title=grid_title,
        save_path=resolved,
        show=show_window,
    )


def _show_single(
    spec: RenderableSpec,
    *,
    config: PlotConfig | None,
    save_path: Path | str | None,
) -> None:
    if isinstance(spec, PlotSpec) and spec.wire_compatible():
        observer = get_active_observer()
        if observer is not None and _observer_supports_dataframe(observer):
            observer.dataframe(spec.name, spec.frame, meta=spec.wire_meta())
            return

    if config is None or not config.enabled:
        return

    resolved = _resolve_save_path(config, spec, save_path)
    show_window = config.show
    spec.render(save_path=resolved, show=show_window)
