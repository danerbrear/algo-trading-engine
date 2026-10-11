"""Route built plot specs to GUI observers or local matplotlib rendering."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from algo_trading_engine.logging import get_logger
from algo_trading_engine.logging.logger import get_active_observer
from algo_trading_engine.plotting.config import PlotConfig
from algo_trading_engine.plotting.spec import (
    BarSpec,
    HeatmapSpec,
    PlotSpec,
    RenderableSpec,
)

SpecInput = RenderableSpec | Sequence[RenderableSpec]


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


def _save_path_for_spec(
    save_path: Path | str | None,
    spec: RenderableSpec,
    count: int,
) -> Path | str | None:
    """Give each spec its own file when one path is shared across several plots."""
    if save_path is None or count == 1:
        return save_path
    path = Path(save_path)
    name = spec.name or type(spec).__name__.lower()
    suffix = path.suffix or ".png"
    return path.with_name(f"{path.stem}_{name}{suffix}")


def show_plot(
    spec_or_specs: SpecInput,
    *,
    config: PlotConfig | None,
    save_path: Path | str | None = None,
) -> None:
    """
    Display or emit plot data.

    1. Single wire-compatible PlotSpec + active observer → JSON to GUI (config ignored).
    2. Else if config is set and disabled → no-op (no matplotlib import).
    3. Else build each spec as its own matplotlib window. A missing config still shows.
       Closing one window leaves the others open.
    """
    if isinstance(spec_or_specs, (PlotSpec, HeatmapSpec, BarSpec)):
        _show_single(spec_or_specs, config=config, save_path=save_path)
        return

    specs = list(spec_or_specs)
    if not specs:
        return
    if config is not None and not config.enabled:
        get_logger().info("Plotting is disabled and will not show.")
        return

    opened = False
    for spec in specs:
        rendered = _show_single(
            spec,
            config=config,
            save_path=_save_path_for_spec(save_path, spec, len(specs)),
            show=False,
            close=False,
        )
        opened = opened or rendered
    if opened:
        from algo_trading_engine.plotting import matplotlib_backend  # pylint: disable=import-outside-toplevel

        matplotlib_backend.show_open_figures()


def _show_single(
    spec: RenderableSpec,
    *,
    config: PlotConfig | None,
    save_path: Path | str | None,
    show: bool = True,
    close: bool = True,
) -> bool:
    if isinstance(spec, PlotSpec) and spec.wire_compatible():
        observer = get_active_observer()
        if observer is not None and _observer_supports_dataframe(observer):
            observer.dataframe(spec.name, spec.frame, meta=spec.wire_meta())
            return False

    if config is not None and not config.enabled:
        get_logger().info("Plotting is disabled and will not show.")
        return False

    resolved = _resolve_save_path(config, spec, save_path)
    spec.render(save_path=resolved, show=show, close=close)
    return True
