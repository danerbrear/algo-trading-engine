"""Plotting configuration DTO."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

from algo_trading_engine.plotting.spec import RenderableSpec


@dataclass(frozen=True)
class PlotConfig:
    """Controls whether plots are built and shown locally with matplotlib."""

    # Determines if a plot can be built or shown
    enabled: bool = True

    # Optional set of additional graphs to show
    specs: Optional[Sequence[RenderableSpec]] = None

    # Where to save the plots to
    save_dir: Optional[Path] = None
