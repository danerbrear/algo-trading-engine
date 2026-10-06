"""Plotting configuration DTO."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass(frozen=True)
class PlotConfig:
    """Controls whether and how plots are rendered locally."""

    enabled: bool = True
    show: bool = True
    save_dir: Optional[Path] = None
