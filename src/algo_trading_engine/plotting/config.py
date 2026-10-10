"""Plotting configuration DTO."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass(frozen=True)
class PlotConfig:
    """Controls whether plots are built and shown locally with matplotlib."""

    enabled: bool = True
    save_dir: Optional[Path] = None
