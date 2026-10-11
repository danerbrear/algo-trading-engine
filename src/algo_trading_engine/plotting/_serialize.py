"""Serialize plot specs to GUI-compatible JSON payloads."""

from __future__ import annotations

import json
from io import StringIO
from typing import Any

import pandas as pd

from algo_trading_engine.plotting.spec import EQUITY_CURVE_NAME, PlotSpec


def _coerce_equity_curve(frame: pd.DataFrame) -> pd.DataFrame:
    coerced = frame.copy()
    coerced["timestamp"] = pd.to_datetime(coerced["timestamp"])
    coerced["equity"] = coerced["equity"].astype(float)
    return coerced.loc[:, ["timestamp", "equity"]]


def coerce_plot_frame(name: str, frame: pd.DataFrame) -> pd.DataFrame:
    """Validate and coerce a plot frame for wire transport."""
    if name == EQUITY_CURVE_NAME:
        missing = [col for col in ("timestamp", "equity") if col not in frame.columns]
        if missing:
            raise ValueError(
                f"DataFrame '{name}' missing columns: {', '.join(missing)}"
            )
        return _coerce_equity_curve(frame)

    if "timestamp" not in frame.columns:
        raise ValueError(f"DataFrame '{name}' missing column: timestamp")
    coerced = frame.copy()
    coerced["timestamp"] = pd.to_datetime(coerced["timestamp"])
    for column in coerced.columns:
        if column != "timestamp":
            coerced[column] = pd.to_numeric(coerced[column], errors="coerce").astype(float)
    return coerced


def plot_payload(spec: PlotSpec) -> dict[str, Any]:
    """Build a dataframe message payload from a PlotSpec."""
    validated = coerce_plot_frame(spec.name, spec.frame)
    payload: dict[str, Any] = {
        "name": spec.name,
        "orient": "split",
        "data": json.loads(validated.to_json(orient="split", date_format="iso")),
    }
    if spec.title:
        payload["title"] = spec.title
    if spec.y_label:
        payload["y_label"] = spec.y_label
    if spec.right_axis:
        payload["right_axis"] = list(spec.right_axis)
    if spec.markers:
        payload["markers"] = {
            key: json.loads(coerce_plot_frame(spec.name, marker).to_json(orient="split", date_format="iso"))
            for key, marker in spec.markers.items()
        }
    if spec.meta:
        payload["meta"] = spec.meta
    return payload


def payload_to_plot_frame(payload: dict[str, Any]) -> pd.DataFrame:
    """Restore a plot DataFrame from a wire payload."""
    if payload.get("orient") != "split":
        raise ValueError(f"Unsupported dataframe orient: {payload.get('orient')}")
    name = payload.get("name", "")
    frame = pd.read_json(StringIO(json.dumps(payload["data"])), orient="split")
    return coerce_plot_frame(name, frame)


def payload_markers(payload: dict[str, Any]) -> dict[str, pd.DataFrame]:
    """Restore marker DataFrames from optional payload metadata."""
    raw = payload.get("markers")
    if not raw:
        return {}
    name = payload.get("name", "")
    return {
        key: coerce_plot_frame(name, pd.read_json(StringIO(json.dumps(data)), orient="split"))
        for key, data in raw.items()
    }
