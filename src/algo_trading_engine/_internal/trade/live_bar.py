"""Apply a live equity quote onto the current OHLCV bar in a paper-trading frame."""

from __future__ import annotations

from datetime import datetime, time
import pandas as pd

from algo_trading_engine.enums import BarTimeInterval
from algo_trading_engine.logging import get_logger

_REGULAR_OPEN = time(9, 30)
_REGULAR_CLOSE = time(16, 0)


def _in_regular_session(now: datetime) -> bool:
    if now.weekday() >= 5:
        return False
    t = now.time()
    return _REGULAR_OPEN <= t < _REGULAR_CLOSE


def apply_live_quote(
    data: pd.DataFrame,
    price: float,
    now: datetime,
    bar_interval: BarTimeInterval,
) -> pd.DataFrame:
    """
    Set the live quote on the current bar's close and widen high/low when needed.

    If the bar is missing, append a single bar at the interval's bar start.
    """
    if data is None or data.empty:
        return data
    if not _in_regular_session(now):
        get_logger().debug("Skipping live quote overlay outside regular session")
        return data

    bar_ts = bar_interval.bar_start(now)
    quote = float(price)
    result = data.copy()

    if bar_ts in result.index:
        row = result.loc[bar_ts]
        result.at[bar_ts, "Close"] = quote
        result.at[bar_ts, "High"] = max(float(row["High"]), quote)
        result.at[bar_ts, "Low"] = min(float(row["Low"]), quote)
        get_logger().info(
            f"Live quote ${quote:.2f} applied to bar {bar_ts} (close, high/low adjusted)"
        )
        return result

    new_row = pd.DataFrame(
        {
            "Open": [quote],
            "High": [quote],
            "Low": [quote],
            "Close": [quote],
            "Volume": [0],
        },
        index=[bar_ts],
    )
    result = pd.concat([result, new_row]).sort_index()
    get_logger().info(f"Live quote ${quote:.2f} appended as new bar at {bar_ts}")
    return result
