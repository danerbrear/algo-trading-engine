# Fix: Paper Trading Indicator Warm-Up

## Problem

When running strategies via `PaperTradingEngine` (e.g. in a Lambda), indicators only have a stored value for the single bar that `on_new_date` is called with. Strategies that look up indicator values at historical bar dates (e.g. ATR at a swing low K bars ago) get `None` and silently skip signals.

### Observed Error

```
2026-05-19 17:30:39 | WARNING | ...uptrend_swing:_is_local_minimum:505 | ⚠️ ATR not available or invalid at 2026-05-19 09:30:00
```

The strategy called `on_new_date` at 13:30 (current bar). It then looked up ATR at 09:30 (the swing low bar, K_BARS_AGO=4 bars back). Since the ATR indicator was only updated once — at 13:30 — no value existed at 09:30.

### Root Cause

`BacktestEngine.run()` iterates every bar in the date range, calling `on_new_date` for each. Each call triggers `Strategy._update_indicators(date)`, which calls `indicator.update(date, data)`. This means indicators accumulate a stored value at every bar datetime.

```python
# backtest/main.py — BacktestEngine.run()
for i, date in enumerate(date_range):
    self.strategy.on_new_date(date, ...)  # indicators updated for EVERY bar
```

`PaperTradingEngine.run()` calls `on_new_date` exactly once for the current bar via the recommendation engine. Indicators get a single `update()` call and store a single value.

```python
# core/engine.py — PaperTradingEngine.run()
recommender.run(run_date)  # on_new_date called ONCE for current bar
```

The `Indicator.get_value_at(date)` method does an exact datetime match against stored values. With only one stored value, any historical lookup returns `None`:

```python
# core/indicators/indicator.py
def get_value_at(self, date: datetime) -> float:
    if date in self._values.index:
        return self._values.loc[date]
    return None  # <-- every historical bar hits this path in paper trading
```

## Proposed Fix

### 1. Add `warm_up_indicators()` to `Strategy` (`core/strategy.py`)

Add a new public method that replays all historical bars through each indicator so they store values at every datetime, matching backtest behavior.

```python
def warm_up_indicators(self) -> None:
    """Replay historical bars through all indicators so they have values
    at every datetime, matching backtest behavior.

    Should be called after set_data() and before the first on_new_date() in
    contexts where the engine does not iterate through every bar (e.g. paper
    trading, Lambda).
    """
    if self.data is None or self.data.empty or not self.indicators:
        return
    for date in self.data.index:
        for indicator in self.indicators:
            indicator.update(date, self.data)
    get_logger().info(
        f"Indicator warm-up complete: {len(self.indicators)} indicator(s), "
        f"{len(self.data)} bar(s)"
    )
```

Insert after `_update_indicators` (around line 298) in `core/strategy.py`.

### 2. Call it in `PaperTradingEngine.from_config()` (`core/engine.py`)

After setting the data on the strategy and before creating the engine instance, warm up indicators:

```python
strategy.set_data(data, retriever.treasury_rates)

# Warm up indicators so they have values at every historical bar,
# matching the backtest engine's behavior.
strategy.warm_up_indicators()

engine = cls(
    strategy=strategy,
    config=config,
    options_handler=options_handler
)
```

This is a one-line insertion at line 533 of `core/engine.py`.

### 3. Skip redundant update in `_update_indicators`

With warm-up already done, the `_update_indicators` call inside `on_new_date` will re-process the current bar. This is harmless (ATR uses Wilder's smoothing, SMA recomputes) but if you want to avoid the duplicate:

```python
def _update_indicators(self, date: datetime) -> bool:
    for indicator in self.indicators:
        try:
            if date in indicator._values.index:
                continue  # already warm — skip
            indicator.update(date, self.data)
        except Exception as e:
            get_logger().error(f"Error updating indicator {indicator.name}: {e}")
            return False
    return True
```

## Files Changed

| File | Change |
|------|--------|
| `core/strategy.py` | Add `warm_up_indicators()` method |
| `core/engine.py` | Call `strategy.warm_up_indicators()` in `PaperTradingEngine.from_config()` |

## Verification

After the fix, running the uptrend swing strategy in Lambda should show ATR values available at all historical bar dates. The warning `ATR not available or invalid at ...` should no longer appear.

To verify:
1. Deploy the updated engine
2. Run the Lambda and confirm ATR lookups at swing low/high dates succeed
3. Run the backtest suite to confirm no regressions (warm-up is a no-op when `on_new_date` already iterates every bar)
