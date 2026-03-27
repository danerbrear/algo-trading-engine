# Warm-Up Period: Replacing `start_date_offset`

## Problem

The current `start_date_offset` is a static integer passed through the strategy constructor and builder chain. It represents a hardcoded number of rows to skip at the beginning of a backtest, but it has no relationship to the actual data requirements of the strategy's indicators. Each strategy picks an arbitrary offset (e.g. 60) that may be too large (wasting data) or too small (producing NaN-based signals).

## Goal

Replace `start_date_offset` with a dynamically calculated warm-up period derived from the indicators attached to the strategy. The warm-up period represents the minimum number of bars each indicator needs before it can produce a valid value.

---

## Implementation Steps

### Step 1: Add `warm_up_period` as an abstract property on `Indicator`

**File:** `src/algo_trading_engine/core/indicators/indicator.py`

- Add an abstract property `warm_up_period -> int` to the `Indicator` ABC.
- This property returns the minimum number of bars the indicator needs before it can produce a valid output.

### Step 2: Implement `warm_up_period` on each concrete indicator

**File:** `src/algo_trading_engine/core/indicators/sma_indicator.py`

- `SMAIndicator.warm_up_period` returns `self.period` (e.g. SMA_20 needs 20 bars).

**File:** `src/algo_trading_engine/core/indicators/average_true_return_indicator.py`

- `ATRIndicator.warm_up_period` returns `self.period + 1` (ATR needs `period` true-range values, each of which requires a previous close, so `period + 1` bars total).

### Step 3: Add `warm_up_period` property to `Strategy`

**File:** `src/algo_trading_engine/core/strategy.py`

- Add a property `warm_up_period -> int` that returns the maximum `warm_up_period` across all indicators in `self.indicators`.
- Returns `0` if the strategy has no indicators (e.g. `CreditSpreadStrategy` which has no `Indicator` instances).
- This replaces the role of `start_date_offset` — the strategy's warm-up is determined solely by its most demanding indicator.

### Step 4: Remove `start_date_offset` from `Strategy.__init__`

**File:** `src/algo_trading_engine/core/strategy.py`

- Remove the `start_date_offset` parameter from `__init__`.
- Remove `self.start_date_offset` instance attribute.
- All usages of `self.start_date_offset` will be replaced with `self.warm_up_period` (or handled by the engine — see Step 6).

### Step 5: Remove `start_date_offset` from `VelocitySignalMomentumStrategy`

**File:** `src/algo_trading_engine/strategies/velocity_signal_momentum_strategy.py`

- Remove `start_date_offset` from the constructor signature and `super().__init__()` call.
- Migrate the manual SMA_15 and SMA_30 calculations in `set_data` to use `SMAIndicator(15)` and `SMAIndicator(30)` via `add_indicator()` in `__init__`. This way `warm_up_period` will correctly return `30`. The velocity/signal columns can still be derived from the indicator values.

**Note:** `CreditSpreadStrategy` is excluded from this change. Its warm-up needs are driven by LSTM data fetching (`lstm_start_date_offset` on `BacktestConfig`), which is a separate concern from indicator warm-up. It will simply remove `start_date_offset` from its constructor and rely on the engine's warm-up logic (which will be `0` since it has no `Indicator` instances).

### Step 6: Update `BacktestEngine.run()` to use `warm_up_period`

**File:** `src/algo_trading_engine/backtest/main.py`

Replace all references to `self.strategy.start_date_offset` with `self.strategy.warm_up_period`:

- **Benchmark start price** (line 199): `self.data.iloc[self.strategy.warm_up_period]['Close']`
- **Progress tracking** (lines 203-205): Use `self.strategy.warm_up_period` for `effective_start_date` and `effective_total_dates`.
- **Progress update guard** (line 230): `if self.progress_tracker and i >= self.strategy.warm_up_period:`

### Step 7: Update `CreditSpreadStrategy` to remove `start_date_offset`

**File:** `src/algo_trading_engine/strategies/credit_spread_minimal.py`

- Remove `start_date_offset` from the constructor signature and `super().__init__()` call.
- Remove the early-return warm-up guard (`if date.date() < self.data.index[self.start_date_offset].date()`). Since `CreditSpreadStrategy` has no `Indicator` instances, its `warm_up_period` will be `0` and the engine will not skip any dates for it. The strategy's LSTM data needs are already handled separately by `lstm_start_date_offset` on `BacktestConfig`, which controls how much extra historical data is fetched.
- No other changes to this strategy.

### Step 8: Remove `start_date_offset` from the builder chain

**Files:**
- `src/algo_trading_engine/backtest/strategy_builder.py`
  - Remove `set_start_date_offset()` from `StrategyBuilder` ABC and all concrete builders.
  - Remove `self._start_date_offset` from builder `reset()` methods.
  - Remove `start_date_offset=...` from `build()` calls.
- `src/algo_trading_engine/backtest/strategy_builder.py` (`create_strategy_from_args`)
  - Remove the `start_date_offset` kwarg handling.

### Step 9: Remove `start_date_offset` from config and CLI

**Files:**
- `src/algo_trading_engine/models/config.py` — `BacktestConfig.lstm_start_date_offset` is a separate concept (days of extra historical data for LSTM training) and should remain, but rename it to clarify it is not the same as the warm-up offset. Alternatively, keep as-is if it's only used for LSTM data fetching.
- `src/algo_trading_engine/backtest/main.py` — CLI argument parser: remove `--start-date-offset` argument if it exists.
- `docs/strategy_builder_guide.md` and `src/algo_trading_engine/strategies/README.md` — Update documentation.

### Step 10: Update `PaperTradingEngine` data fetching

**File:** `src/algo_trading_engine/core/engine.py`

- In `PaperTradingEngine.from_config()`, after the strategy is created and indicators are attached, use `strategy.warm_up_period` to determine how much extra historical data to fetch.
- Currently it fetches 120 days back (hardcoded for LSTM). Instead, fetch enough data to satisfy both the indicator warm-up and any LSTM needs: e.g. `max(strategy.warm_up_period, 120)` days before the run date.

### Step 11: Write unit tests

**Files:**
- `tests/core/indicators/sma_indicator_test.py` — Add test for `warm_up_period` property.
- `tests/core/indicators/atr_indicator_test.py` — Add test for `warm_up_period` property.
- `tests/core/test_strategy.py` (new or existing) — Test `Strategy.warm_up_period` with zero indicators, one indicator, and multiple indicators (should return max).
- `tests/backtest/backtest_engine_factory_test.py` — Update mocks from `start_date_offset` to `warm_up_period`.
- `tests/strategies/velocity_vertical_spread_test.py` and `tests/strategies/strategy_enhancements_test.py` — Remove `start_date_offset` from strategy construction and verify warm-up behavior.
- `tests/test_cli_integration.py` — Remove `start_date_offset` assertions.

---

## Design Decision: Who Skips Warm-Up Dates?

**Current behavior:** The engine iterates all dates and strategies individually decide whether to skip early dates (e.g. `CreditSpreadStrategy` returns early if before offset). The engine only uses the offset for benchmark and progress tracking.

**Recommended approach:** Keep the same pattern — the engine iterates all dates (so indicators get updated from bar 0), but the engine skips calling `on_new_date` for dates within the warm-up window. This centralizes the skip logic:

```python
for i, date in enumerate(date_range):
    self.strategy._update_indicators(date)
    if i < self.strategy.warm_up_period:
        continue
    self.strategy.on_new_date(date, ...)
```

This means:
- Indicators get fed data from bar 0 (they need it to build up their rolling windows).
- Strategy trading logic only fires after warm-up is complete.
- Individual strategies no longer need their own warm-up guards.

**Note:** This changes the current contract where `on_new_date` calls `_update_indicators` internally. The engine would need to call `_update_indicators` before the warm-up skip check, and `on_new_date` should no longer call it (or it should be idempotent).

---

## Summary of Files Changed

| File | Change |
|------|--------|
| `core/indicators/indicator.py` | Add abstract `warm_up_period` property |
| `core/indicators/sma_indicator.py` | Implement `warm_up_period` |
| `core/indicators/average_true_return_indicator.py` | Implement `warm_up_period` |
| `core/strategy.py` | Add `warm_up_period` property, remove `start_date_offset` |
| `strategies/credit_spread_minimal.py` | Remove `start_date_offset` and warm-up guard only |
| `strategies/velocity_signal_momentum_strategy.py` | Remove `start_date_offset`, migrate to `add_indicator()` |
| `backtest/main.py` | Replace `start_date_offset` with `warm_up_period`, update run loop |
| `backtest/strategy_builder.py` | Remove `set_start_date_offset` from all builders |
| `core/engine.py` | Use `warm_up_period` for data fetching in `PaperTradingEngine` |
| `models/config.py` | No change (lstm_start_date_offset is separate) |
| `strategies/README.md` | Update documentation |
| `docs/strategy_builder_guide.md` | Update documentation |
| Tests (multiple) | Update to remove `start_date_offset`, add `warm_up_period` tests |
