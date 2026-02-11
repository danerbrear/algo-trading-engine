# Logging Improvement Plan

This document outlines the plan for improving logging across the BacktestEngine and TradingEngine (paper trading / recommendation flow) according to the following requirements.

---

## Requirements Summary

| Component       | Singleton logger | Output destination | Stdout behavior |
|----------------|------------------|--------------------|-----------------|
| **BacktestEngine** | All logs → singleton | `backtest.log` only (overwritten each run) | Stdout = **progress bar only**; all other output → log file |
| **TradingEngine**  | All logs → singleton | `trade.log` only (overwritten each run)   | Only recommendation-relevant messages to stdout |

---

## Implementation Checklist

Use this checklist to track what is done vs. remaining. Based on current working directory state.

### Completed

- [x] **Singleton logger (Step 1)** — `common/logger.py` added with Loguru; `configure_logger(run_type, log_dir, log_level)`, `get_logger()`, `log_and_echo()`. `log_level` is `"debug"`, `"info"`, or `"warn"`. File sink only; `logger.remove(None)` on configure so repeated calls work (e.g. tests).
- [x] **BacktestEngine (Step 2)** — At start of `run()`, `configure_logger("backtest", log_level=...)` (derived from `quiet_mode`). All `print()` in `backtest/main.py` replaced with `get_logger().info(...)` / `.warning(...)` / `.error(...)`. CLI `main()` keeps `print()` for startup and result lines only.
- [x] **Logger unit tests** — `tests/common/logger_test.py` covers configure_logger, get_logger, log_and_echo, log levels, invalid log_level, overwrite on reconfigure.

### Remaining

- [ ] **ProgressTracker (Section 4)** — `progress_print()` still writes to stdout (via `tracker.write()` or `print()`). Should be changed to log to file only (`get_logger().debug()` / `.info()` by force). `ProgressTracker.close()` still uses `print()` for "Processing completed" / timing; should use `get_logger().info(...)`.
- [ ] **Engine core (Step 3)** — `core/engine.py` still uses `print()` in `get_current_volumes_for_position`, `compute_exit_price`, `check_univeral_close_conditions`, and PaperTradingEngine `run()`. Replace with `get_logger().info(...)` / `.warning(...)` / `.error(...)`.
- [ ] **TradingEngine / recommendation (Step 4)** — At start of recommendation run, call `configure_logger("trade", log_level=...)`. Use `get_logger()` for all non–recommendation logs. Use `log_and_echo()` only for recommendation-relevant lines (open positions, run date, capital summary, final outcome).
- [ ] **Strategies and options_handler** — After ProgressTracker is updated, strategies and `options_handler` that use `progress_print` will automatically log to file. No change required unless adding new log sites. (`logs/` is already in `.gitignore`.)

### Validation (Section 7)

- [ ] Run backtest; confirm `logs/backtest.log` has engine/run content; confirm stdout shows only progress bar (once ProgressTracker is updated).
- [ ] Run recommendation flow; confirm `logs/trade.log` has full detail and stdout shows only recommendation-relevant lines (after Step 4).

---

## 1. Singleton Logger (Loguru)

**Library:** [Loguru](https://github.com/Delgan/loguru) (`loguru>=0.7.0` in pyproject.toml). Single import, no handler boilerplate, easy file sink with `mode="w"` for overwrite-each-run.

### 1.1 Location and API

- **Module:** `src/algo_trading_engine/common/logger.py`.
- **API:**
  - `configure_logger(run_type: "backtest" | "trade", log_dir: str = "logs", log_level: str = "info")` — call once at start of a run; removes default stderr handler and adds a single file sink (overwrite). **log_level** controls log file verbosity only (not stdout); one of `"debug"`, `"info"`, or `"warn"`.
  - `get_logger()` — returns the Loguru logger instance for use everywhere.
  - `log_and_echo(message: str)` — logs to the current log file and prints to stdout; use only for recommendation-relevant messages in the trading flow.
- **Behavior:**
  - File sink only; no stdout from the logger. Stdout is only the progress bar (tqdm) during backtest; for trade, use `log_and_echo` for the few recommendation-relevant lines.

### 1.2 Configuration

- **Log file names (fixed, overwritten each run):**
  - **Backtest:** `backtest.log`
  - **Trading / recommendation:** `trade.log`
- **Directory:** `log_dir` argument (default `"logs"`) → full paths `logs/backtest.log`, `logs/trade.log`; directory is created if missing.
- **log_level:** One of `"debug"`, `"info"`, or `"warn"`. Controls what is written to the log file only (not stdout). Default `"info"`.
- **Format:** `{time:YYYY-MM-DD HH:mm:ss} | {level} | {name}:{function}:{line} | {message}` (set in `logger.add(...)`).

---

## 2. BacktestEngine

### 2.1 Rules

1. **All logs** from the backtest engine (and code it calls, including strategies and options_handler) **go through the singleton logger** and are written **only to the log file** (no stdout).
2. **ProgressTracker is only responsible for the progress bar:** the tqdm bar is the only stdout from progress. `progress_print(message)` no longer writes to stdout; it writes to the **log file** only (as DEBUG when `force=False`, INFO when `force=True`). So stdout during backtest = progress bar only.
3. **log_level** controls **log file verbosity only**: `configure_logger(..., log_level="info")` (default), `"debug"`, or `"warn"`. It does not affect what is printed to stdout (only the bar is on stdout).

### 2.2 Current `print()` Call Sites to Migrate (backtest/main.py)

Replace with logger calls (e.g. `logger.info(...)` / `logger.warning(...)`); ensure no handler attached to the singleton writes to stdout.

- Abort / validation: invalid data message.
- Run summary: date range, “Running backtest on N days”.
- _end: “Closing backtest”, last trading date, last closing price.
- Volume stats: volume validation statistics block.
- Results summary: benchmark return, trading days, total positions, final capital, total return, Sharpe ratio.
- _add_position: volume validation failed, not enough capital, net credit added, “Adding position”.
- _remove_position: volume validation failed, skipping closure, position closed by assignment, cost to close, “Position closed” block.
- _print_position_statistics / _print_overall_statistics / _print_strategy_statistics: all printed lines.
- _handle_insufficient_volume_closure: skip/proceed messages.
- CLI (e.g. `if __name__ == "__main__"`): “Starting backtest”, date range, symbol, capital, stop loss/profit target, success/failure, errors.

All of the above should go to the **singleton logger → log file**; none to stdout (so that stdout stays clean for progress only).

### 2.3 Engine / Strategy Code Used by Backtest

- **core/engine.py** (used by BacktestEngine for `check_univeral_close_conditions`, `get_current_volumes_for_position`, `compute_exit_price`): replace `print()` with the singleton logger so that when called from backtest, those logs go to the backtest log file.
- **Strategies** (e.g. velocity_signal_momentum_strategy, credit_spread_minimal): already use `progress_print`. **progress_print now writes to the log file only** (no stdout). So all those messages automatically go to the log file; no code change required beyond ensuring `configure_logger("backtest", log_level=...)` is called at run start. Use `force=True` for messages that should appear in the log file even when log_level is `"warn"` (they log at INFO).

---

## 3. TradingEngine (Paper Trading / Recommendation Flow)

### 3.1 Rules

1. **All logs** from the recommendation flow go through the **singleton logger** and are written to **trade.log**.
2. **Stdout:** only messages **relevant to the recommendation** are printed to stdout (e.g. open positions summary, run date, capital status, final recommendation outcome). All other messages (debug, volume fetches, option bar data, etc.) go only to the log file.

### 3.2 “Recommendation-relevant” (Stdout) vs “All logs” (File)

- **To stdout (recommendation-relevant):**
  - Run date and high-level status (e.g. “Running recommendation flow for &lt;date&gt;”).
  - Open positions summary (count, status lines).
  - Capital manager status summary (e.g. `get_status_summary`).
  - Final outcome (e.g. recommendation result, errors that affect the user’s decision).

- **To log file only (via singleton logger):**
  - Options handler: “No option bar data”, “Fetched volume data”, “No volume data”, “Error fetching volume”, “Error computing exit price”.
  - Universal close: “Position expired”, “Profit target hit”, “Stop loss hit”.
  - Errors: “Options handler not available”, “Failed to load capital allocation config”, “Failed to run recommendation engine”.
  - Any other debug/info that is not needed for the user’s immediate recommendation decision.

Implementation approach: use the singleton logger for everything; add a small helper or a second handler that echoes only “recommendation-relevant” messages to stdout (e.g. a custom handler or a `log_and_echo_recommendation(message)` that logs to file and optionally prints). Alternatively, keep explicit `print()` only for the few recommendation-relevant lines and log everything (including those) with the singleton so the file has the full record.

### 3.3 Current `print()` Call Sites (engine.py – PaperTradingEngine / run)

- Recommendation-relevant (keep or mirror to stdout): open positions count, “Open position status” block, run date, capital status summary.
- Log file only: options handler not available, failed to load capital config, volume/bar messages, universal close messages, “Failed to run recommendation engine” (can be both: log + stdout for visibility).

---

## 4. ProgressTracker Behavior

- **progress_tracker.py:** ProgressTracker is **only responsible for the progress bar** (tqdm). The bar continues to write to **stdout** (description, postfix, completion).
- **progress_print(message, force=False):** No longer writes to stdout. It writes to the **log file** only via the singleton logger: `force=False` → `logger.debug(...)` (only visible when log level is DEBUG); `force=True` → `logger.info(...)`. So all existing `progress_print` call sites now feed the log file; stdout stays clean except for the bar.
- **ProgressTracker.close():** The "Processing completed" / timing summary is no longer printed to stdout; it is written to the log file via `get_logger().info(...)`.
- **log_level:** Pass `log_level` into `configure_logger(..., log_level="debug"|"info"|"warn")` to control **log file verbosity** only (nothing from the logger goes to stdout).

---

## 5. Implementation Order

1. **Singleton logger (done):** `common/logger.py` uses Loguru; `configure_logger(run_type="backtest"|"trade", log_dir="logs", level="INFO")`, `get_logger()`, `log_and_echo()` for recommendation-relevant stdout in trade flow.
2. **BacktestEngine:** At start of `run()`, call `configure_logger("backtest", log_level=...)` (e.g. from engine config or default `"info"`). Replace all `print()` in `backtest/main.py` with `get_logger().info(...)` / `.warning(...)`. ProgressTracker still only shows the bar; `progress_print` already goes to the log file.
3. **Engine (core/engine.py):** Replace `print()` with `get_logger().info(...)` / `.warning(...)`. When called from backtest, log file is `backtest.log`; when called from recommendation, call `configure_logger("trade")` at start of recommendation run so log file is `trade.log`.
4. **TradingEngine (recommendation):** At start of run, call `configure_logger("trade")`. Use `get_logger()` for all logs; use `log_and_echo()` only for the few recommendation-relevant lines that must appear on stdout.
5. **Strategies / options_handler:** Replace or supplement `progress_print` with `get_logger().info(...)` where the message should appear in the log file; keep `progress_print` for stdout where progress is desired.

---

## 6. Log File Paths and Naming

- **Fixed filenames, overwritten each run:**
  - **Backtest:** `backtest.log` (e.g. under `logs/` → `logs/backtest.log`).
  - **Trading / recommendation:** `trade.log` (e.g. under `logs/` → `logs/trade.log`).
- Directory (e.g. `logs/`) can be configurable; ensure it exists or is created. Each run overwrites the same file.
- Consider adding `logs/` to `.gitignore` if not already.

---

## 7. Testing and Validation

- **Backtest:** Run a short backtest; confirm log file contains all former print content and stdout shows only progress bar (and any forced progress_print).
- **Recommendation:** Run recommendation flow; confirm log file has full detail and stdout shows only recommendation-relevant lines.
- **Unit tests:** Optionally add tests that configure the logger to a temp file and assert expected log lines appear (and, if desired, that no unexpected stdout occurs when capturing stdout).

---

## 8. Summary

| Area                  | Action |
|-----------------------|--------|
| **Singleton logger**  | **Done.** `common/logger.py` using Loguru; `configure_logger(run_type, log_dir, log_level="info")`; fixed filenames `backtest.log` / `trade.log` (overwritten each run). **log_level** is `"debug"`, `"info"`, or `"warn"` and affects log file verbosity only. Use `log_and_echo()` in trade flow for recommendation-relevant stdout. |
| **BacktestEngine**    | Call `configure_logger("backtest", log_level=...)` at start of run; replace all `print()` → `get_logger().info(...)` / `.warning(...)`. Stdout = progress bar only. |
| **TradingEngine**    | Call `configure_logger("trade", log_level=...)` at start of run; all logs → `get_logger()` → `trade.log`; use `log_and_echo()` only for recommendation-relevant stdout. |
| **ProgressTracker**  | **Done.** Only shows the progress bar on stdout. `progress_print` → log file only (DEBUG/INFO by force). `close()` summary → log file. Pass **log_level** into `configure_logger` for log file verbosity. |
