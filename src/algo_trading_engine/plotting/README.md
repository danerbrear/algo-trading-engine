# Plotting

Build chart data first, then show it. Plotting stays optional: matplotlib is only imported when local rendering runs.

## Two-step pattern

```python
from algo_trading_engine.plotting import PlotConfig, build_plot_spec, show_plot

spec = build_plot_spec(
    frame,
    name="equity_curve",
    title="Equity Curve",
    y_label="Equity ($)",
)
show_plot(spec, config=self.plot_config)
```

- `build_plot_spec` is pure (no matplotlib, no observer).
- `show_plot` is the only display entry point.

Pass a sequence of specs to `show_plot` for a multi-panel figure (`ncols` controls layout).

## Enabling plots in backtests

```python
from algo_trading_engine import BacktestConfig
from algo_trading_engine.plotting import PlotConfig

config = BacktestConfig(
    ...,
    plot_config=PlotConfig(enabled=True),
)
```

- `plot_config=None` (default): no local matplotlib import and no windows.
- `PlotConfig.enabled=False`: build specs in your code, but do not build or show a matplotlib figure.
- `PlotConfig.enabled=True`: build the figure and show it.
- `PlotConfig.save_dir`: when set, each spec also saves to `<save_dir>/<name>.png`.

The engine calls `strategy.set_plot_config(config.plot_config)` so strategies can use `self.plot_config`.

## PlotSpec layout

- One line per non-x column in the DataFrame.
- `x` defaults to `"timestamp"`; use `"index"`, `"epoch"`, or `"Date"` for other charts.
- `right_axis`: column names plotted on twin y-axes (e.g. SPY overlay on an equity curve).
- `markers`: `{label: DataFrame}` scatter overlays sharing the same `x` column.
- `y_ticks`: categorical y-axis labels, e.g. signal class names.
- Horizontal reference lines: add a constant column (e.g. initial capital).

Use `HeatmapSpec` for confusion-matrix style charts.

## GUI runs

When a run observer is active (`ALGO_GUI_RUN_ID`), a single wire-compatible `PlotSpec` (`x="timestamp"`) is sent to the host as JSON. No local window opens, and `PlotConfig` is not consulted on that path. Grids, heatmaps, and non-timestamp x values render locally only.

## Install

Local rendering requires the plot extra:

```bash
pip install algo-trading-engine[plot]
```

Without matplotlib, enabled plotting raises an `ImportError` with this hint.
