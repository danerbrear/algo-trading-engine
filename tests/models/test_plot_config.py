"""Tests for BacktestConfig.plot_config wiring."""

from datetime import datetime

from algo_trading_engine.models.config import BacktestConfig
from algo_trading_engine.plotting import PlotConfig
from algo_trading_engine.core.strategy import Strategy


class _MinimalStrategy(Strategy):
    def on_new_date(self, date, positions, add_position, remove_position):
        pass

    def on_end(self, positions, remove_position, date):
        pass

    def validate_data(self, data):
        return True


def test_backtest_config_plot_config_default_is_none():
    config = BacktestConfig(
        initial_capital=1000.0,
        start_date=datetime(2024, 1, 1),
        end_date=datetime(2024, 6, 1),
        symbol="SPY",
        strategy_type="velocity_momentum",
    )
    assert config.plot_config is None


def test_strategy_set_plot_config():
    strategy = _MinimalStrategy()
    assert strategy.plot_config is None
    plot_config = PlotConfig(enabled=True, show=False)
    strategy.set_plot_config(plot_config)
    assert strategy.plot_config == plot_config
