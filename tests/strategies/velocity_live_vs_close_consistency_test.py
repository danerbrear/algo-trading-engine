"""
Signal consistency when live and close prices match on the same bar (engine overlay).
"""

import unittest
from datetime import datetime, timedelta
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd

from algo_trading_engine._internal.strategies.velocity_signal_momentum_strategy import (
    VelocitySignalMomentumStrategy,
)


class TestVelocityLiveVsCloseConsistency(unittest.TestCase):
    def setUp(self):
        dates = pd.date_range(end=datetime.now() - timedelta(days=1), periods=90, freq="D")
        prices = np.concatenate([np.linspace(550, 570, 60), np.linspace(570, 600, 30)])
        self.historical_data = pd.DataFrame(
            {
                "Open": prices - 1,
                "High": prices + 1,
                "Low": prices - 2,
                "Close": prices,
                "Volume": np.random.uniform(10_000_000, 20_000_000, len(dates)),
            },
            index=dates,
        )
        self.bar_date = self.historical_data.index[-1]
        self.run_date = datetime(
            self.bar_date.year, self.bar_date.month, self.bar_date.day, 15, 0, 0
        )
        self.reference_close = 605.5

    def _strategy_with_today_bar(self, close: float) -> VelocitySignalMomentumStrategy:
        mock_options_handler = Mock()
        strategy = VelocitySignalMomentumStrategy(
            get_contract_list_for_date=mock_options_handler.get_contract_list_for_date,
            get_option_bar=mock_options_handler.get_option_bar,
            get_options_chain=mock_options_handler.get_options_chain,
        )
        data = self.historical_data.copy()
        data.loc[self.bar_date, "Close"] = close
        strategy.set_data(data)
        return strategy

    def test_same_close_on_bar_yields_same_signal(self):
        strategy_a = self._strategy_with_today_bar(self.reference_close)
        strategy_b = self._strategy_with_today_bar(self.reference_close)
        with patch.object(strategy_a, "_check_trend_success", return_value=(True, 5, 0.03)):
            signal_a = strategy_a._has_buy_signal(self.run_date)
        with patch.object(strategy_b, "_check_trend_success", return_value=(True, 5, 0.03)):
            signal_b = strategy_b._has_buy_signal(self.run_date)
        self.assertEqual(signal_a, signal_b)
