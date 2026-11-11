"""
Velocity strategy reads the current bar from pre-loaded data (live overlay is engine-owned).
"""

import unittest
from datetime import datetime, timedelta
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd

from algo_trading_engine._internal.strategies.velocity_signal_momentum_strategy import (
    VelocitySignalMomentumStrategy,
)


class TestVelocityLivePrice(unittest.TestCase):
    def setUp(self):
        dates = pd.date_range(end=datetime.now() - timedelta(days=1), periods=60, freq="D")
        self.test_data = pd.DataFrame(
            {
                "Open": np.random.uniform(580, 600, len(dates)),
                "High": np.random.uniform(590, 610, len(dates)),
                "Low": np.random.uniform(570, 590, len(dates)),
                "Close": np.linspace(580, 600, len(dates)),
                "Volume": np.random.uniform(10_000_000, 20_000_000, len(dates)),
            },
            index=dates,
        )
        self.bar_date = self.test_data.index[-1]
        self.run_date = datetime(
            self.bar_date.year, self.bar_date.month, self.bar_date.day, 15, 0, 0
        )

    def _strategy(self) -> VelocitySignalMomentumStrategy:
        mock_options_handler = Mock()
        return VelocitySignalMomentumStrategy(
            get_contract_list_for_date=mock_options_handler.get_contract_list_for_date,
            get_option_bar=mock_options_handler.get_option_bar,
            get_options_chain=mock_options_handler.get_options_chain,
        )

    def test_get_current_bar_timestamp_snaps_to_last_bar(self):
        strategy = self._strategy()
        strategy.set_data(self.test_data.copy())
        bar_ts = strategy.get_current_bar_timestamp(self.run_date)
        self.assertEqual(bar_ts, self.bar_date.to_pydatetime())

    def test_has_buy_signal_does_not_append_rows(self):
        strategy = self._strategy()
        strategy.set_data(self.test_data.copy())
        length_before = len(strategy.data)
        with patch.object(strategy, "_check_trend_success", return_value=(False, 0, 0.0)):
            strategy._has_buy_signal(self.run_date)
        self.assertEqual(len(strategy.data), length_before)
