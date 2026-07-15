import unittest
from datetime import datetime
from unittest.mock import Mock

from algo_trading_engine.core.strategy import Strategy
from algo_trading_engine.enums import UniversalCloseCondition


class MinimalStrategy(Strategy):
    def on_new_date(self, date, positions, add_position, remove_position):
        pass

    def on_end(self, positions, remove_position, date):
        pass

    def validate_data(self, data):
        return True


class TestUniversalCloseConditionsConfig(unittest.TestCase):
    def test_default_includes_all_conditions(self):
        strategy = MinimalStrategy()
        self.assertEqual(
            strategy.universal_close_conditions,
            frozenset(UniversalCloseCondition),
        )

    def test_custom_subset_excludes_others(self):
        strategy = MinimalStrategy(
            universal_close_conditions=(
                UniversalCloseCondition.ASSIGNMENT,
                UniversalCloseCondition.STOP_LOSS,
            ),
        )
        self.assertEqual(
            strategy.universal_close_conditions,
            frozenset(
                {
                    UniversalCloseCondition.ASSIGNMENT,
                    UniversalCloseCondition.STOP_LOSS,
                }
            ),
        )
        self.assertNotIn(UniversalCloseCondition.PROFIT_TARGET, strategy.universal_close_conditions)


class TestUniversalCloseConditionsGating(unittest.TestCase):
    def setUp(self):
        from algo_trading_engine.core.engine import TradingEngine

        self.engine_cls = TradingEngine

    def _make_engine_with_strategy(self, strategy):
        engine = Mock(spec=self.engine_cls)
        engine.strategy = strategy
        engine._should_close_due_to_profit_target = (
            self.engine_cls._should_close_due_to_profit_target.__get__(engine, self.engine_cls)
        )
        engine._should_close_due_to_stop = (
            self.engine_cls._should_close_due_to_stop.__get__(engine, self.engine_cls)
        )
        engine._should_close_due_to_assignment = (
            self.engine_cls._should_close_due_to_assignment.__get__(engine, self.engine_cls)
        )
        return engine

    def test_profit_target_gated_when_excluded(self):
        strategy = MinimalStrategy(
            profit_target=0.5,
            universal_close_conditions=(
                UniversalCloseCondition.ASSIGNMENT,
                UniversalCloseCondition.STOP_LOSS,
            ),
        )
        engine = self._make_engine_with_strategy(strategy)
        position = Mock()
        position.profit_target_hit = Mock(return_value=True)

        self.assertFalse(engine._should_close_due_to_profit_target(position, 0.5))

    def test_profit_target_applies_when_included(self):
        strategy = MinimalStrategy(profit_target=0.5)
        engine = self._make_engine_with_strategy(strategy)
        position = Mock()
        position.profit_target_hit = Mock(return_value=True)

        self.assertTrue(engine._should_close_due_to_profit_target(position, 0.5))


if __name__ == "__main__":
    unittest.main()
