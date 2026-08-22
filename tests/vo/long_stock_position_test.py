"""Unit tests for LongStockPosition."""

from datetime import datetime

import pytest

from algo_trading_engine.common.models import StrategyType
from algo_trading_engine.vo.position import LongStockPosition, create_position


def _stock_position(entry_price: float = 100.0, quantity: float = 2.5) -> LongStockPosition:
    position = create_position(
        symbol="SPY",
        expiration_date=None,
        strategy_type=StrategyType.LONG_STOCK,
        strike_price=0.0,
        entry_date=datetime(2026, 1, 15, 10, 0),
        entry_price=entry_price,
        spread_options=[],
    )
    position.set_quantity(quantity)
    return position


def test_long_stock_contract_multiplier_and_legs():
    position = _stock_position()
    assert position.contract_multiplier() == 1
    assert position.uses_option_legs() is False
    assert position.is_expired_for_assignment(datetime(2026, 2, 1)) is False


def test_long_stock_pnl():
    position = _stock_position(entry_price=100.0, quantity=2.5)
    assert position.get_return_dollars(110.0) == pytest.approx(25.0)
    assert position._get_return(110.0) == pytest.approx(0.10)


def test_long_stock_max_risk_per_share():
    position = _stock_position(entry_price=50.0, quantity=3.0)
    assert position.max_risk_dollars_per_contract() == pytest.approx(50.0)


def test_long_stock_exit_price_from_underlying():
    position = _stock_position()
    assert position.calculate_exit_price_from_bars(None, None, 105.5) == pytest.approx(105.5)


def test_long_stock_assignment_raises():
    position = _stock_position()
    with pytest.raises(RuntimeError, match="assignment"):
        position.get_return_dollars_from_assignment(100.0)


def test_create_position_long_stock_factory():
    position = create_position(
        symbol="SPY",
        expiration_date=None,
        strategy_type=StrategyType.LONG_STOCK,
        strike_price=0.0,
        entry_date=datetime(2026, 1, 15),
        entry_price=420.0,
    )
    assert isinstance(position, LongStockPosition)
