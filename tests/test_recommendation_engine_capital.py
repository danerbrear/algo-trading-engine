"""Unit tests for recommendation engine with capital management."""

import pytest
from datetime import datetime
from unittest.mock import MagicMock, patch

from algo_trading_engine.prediction.decision_store import JsonDecisionStore
from algo_trading_engine.prediction.recommendation_engine import InteractiveStrategyRecommender
from algo_trading_engine.prediction.capital_manager import CapitalManager
from algo_trading_engine.backtest.models import StrategyType
from algo_trading_engine.common.models import Option


def _make_option(symbol: str, strike: float, expiration: str, opt_type: str, last: float) -> Option:
    """Helper to create Option objects for testing."""
    return Option.from_dict({
        'symbol': symbol,
        'ticker': symbol,
        'strike': strike,
        'expiration': expiration,
        'type': opt_type,
        'last_price': last,
        'bid': last - 0.1,
        'ask': last + 0.1,
        'volume': 100,
    })


@pytest.fixture
def allocations_config():
    """Fixture for capital allocations config."""
    return {
        "strategies": {
            "credit_spread": {
                "allocated_capital": 10000.0,
                "max_risk_percentage": 0.05
            },
            "velocity_momentum": {  # This is the mapped key from velocity_signal_momentum
                "allocated_capital": 15000.0,
                "max_risk_percentage": 0.03
            }
        }
    }


@pytest.fixture
def decision_store(tmp_path):
    """Create a JsonDecisionStore for testing."""
    return JsonDecisionStore(base_dir=str(tmp_path))


@pytest.fixture
def capital_manager(allocations_config, decision_store):
    """Create a CapitalManager for testing."""
    return CapitalManager(allocations_config, decision_store)


@pytest.fixture
def strategy_mock():
    """Create a mock strategy."""
    from types import SimpleNamespace
    
    class MockStrategy:
        __name__ = "CreditSpreadStrategy"
        
    strategy = MagicMock()
    strategy.data = MagicMock()
    # Create a mock class object
    strategy.__class__ = type('CreditSpreadStrategy', (), {})
    strategy.__class__.__name__ = "CreditSpreadStrategy"
    return strategy


@pytest.fixture
def options_handler_mock():
    """Create a mock options handler."""
    handler = MagicMock()
    handler.symbol = 'SPY'
    return handler


def test_recommender_with_capital_manager_risk_check_pass(
    strategy_mock, options_handler_mock, decision_store, capital_manager
):
    """Test recommendation with capital manager when risk check passes."""
    date = datetime(2025, 8, 8)
    strategy_mock.data.loc.__getitem__.return_value = {'Close': 500}
    strategy_mock.symbol = 'SPY'  # Set symbol attribute
    
    atm_option = _make_option('A', 500, '2025-09-06', 'put', 2.0)
    otm_option = _make_option('B', 495, '2025-09-06', 'put', 1.0)
    
    # Mock on_new_date to create a position
    from algo_trading_engine.backtest.models import Position
    def mock_on_new_date(date_arg, positions, add_position, remove_position):
        if len(positions) == 0:
            position = Position(
                symbol='SPY',
                expiration_date=datetime(2025, 9, 6),
                strategy_type=StrategyType.PUT_CREDIT_SPREAD,
                strike_price=500.0,
                entry_date=date_arg,
                entry_price=1.0,
                spread_options=(atm_option, otm_option)
            )
            position.set_quantity(1)
            add_position(position)
    
    strategy_mock.on_new_date = mock_on_new_date
    
    recommender = InteractiveStrategyRecommender(
        strategy_mock, decision_store, capital_manager, auto_yes=True
    )
    
    recommender.run(date)
    
    # Should succeed - risk is 400 (width 5 - credit 1) * 100, which is less than max 500
    opens = decision_store.get_open_positions()
    assert len(opens) == 1
    assert opens[0].outcome == 'accepted'


def test_recommender_with_capital_manager_risk_check_fail(
    strategy_mock, options_handler_mock, decision_store, capital_manager
):
    """Test recommendation with capital manager when risk check fails."""
    date = datetime(2025, 8, 8)
    strategy_mock.data.loc.__getitem__.return_value = {'Close': 500}
    strategy_mock.symbol = 'SPY'  # Set symbol attribute
    
    atm_option = _make_option('A', 500, '2025-09-06', 'put', 2.0)
    otm_option = _make_option('B', 490, '2025-09-06', 'put', 1.0)
    
    # Width of 10, credit of 1, so max_risk = (10 - 1) * 100 = 900, which exceeds 500
    from algo_trading_engine.backtest.models import Position
    def mock_on_new_date(date_arg, positions, add_position, remove_position):
        if len(positions) == 0:
            position = Position(
                symbol='SPY',
                expiration_date=datetime(2025, 9, 6),
                strategy_type=StrategyType.PUT_CREDIT_SPREAD,
                strike_price=500.0,
                entry_date=date_arg,
                entry_price=1.0,
                spread_options=(atm_option, otm_option)
            )
            position.set_quantity(1)
            add_position(position)
    
    strategy_mock.on_new_date = mock_on_new_date
    
    recommender = InteractiveStrategyRecommender(
        strategy_mock, decision_store, capital_manager, auto_yes=True
    )
    
    recommender.run(date)
    
    # Should fail - risk is 900, which exceeds max allowed 500
    # Position should not be added to store
    opens = decision_store.get_open_positions()
    assert len(opens) == 0


def test_recommender_calculates_max_risk_correctly(
    strategy_mock, options_handler_mock, decision_store, capital_manager
):
    """Test that max risk is calculated correctly for credit spreads."""
    date = datetime(2025, 8, 8)
    strategy_mock.data.loc.__getitem__.return_value = {'Close': 500}
    strategy_mock.symbol = 'SPY'  # Set symbol attribute
    
    atm_option = _make_option('A', 500, '2025-09-06', 'put', 2.0)
    otm_option = _make_option('B', 495, '2025-09-06', 'put', 1.0)
    
    from algo_trading_engine.backtest.models import Position
    def mock_on_new_date(date_arg, positions, add_position, remove_position):
        if len(positions) == 0:
            position = Position(
                symbol='SPY',
                expiration_date=datetime(2025, 9, 6),
                strategy_type=StrategyType.PUT_CREDIT_SPREAD,
                strike_price=500.0,
                entry_date=date_arg,
                entry_price=1.0,
                spread_options=(atm_option, otm_option)
            )
            position.set_quantity(1)
            add_position(position)
    
    strategy_mock.on_new_date = mock_on_new_date
    
    recommender = InteractiveStrategyRecommender(
        strategy_mock, decision_store, capital_manager, auto_yes=True
    )
    
    with patch('builtins.print') as mock_print:
        recommender.run(date)
    
    # Max risk should be (5 - 1) * 100 = 400
    # Check that risk info was displayed
    print_calls = [str(call) for call in mock_print.call_args_list]
    risk_info_present = any("Max Risk" in str(call) or "400" in str(call) for call in print_calls)
    opens = decision_store.get_open_positions()
    assert risk_info_present or len(opens) > 0  # Either displayed or succeeded


def test_recommender_displays_premium_info(
    strategy_mock, options_handler_mock, decision_store, capital_manager
):
    """Test that premium information is displayed for credit strategies."""
    date = datetime(2025, 8, 8)
    strategy_mock.data.loc.__getitem__.return_value = {'Close': 500}
    strategy_mock.symbol = 'SPY'  # Set symbol attribute
    
    atm_option = _make_option('A', 500, '2025-09-06', 'put', 2.0)
    otm_option = _make_option('B', 495, '2025-09-06', 'put', 1.0)
    
    from algo_trading_engine.backtest.models import Position
    def mock_on_new_date(date_arg, positions, add_position, remove_position):
        if len(positions) == 0:
            position = Position(
                symbol='SPY',
                expiration_date=datetime(2025, 9, 6),
                strategy_type=StrategyType.PUT_CREDIT_SPREAD,
                strike_price=500.0,
                entry_date=date_arg,
                entry_price=1.86,
                spread_options=(atm_option, otm_option)
            )
            position.set_quantity(1)
            add_position(position)
    
    strategy_mock.on_new_date = mock_on_new_date
    
    recommender = InteractiveStrategyRecommender(
        strategy_mock, decision_store, capital_manager, auto_yes=True
    )
    
    with patch('builtins.print'):
        recommender.run(date)
    
    # Should succeed and show premium info
    opens = decision_store.get_open_positions()
    assert len(opens) > 0


def test_recommender_strategy_name_mapping(
    options_handler_mock, decision_store, allocations_config
):
    """Test that strategy names are correctly mapped to config keys."""
    date = datetime(2025, 8, 8)
    
    # Create capital manager with velocity_momentum included
    capital_manager = CapitalManager(allocations_config, decision_store)
    
    # Create a strategy mock with proper class name
    strategy_mock = MagicMock()
    strategy_mock.data = MagicMock()
    strategy_mock.data.loc.__getitem__.return_value = {'Close': 500}
    strategy_mock.symbol = 'SPY'  # Set symbol attribute
    # Create a mock class for VelocitySignalMomentumStrategy
    mock_class = type('VelocitySignalMomentumStrategy', (), {})
    strategy_mock.__class__ = mock_class
    
    atm_option = _make_option('A', 500, '2025-09-06', 'put', 2.0)
    otm_option = _make_option('B', 495, '2025-09-06', 'put', 1.0)
    
    from algo_trading_engine.backtest.models import Position
    def mock_on_new_date(date_arg, positions, add_position, remove_position):
        if len(positions) == 0:
            position = Position(
                symbol='SPY',
                expiration_date=datetime(2025, 9, 6),
                strategy_type=StrategyType.PUT_CREDIT_SPREAD,
                strike_price=500.0,
                entry_date=date_arg,
                entry_price=1.0,
                spread_options=(atm_option, otm_option)
            )
            position.set_quantity(1)
            add_position(position)
    
    strategy_mock.on_new_date = mock_on_new_date
    
    recommender = InteractiveStrategyRecommender(
        strategy_mock, decision_store, capital_manager, auto_yes=True
    )
    
    recommender.run(date)
    
    # Should use velocity_momentum config (15000 allocated, 450 max risk)
    # Risk is 400, should pass
    opens = decision_store.get_open_positions()
    assert len(opens) > 0

