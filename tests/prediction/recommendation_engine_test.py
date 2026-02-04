from datetime import datetime
from unittest.mock import MagicMock

from algo_trading_engine.prediction.decision_store import JsonDecisionStore, ProposedPositionRequestDTO, DecisionResponseDTO, generate_decision_id
from algo_trading_engine.prediction.recommendation_engine import InteractiveStrategyRecommender
from algo_trading_engine.prediction.capital_manager import CapitalManager
from algo_trading_engine.common.models import StrategyType
from algo_trading_engine.common.models import Option


def _make_option(symbol: str, strike: float, expiration: str, opt_type: str, last: float, volume: int = 100) -> Option:
    return Option.from_dict({
        'symbol': symbol,
        'ticker': symbol,
        'strike': strike,
        'expiration': expiration,
        'type': opt_type,
        'last_price': last,
        'bid': last - 0.1,
        'ask': last + 0.1,
        'volume': volume,
    })


def test_store_append_and_read(tmp_path):
    store = JsonDecisionStore(base_dir=str(tmp_path))
    date = datetime(2025, 8, 8)
    legs = (
        _make_option('OPT1', 500, '2025-09-06', 'call', 2.0),
        _make_option('OPT2', 505, '2025-09-06', 'call', 1.0),
    )
    proposal = ProposedPositionRequestDTO(
        symbol='SPY',
        strategy_type=StrategyType.CALL_CREDIT_SPREAD,
        legs=legs,
        credit=1.1,
        width=5.0,
        probability_of_profit=0.6,
        confidence=0.65,
        expiration_date='2025-09-06',
        created_at=date.isoformat(),
    )
    rec_id = generate_decision_id(proposal, date.isoformat())
    record = DecisionResponseDTO(
        id=rec_id,
        proposal=proposal,
        outcome='accepted',
        decided_at=date.isoformat(),
        rationale='test',
        quantity=1,
        entry_price=proposal.credit,
    )

    store.append_decision(record)
    open_positions = store.get_open_positions(symbol='SPY', strategy_type=StrategyType.CALL_CREDIT_SPREAD)

    assert len(open_positions) == 1
    assert open_positions[0].id == rec_id


def test_recommender_open_accept(monkeypatch, tmp_path):
    # Create proper option objects first
    atm_option = _make_option('A', 500, '2025-09-06', 'put', 2.0)
    otm_option = _make_option('B', 495, '2025-09-06', 'put', 1.0)
    
    # Strategy mock
    strategy = MagicMock()
    strategy.data = MagicMock()
    strategy.symbol = 'SPY'  # Set symbol attribute
    date = datetime(2025, 8, 8)
    strategy.data.loc.__getitem__.return_value = {'Close': 500}
    # Set proper class name for strategy name extraction
    mock_class = type('CreditSpreadStrategy', (), {})
    strategy.__class__ = mock_class
    
    # Mock on_new_date to create a position via add_position callback
    from algo_trading_engine.vo import create_position
    def mock_on_new_date(date_arg, positions, add_position, remove_position):
        if len(positions) == 0:
            # Create a position and call add_position
            position = create_position(
                symbol='SPY',
                expiration_date=datetime(2025, 9, 6),
                strategy_type=StrategyType.PUT_CREDIT_SPREAD,
                strike_price=500.0,
                entry_date=date_arg,
                entry_price=1.05,
                spread_options=(atm_option, otm_option)
            )
            position.set_quantity(1)
            add_position(position)
    
    strategy.on_new_date = mock_on_new_date

    store = JsonDecisionStore(base_dir=str(tmp_path))
    
    # Create capital manager with default config for testing
    allocations_config = {
        "strategies": {
            "credit_spread": {
                "allocated_capital": 10000.0,
                "max_risk_percentage": 0.05
            }
        }
    }
    capital_manager = CapitalManager(allocations_config, store)

    recommender = InteractiveStrategyRecommender(strategy, store, capital_manager, auto_yes=True)
    recommender.run(date)
    
    opens = store.get_open_positions()
    assert len(opens) == 1


def test_recommender_close_accept(monkeypatch, tmp_path):
    # Prepare an accepted open decision
    legs = (
        _make_option('A', 500, '2025-09-06', 'call', 2.0, 500),
        _make_option('B', 505, '2025-09-06', 'call', 1.0, 500),
    )
    proposal = ProposedPositionRequestDTO(
        symbol='SPY',
        strategy_type=StrategyType.CALL_CREDIT_SPREAD,
        legs=legs,
        credit=1.0,
        width=5.0,
        probability_of_profit=0.6,
        confidence=0.6,
        expiration_date='2025-09-06',
        created_at=datetime(2025, 7, 1).isoformat(),
    )
    decided_at = datetime(2025, 7, 1).isoformat()
    rec_id = generate_decision_id(proposal, decided_at)
    record = DecisionResponseDTO(
        id=rec_id,
        proposal=proposal,
        outcome='accepted',
        decided_at=decided_at,
        rationale='init',
        quantity=1,
        entry_price=1.0,
    )

    store = JsonDecisionStore(base_dir=str(tmp_path))
    store.append_decision(record)

    # Mock strategy
    strategy = MagicMock()
    strategy.symbol = 'SPY'
    strategy.data = MagicMock()
    strategy.data.loc.__getitem__.return_value = {'Close': 500}
    mock_class = type('CreditSpreadStrategy', (), {})
    strategy.__class__ = mock_class
    
    # Mock on_new_date to close the position via remove_position callback
    from algo_trading_engine.vo import create_position
    def mock_on_new_date(date_arg, positions, add_position, remove_position):
        if len(positions) > 0:
            # Close the position
            for position in positions:
                remove_position(date_arg, position, 0.5, None, None)
    
    strategy.on_new_date = mock_on_new_date

    # Create capital manager with default config for testing
    allocations_config = {
        "strategies": {
            "credit_spread": {
                "allocated_capital": 10000.0,
                "max_risk_percentage": 0.05
            }
        }
    }
    capital_manager = CapitalManager(allocations_config, store)
    
    # Run recommender
    rec_engine = InteractiveStrategyRecommender(strategy, store, capital_manager, auto_yes=True)
    rec_engine.run(datetime(2025, 8, 8))

    # Verify position was closed
    open_positions = store.get_open_positions()
    assert len(open_positions) == 0  # Position should be closed

