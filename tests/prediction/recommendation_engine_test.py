from datetime import datetime
from unittest.mock import MagicMock

from src.prediction.decision_store import JsonDecisionStore, ProposedPositionRequest, DecisionResponse, generate_decision_id
from src.prediction.recommendation_engine import InteractiveStrategyRecommender
from src.backtest.models import StrategyType
from src.common.models import Option


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
    proposal = ProposedPositionRequest(
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
    record = DecisionResponse(
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
    date = datetime(2025, 8, 8)
    strategy.data.loc.__getitem__.return_value = {'Close': 500}
    strategy._make_prediction.return_value = {'strategy': 2, 'confidence': 0.7}
    strategy.recommend_open_position.return_value = {
        'strategy_type': StrategyType.PUT_CREDIT_SPREAD,
        'legs': [atm_option, otm_option],
        'credit': 1.05,
        'width': 5,
        'probability_of_profit': 0.68,
        'confidence': 0.7,
        'expiration_date': '2025-09-06'
    }
    
    strategy._find_best_spread.return_value = {
        'expiry': datetime(2025, 9, 6),
        'width': 5,
        'atm_strike': 500,
        'otm_strike': 495,
        'atm_option': atm_option,
        'otm_option': otm_option,
        'credit': 1.05,
        'risk_reward': 0.45,
        'prob_profit': 0.68,
    }
    strategy._ensure_volume_data.side_effect = lambda opt, d: opt

    options_handler = MagicMock()
    options_handler.symbol = 'SPY'
    store = JsonDecisionStore(base_dir=str(tmp_path))

    rec = InteractiveStrategyRecommender(strategy, options_handler, store, auto_yes=True).recommend_open_position(date)
    assert rec is not None
    opens = store.get_open_positions()
    assert len(opens) == 1


def test_recommender_close_accept(monkeypatch, tmp_path):
    # Prepare an accepted open decision
    legs = (
        _make_option('A', 500, '2025-09-06', 'call', 2.0, 500),
        _make_option('B', 505, '2025-09-06', 'call', 1.0, 500),
    )
    proposal = ProposedPositionRequest(
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
    record = DecisionResponse(
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

    # Mock option chain path
    strategy = MagicMock()
    strategy.options_data = {}
    options_handler = MagicMock()
    
    # Note: The recommendation engine uses strategy.new_options_handler.get_option_bar()
    # which is mocked via strategy.recommend_close_positions.return_value
    # The legacy get_specific_option_contract method is no longer used

    # Mock strategy to recommend closing the position
    from src.backtest.models import Position
    mock_position = Position(
        symbol='SPY',
        expiration_date=datetime(2025, 9, 6),
        strategy_type=StrategyType.CALL_CREDIT_SPREAD,
        strike_price=500.0,
        entry_date=datetime(2025, 7, 1),
        entry_price=1.0,
        spread_options=legs
    )
    mock_position.set_quantity(1)
    
    strategy.recommend_close_positions.return_value = [{
        "position": mock_position,
        "exit_price": 0.5,
        "rationale": "test_close"
    }]

    # Monkeypatch prompt to auto-yes
    rec_engine = InteractiveStrategyRecommender(strategy, options_handler, store, auto_yes=True)
    closed = rec_engine.recommend_close_positions(datetime(2025, 8, 8))

    assert len(closed) == 1
    assert closed[0].exit_price is not None

