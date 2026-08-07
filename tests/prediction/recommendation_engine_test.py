from datetime import datetime
from unittest.mock import MagicMock, patch
from algo_trading_engine.prediction.decision_store import JsonDecisionStore, ProposedPositionRequestDTO, DecisionResponseDTO, generate_decision_id
from algo_trading_engine.prediction.recommendation_engine import InteractiveStrategyRecommender
from algo_trading_engine.prediction.capital_manager import CapitalManager
from algo_trading_engine.common.models import StrategyType
from algo_trading_engine.common.models import Option
from algo_trading_engine.vo import create_position

def _make_option(symbol: str, strike: float, expiration: str, opt_type: str, last: float, volume: int=100) -> Option:
    return Option.from_dict({'symbol': symbol, 'ticker': symbol, 'strike': strike, 'expiration': expiration, 'type': opt_type, 'last_price': last, 'bid': last - 0.1, 'ask': last + 0.1, 'volume': volume})

def test_store_append_and_read(tmp_path):
    store = JsonDecisionStore(base_dir=str(tmp_path))
    date = datetime(2025, 8, 8)
    legs = (_make_option('OPT1', 500, '2025-09-06', 'call', 2.0), _make_option('OPT2', 505, '2025-09-06', 'call', 1.0))
    proposal = ProposedPositionRequestDTO(symbol='SPY', strategy_type=StrategyType.CALL_CREDIT_SPREAD, legs=legs, credit=1.1, width=5.0, probability_of_profit=0.6, confidence=0.65, expiration_date='2025-09-06', created_at=date.isoformat())
    rec_id = generate_decision_id(proposal, date.isoformat())
    record = DecisionResponseDTO(id=rec_id, proposal=proposal, outcome='accepted', decided_at=date.isoformat(), rationale='test', quantity=1, entry_price=proposal.credit)
    store.append_decision(record)
    open_positions = store.get_open_positions(symbol='SPY', strategy_type=StrategyType.CALL_CREDIT_SPREAD)
    assert len(open_positions) == 1
    assert open_positions[0].id == rec_id

def test_different_strategies_produce_distinct_decision_ids(tmp_path):
    """Two strategies with identical proposals except strategy_name must get different IDs."""
    date = datetime(2025, 8, 8)
    legs = (_make_option('OPT1', 500, '2025-09-06', 'put', 2.0), _make_option('OPT2', 495, '2025-09-06', 'put', 1.0))
    shared_kwargs = dict(symbol='SPY', strategy_type=StrategyType.PUT_CREDIT_SPREAD, legs=legs, credit=1.0, width=5.0, probability_of_profit=0.7, confidence=0.7, expiration_date='2025-09-06', created_at=date.isoformat())
    proposal_a = ProposedPositionRequestDTO(**shared_kwargs, strategy_name='credit_spread')
    proposal_b = ProposedPositionRequestDTO(**shared_kwargs, strategy_name='velocity_momentum')
    id_a = generate_decision_id(proposal_a, date.isoformat())
    id_b = generate_decision_id(proposal_b, date.isoformat())
    assert id_a != id_b, 'IDs must differ when strategy_name differs'
    store = JsonDecisionStore(base_dir=str(tmp_path))
    store.append_decision(DecisionResponseDTO(id=id_a, proposal=proposal_a, outcome='accepted', decided_at=date.isoformat(), rationale='a', quantity=1, entry_price=1.0))
    store.append_decision(DecisionResponseDTO(id=id_b, proposal=proposal_b, outcome='accepted', decided_at=date.isoformat(), rationale='b', quantity=1, entry_price=1.0))
    assert len(store.get_open_positions()) == 2

def test_recommender_open_accept(tmp_path):
    atm_option = _make_option('A', 500, '2025-09-06', 'put', 2.0)
    otm_option = _make_option('B', 495, '2025-09-06', 'put', 1.0)
    strategy = MagicMock()
    strategy.data = MagicMock()
    strategy.symbol = 'SPY'
    date = datetime(2025, 8, 8)
    strategy.data.loc.__getitem__.return_value = {'Close': 500}
    mock_class = type('CreditSpreadStrategy', (), {})
    strategy.__class__ = mock_class
    from algo_trading_engine.vo import create_position

    def mock_on_new_date(date_arg, positions, add_position, _remove_position):
        if len(positions) == 0:
            position = create_position(symbol='SPY', expiration_date=datetime(2025, 9, 6), strategy_type=StrategyType.PUT_CREDIT_SPREAD, strike_price=500.0, entry_date=date_arg, entry_price=1.05, spread_options=(atm_option, otm_option))
            position.set_quantity(1)
            add_position(position)
    strategy.on_new_date = mock_on_new_date
    store = JsonDecisionStore(base_dir=str(tmp_path))
    allocations_config = {'strategies': {'credit_spread': {'allocated_capital': 10000.0, 'max_risk_percentage': 0.05}}}
    capital_manager = CapitalManager(allocations_config, store)
    recommender = InteractiveStrategyRecommender(strategy, store, capital_manager, auto_yes=True)
    recommender.run(date)
    opens = store.get_open_positions()
    assert len(opens) == 1

def test_recommender_close_accept(tmp_path):
    legs = (_make_option('A', 500, '2025-09-06', 'call', 2.0, 500), _make_option('B', 505, '2025-09-06', 'call', 1.0, 500))
    proposal = ProposedPositionRequestDTO(symbol='SPY', strategy_type=StrategyType.CALL_CREDIT_SPREAD, legs=legs, credit=1.0, width=5.0, probability_of_profit=0.6, confidence=0.6, expiration_date='2025-09-06', created_at=datetime(2025, 7, 1).isoformat(), strategy_name='credit_spread')
    decided_at = datetime(2025, 7, 1).isoformat()
    rec_id = generate_decision_id(proposal, decided_at)
    record = DecisionResponseDTO(id=rec_id, proposal=proposal, outcome='accepted', decided_at=decided_at, rationale='init', quantity=1, entry_price=1.0)
    store = JsonDecisionStore(base_dir=str(tmp_path))
    store.append_decision(record)
    strategy = MagicMock()
    strategy.symbol = 'SPY'
    strategy.data = MagicMock()
    strategy.data.loc.__getitem__.return_value = {'Close': 500}
    mock_class = type('CreditSpreadStrategy', (), {})
    strategy.__class__ = mock_class
    from algo_trading_engine.vo import create_position

    def mock_on_new_date(date_arg, positions, _add_position, remove_position):
        if len(positions) > 0:
            for position in positions:
                remove_position(date_arg, position, 0.5, None, None)
    strategy.on_new_date = mock_on_new_date
    allocations_config = {'strategies': {'credit_spread': {'allocated_capital': 10000.0, 'max_risk_percentage': 0.05}}}
    capital_manager = CapitalManager(allocations_config, store)
    rec_engine = InteractiveStrategyRecommender(strategy, store, capital_manager, auto_yes=True)
    rec_engine.run(datetime(2025, 8, 8))
    open_positions = store.get_open_positions()
    assert len(open_positions) == 0

def test_recommender_open_calls_on_add_position_success(tmp_path):
    """on_add_position_success is called after an accepted open recommendation."""
    atm_option = _make_option('A', 500, '2025-09-06', 'put', 2.0)
    otm_option = _make_option('B', 495, '2025-09-06', 'put', 1.0)
    strategy = MagicMock()
    strategy.data = MagicMock()
    strategy.symbol = 'SPY'
    date = datetime(2025, 8, 8)
    strategy.data.loc.__getitem__.return_value = {'Close': 500}
    mock_class = type('CreditSpreadStrategy', (), {})
    strategy.__class__ = mock_class
    created_position = create_position(symbol='SPY', expiration_date=datetime(2025, 9, 6), strategy_type=StrategyType.PUT_CREDIT_SPREAD, strike_price=500.0, entry_date=date, entry_price=1.05, spread_options=(atm_option, otm_option))
    created_position.set_quantity(1)

    def mock_on_new_date(_date_arg, positions, add_position, _remove_position):
        if len(positions) == 0:
            add_position(created_position)
    strategy.on_new_date = mock_on_new_date
    store = JsonDecisionStore(base_dir=str(tmp_path))
    allocations_config = {'strategies': {'credit_spread': {'allocated_capital': 10000.0, 'max_risk_percentage': 0.05}}}
    capital_manager = CapitalManager(allocations_config, store)
    recommender = InteractiveStrategyRecommender(strategy, store, capital_manager, auto_yes=True)
    recommender.run(date)
    strategy.on_add_position_success.assert_called_once_with(created_position)

def test_recommender_close_calls_on_remove_position_success(tmp_path):
    """on_remove_position_success is called after an accepted close recommendation."""
    legs = (_make_option('A', 500, '2025-09-06', 'call', 2.0, 500), _make_option('B', 505, '2025-09-06', 'call', 1.0, 500))
    proposal = ProposedPositionRequestDTO(symbol='SPY', strategy_type=StrategyType.CALL_CREDIT_SPREAD, legs=legs, credit=1.0, width=5.0, probability_of_profit=0.6, confidence=0.6, expiration_date='2025-09-06', created_at=datetime(2025, 7, 1).isoformat(), strategy_name='credit_spread')
    decided_at = datetime(2025, 7, 1).isoformat()
    rec_id = generate_decision_id(proposal, decided_at)
    record = DecisionResponseDTO(id=rec_id, proposal=proposal, outcome='accepted', decided_at=decided_at, rationale='init', quantity=1, entry_price=1.0)
    store = JsonDecisionStore(base_dir=str(tmp_path))
    store.append_decision(record)
    strategy = MagicMock()
    strategy.symbol = 'SPY'
    strategy.data = MagicMock()
    strategy.data.loc.__getitem__.return_value = {'Close': 500}
    mock_class = type('CreditSpreadStrategy', (), {})
    strategy.__class__ = mock_class
    close_date = datetime(2025, 8, 8)

    def mock_on_new_date(date_arg, positions, _add_position, remove_position):
        for position in positions:
            remove_position(date_arg, position, 0.5, 500.0, [200, 300])
    strategy.on_new_date = mock_on_new_date
    allocations_config = {'strategies': {'credit_spread': {'allocated_capital': 10000.0, 'max_risk_percentage': 0.05}}}
    capital_manager = CapitalManager(allocations_config, store)
    recommender = InteractiveStrategyRecommender(strategy, store, capital_manager, auto_yes=True)
    recommender.run(close_date)
    strategy.on_remove_position_success.assert_called_once()
    call_args = strategy.on_remove_position_success.call_args
    assert call_args[0][0] == close_date
    assert call_args[0][2] == 0.5
    assert call_args[0][3] == 500.0
    assert call_args[0][4] == [200, 300]

def test_strategy_b_ignores_strategy_a_open_position(tmp_path):
    """Strategy B must not see open positions that belong to Strategy A.

    Seed the store with an open position created by 'credit_spread' strategy,
    then run the recommender with a 'velocity_momentum' strategy and verify
    that on_new_date receives an empty positions tuple.
    """
    legs_a = (_make_option('A_ATM', 500, '2025-09-06', 'put', 2.0), _make_option('A_OTM', 495, '2025-09-06', 'put', 1.0))
    proposal_a = ProposedPositionRequestDTO(symbol='SPY', strategy_type=StrategyType.PUT_CREDIT_SPREAD, legs=legs_a, credit=1.0, width=5.0, probability_of_profit=0.7, confidence=0.7, expiration_date='2025-09-06', created_at=datetime(2025, 7, 1).isoformat(), strategy_name='credit_spread')
    decided_at_a = datetime(2025, 7, 1).isoformat()
    record_a = DecisionResponseDTO(id=generate_decision_id(proposal_a, decided_at_a), proposal=proposal_a, outcome='accepted', decided_at=decided_at_a, rationale='strategy_a_open', quantity=1, entry_price=1.0)
    store = JsonDecisionStore(base_dir=str(tmp_path))
    store.append_decision(record_a)
    assert len(store.get_open_positions()) == 1
    strategy_b = MagicMock()
    strategy_b.data = MagicMock()
    strategy_b.symbol = 'SPY'
    date = datetime(2025, 8, 8)
    strategy_b.data.loc.__getitem__.return_value = {'Close': 500}
    strategy_b.__class__ = type('VelocitySignalMomentumStrategy', (), {})
    received_positions = []

    def mock_on_new_date(_date_arg, positions, _add_position, _remove_position):
        received_positions.extend(positions)
    strategy_b.on_new_date = mock_on_new_date
    allocations_config = {'strategies': {'velocity_momentum': {'allocated_capital': 10000.0, 'max_risk_percentage': 0.05}}}
    capital_manager = CapitalManager(allocations_config, store)
    recommender = InteractiveStrategyRecommender(strategy_b, store, capital_manager, auto_yes=True)
    recommender.run(date)
    assert received_positions == []

def test_strategy_sees_its_own_open_position(tmp_path):
    """Strategy A must see its own open positions during execution."""
    legs = (_make_option('A_ATM', 500, '2025-09-06', 'put', 2.0), _make_option('A_OTM', 495, '2025-09-06', 'put', 1.0))
    proposal = ProposedPositionRequestDTO(symbol='SPY', strategy_type=StrategyType.PUT_CREDIT_SPREAD, legs=legs, credit=1.0, width=5.0, probability_of_profit=0.7, confidence=0.7, expiration_date='2025-09-06', created_at=datetime(2025, 7, 1).isoformat(), strategy_name='credit_spread')
    decided_at = datetime(2025, 7, 1).isoformat()
    record = DecisionResponseDTO(id=generate_decision_id(proposal, decided_at), proposal=proposal, outcome='accepted', decided_at=decided_at, rationale='own_position', quantity=1, entry_price=1.0)
    store = JsonDecisionStore(base_dir=str(tmp_path))
    store.append_decision(record)
    strategy_a = MagicMock()
    strategy_a.data = MagicMock()
    strategy_a.symbol = 'SPY'
    date = datetime(2025, 8, 8)
    strategy_a.data.loc.__getitem__.return_value = {'Close': 500}
    strategy_a.__class__ = type('CreditSpreadStrategy', (), {})
    received_positions = []

    def mock_on_new_date(_date_arg, positions, _add_position, _remove_position):
        received_positions.extend(positions)
    strategy_a.on_new_date = mock_on_new_date
    allocations_config = {'strategies': {'credit_spread': {'allocated_capital': 10000.0, 'max_risk_percentage': 0.05}}}
    capital_manager = CapitalManager(allocations_config, store)
    recommender = InteractiveStrategyRecommender(strategy_a, store, capital_manager, auto_yes=True)
    recommender.run(date)
    assert len(received_positions) == 1

def test_get_exit_price_from_user_prompts_with_bar_data_uses_defaults(tmp_path):
    """With auto_yes=False and get_option_bar returning data, prompt for exit price with default from bar (Enter = use default)."""
    atm_option = _make_option('SPY_500P', 500, '2025-09-06', 'put', 2.0)
    otm_option = _make_option('SPY_495P', 495, '2025-09-06', 'put', 1.0)
    position = create_position(symbol='SPY', expiration_date=datetime(2025, 9, 6), strategy_type=StrategyType.PUT_CREDIT_SPREAD, strike_price=500.0, entry_date=datetime(2025, 8, 1), entry_price=1.05, spread_options=(atm_option, otm_option))
    position.set_quantity(1)
    strategy = MagicMock()
    strategy.symbol = 'SPY'
    atm_bar = MagicMock()
    atm_bar.close_price = 1.8
    otm_bar = MagicMock()
    otm_bar.close_price = 0.9
    strategy.get_current_option_bar = MagicMock(side_effect=[atm_bar, otm_bar])
    store = JsonDecisionStore(base_dir=str(tmp_path))
    allocations_config = {'strategies': {'credit_spread': {'allocated_capital': 10000.0, 'max_risk_percentage': 0.05}}}
    capital_manager = CapitalManager(allocations_config, store)
    recommender = InteractiveStrategyRecommender(strategy, store, capital_manager, auto_yes=False)
    with patch('algo_trading_engine.prediction.recommendation_engine.input', side_effect=['', '']):
        result = recommender._get_exit_price_from_user_prompts(position, datetime(2025, 8, 8))
    assert result is not None
    assert abs(result - 0.9) < 1e-06

def test_get_exit_price_from_user_prompts_without_bar_data_prompts_without_default(tmp_path):
    """With auto_yes=False and get_option_bar NOT returning data, prompt for net exit price without default."""
    atm_option = _make_option('SPY_500P', 500, '2025-09-06', 'put', 2.0)
    otm_option = _make_option('SPY_495P', 495, '2025-09-06', 'put', 1.0)
    position = create_position(symbol='SPY', expiration_date=datetime(2025, 9, 6), strategy_type=StrategyType.PUT_CREDIT_SPREAD, strike_price=500.0, entry_date=datetime(2025, 8, 1), entry_price=1.05, spread_options=(atm_option, otm_option))
    position.set_quantity(1)
    strategy = MagicMock()
    strategy.symbol = 'SPY'
    strategy.get_current_option_bar = MagicMock(return_value=None)
    strategy.get_current_underlying_price = MagicMock(return_value=None)
    store = JsonDecisionStore(base_dir=str(tmp_path))
    allocations_config = {'strategies': {'credit_spread': {'allocated_capital': 10000.0, 'max_risk_percentage': 0.05}}}
    capital_manager = CapitalManager(allocations_config, store)
    recommender = InteractiveStrategyRecommender(strategy, store, capital_manager, auto_yes=False)
    with patch('algo_trading_engine.prediction.recommendation_engine.input', return_value='0.75'):
        result = recommender._get_exit_price_from_user_prompts(position, datetime(2025, 8, 8))
    assert result is not None
    assert abs(result - 0.75) < 1e-06
