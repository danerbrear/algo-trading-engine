"""Integration tests for PaperTradingEngine.run() method.

These tests verify the full execution path of PaperTradingEngine including:
- Engine initialization
- Running with and without open positions
- Integration with recommendation engine
- Status display for open positions
"""
import pytest
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch, Mock
import pandas as pd
from algo_trading_engine.indicators.indicator import Indicator
from algo_trading_engine.strategy import IndicatorUpdateError, Strategy
from algo_trading_engine.trade import PaperTradingEngine, compute_paper_trading_fetch_start_date, DEFAULT_PAPER_TRADING_LSTM_LOOKBACK_DAYS
from algo_trading_engine.enums import BarTimeInterval
from algo_trading_engine.models.config import PaperTradingConfig
from algo_trading_engine.database.decision_store import DecisionStore, JsonDecisionStore, generate_decision_id
from algo_trading_engine.dto import DecisionResponseDTO, ProposedPositionRequestDTO
from algo_trading_engine.enums import StrategyType
from algo_trading_engine.vo import Option

def _make_option(symbol: str, strike: float, expiration: str, opt_type: str, last: float, volume: int=100) -> Option:
    """Helper to create Option objects for testing."""
    return Option.from_dict({'symbol': symbol, 'ticker': symbol, 'timestamp': datetime(2025, 8, 8), 'strike': strike, 'expiration': expiration, 'type': opt_type, 'last_price': last, 'bid': last - 0.1, 'ask': last + 0.1, 'volume': volume})

@pytest.fixture
def mock_strategy():
    """Create a mock strategy for testing."""
    strategy = MagicMock()
    strategy.symbol = 'SPY'
    strategy.__class__ = type('CreditSpreadStrategy', (), {})
    dates = pd.date_range(start='2025-01-01', end='2025-08-08', freq='D')
    data = pd.DataFrame({'Close': [500.0] * len(dates), 'Open': [499.0] * len(dates), 'High': [501.0] * len(dates), 'Low': [498.0] * len(dates)}, index=dates)
    strategy.data = data
    strategy.set_data = Mock()
    strategy.indicators = []
    strategy.recommend_open_position.return_value = None
    strategy.recommend_close_positions.return_value = []
    return strategy

@pytest.fixture
def mock_options_handler():
    """Create a mock options handler."""
    handler = MagicMock()
    handler.symbol = 'SPY'
    return handler

@pytest.fixture
def paper_trading_config():
    """Create a PaperTradingConfig for testing."""
    return PaperTradingConfig(symbol='SPY', strategy_type='credit_spread', api_key='test_api_key', use_free_tier=True)

@pytest.fixture
def decision_store_with_allocations(tmp_path):
    """Create a decision store and initialize capital allocations config."""
    decisions_path = tmp_path / 'predictions' / 'decisions'
    decisions_path.mkdir(parents=True, exist_ok=True)
    store = JsonDecisionStore(base_dir=str(decisions_path.parent))
    config_path = tmp_path / 'config' / 'strategies'
    config_path.mkdir(parents=True, exist_ok=True)
    config_file = config_path / 'capital_allocations.json'
    import json
    config_data = {'strategies': {'credit_spread': {'allocated_capital': 10000.0, 'max_risk_percentage': 0.05}}}
    with open(config_file, 'w') as f:
        json.dump(config_data, f)
    return (store, str(config_file))

def test_paper_trading_engine_run_no_open_positions(mock_strategy, mock_options_handler, paper_trading_config, decision_store_with_allocations, monkeypatch, tmp_path):
    """Test PaperTradingEngine.run() when there are no open positions.
    
    This test verifies that the engine runs successfully when there are no
    existing open positions to check.
    """
    store, config_file = decision_store_with_allocations
    monkeypatch.chdir(tmp_path)
    engine = PaperTradingEngine(strategy=mock_strategy, config=paper_trading_config, options_handler=mock_options_handler)
    with patch.object(mock_strategy, 'get_current_underlying_price', return_value=500.0), patch('algo_trading_engine.data_retriever.DataRetriever.get_live_price', return_value=500.0), patch('algo_trading_engine.trade.InteractiveStrategyRecommender') as mock_recommender_class:
        mock_recommender = Mock()
        mock_recommender.get_open_positions_status.return_value = []
        mock_recommender.run.return_value = None
        mock_recommender_class.return_value = mock_recommender
        success = engine.run()
        assert success is True
        mock_recommender_class.assert_called_once()
        mock_recommender.run.assert_called_once()
        mock_recommender.get_open_positions_status.assert_not_called()

def test_paper_trading_engine_run_with_open_positions(mock_strategy, mock_options_handler, paper_trading_config, decision_store_with_allocations, monkeypatch, tmp_path):
    """Test PaperTradingEngine.run() when there are existing open positions.
    
    This test would have caught the bug where 'recommender' was used before
    being defined when open positions exist.
    """
    store, config_file = decision_store_with_allocations
    monkeypatch.chdir(tmp_path)
    default_store = JsonDecisionStore()
    legs = (_make_option('SPY_PUT_500', 500, '2025-09-06', 'put', 2.0), _make_option('SPY_PUT_495', 495, '2025-09-06', 'put', 1.0))
    proposal = ProposedPositionRequestDTO(symbol='SPY', strategy_type=StrategyType.PUT_CREDIT_SPREAD, legs=legs, credit=1.0, width=5.0, probability_of_profit=0.68, confidence=0.7, expiration_date='2025-09-06', created_at=datetime(2025, 7, 1).isoformat(), strategy_name='credit_spread')
    decided_at = datetime(2025, 7, 1).isoformat()
    rec_id = generate_decision_id(proposal, decided_at)
    record = DecisionResponseDTO(id=rec_id, proposal=proposal, decided_at=decided_at, rationale='test position', quantity=1, entry_price=1.0)
    default_store.append_decision(record)
    open_positions = default_store.get_open_positions(symbol='SPY')
    assert len(open_positions) == 1, 'Should have one open position'
    engine = PaperTradingEngine(strategy=mock_strategy, config=paper_trading_config, options_handler=mock_options_handler)
    with patch.object(mock_strategy, 'get_current_underlying_price', return_value=500.0), patch('algo_trading_engine.data_retriever.DataRetriever.get_live_price', return_value=500.0), patch('algo_trading_engine.trade.InteractiveStrategyRecommender') as mock_recommender_class:
        mock_recommender = Mock()
        mock_recommender.get_open_positions_status.return_value = [{'symbol': 'SPY', 'strategy_type': 'PUT_CREDIT_SPREAD', 'quantity': 1, 'entry_price': 1.0, 'exit_price': 0.5, 'pnl_dollars': 50.0, 'pnl_percent': 0.5, 'days_held': 38, 'dte': 29}]
        mock_recommender.run.return_value = None
        mock_recommender_class.return_value = mock_recommender
        success = engine.run()
        assert success is True
        mock_recommender_class.assert_called_once()
        mock_recommender.get_open_positions_status.assert_called_once()
        mock_recommender.run.assert_called_once()

def test_paper_trading_engine_run_with_recommendation_engine_failure(mock_strategy, mock_options_handler, paper_trading_config, decision_store_with_allocations, monkeypatch, tmp_path):
    """Test PaperTradingEngine.run() handles recommendation engine failures gracefully."""
    store, config_file = decision_store_with_allocations
    monkeypatch.chdir(tmp_path)
    engine = PaperTradingEngine(strategy=mock_strategy, config=paper_trading_config, options_handler=mock_options_handler)
    with patch.object(mock_strategy, 'get_current_underlying_price', return_value=500.0), patch('algo_trading_engine.data_retriever.DataRetriever.get_live_price', return_value=500.0), patch('algo_trading_engine.trade.InteractiveStrategyRecommender') as mock_recommender_class:
        mock_recommender = Mock()
        mock_recommender.run.side_effect = Exception('Test error')
        mock_recommender_class.return_value = mock_recommender
        success = engine.run()
        assert success is False

def test_paper_trading_engine_run_without_options_handler(mock_strategy, paper_trading_config, decision_store_with_allocations, monkeypatch, tmp_path):
    """Test PaperTradingEngine.run() fails gracefully without options handler."""
    store, config_file = decision_store_with_allocations
    monkeypatch.chdir(tmp_path)
    if hasattr(mock_strategy, 'options_handler'):
        delattr(mock_strategy, 'options_handler')
    engine = PaperTradingEngine(strategy=mock_strategy, config=paper_trading_config, options_handler=None)
    success = engine.run()
    assert success is False

def test_paper_trading_engine_run_missing_capital_config(mock_strategy, mock_options_handler, paper_trading_config, tmp_path, monkeypatch):
    """Test PaperTradingEngine.run() handles missing capital config gracefully."""
    store = JsonDecisionStore(base_dir=str(tmp_path))
    monkeypatch.chdir(tmp_path)
    engine = PaperTradingEngine(strategy=mock_strategy, config=paper_trading_config, options_handler=mock_options_handler)
    with patch.object(mock_strategy, 'get_current_underlying_price', return_value=500.0), patch('algo_trading_engine.data_retriever.DataRetriever.get_live_price', return_value=500.0), patch('algo_trading_engine.trade.InteractiveStrategyRecommender') as mock_recommender_class:
        mock_recommender = Mock()
        mock_recommender.get_open_positions_status.return_value = []
        mock_recommender.run.return_value = None
        mock_recommender_class.return_value = mock_recommender
        success = engine.run()
        assert success is True

def test_paper_trading_engine_strategy_name_extraction(mock_options_handler, paper_trading_config):
    """Test that strategy name is correctly extracted from strategy class."""
    test_cases = [('CreditSpreadStrategy', 'credit_spread'), ('VelocitySignalMomentumStrategy', 'velocity_signal_momentum'), ('MyCustomStrategy', 'my_custom')]
    for class_name, expected_name in test_cases:
        strategy = MagicMock()
        strategy.symbol = 'SPY'
        strategy.__class__ = type(class_name, (), {})
        dates = pd.date_range(start='2025-01-01', end='2025-08-08', freq='D')
        strategy.data = pd.DataFrame({'Close': [500.0] * len(dates)}, index=dates)
        engine = PaperTradingEngine(strategy=strategy, config=paper_trading_config, options_handler=mock_options_handler)
        strategy_name = engine._get_strategy_name_from_class()
        if expected_name == 'velocity_signal_momentum':
            assert strategy_name in ['velocity_signal_momentum', 'velocity_momentum']
        else:
            assert strategy_name == expected_name

class TestComputePaperTradingFetchStartDate:
    """Unit tests for compute_paper_trading_fetch_start_date (warm-up + LSTM window)."""

    def test_uses_earlier_date_when_indicator_warmup_exceeds_lstm_lookback(self):
        strategy = MagicMock()
        strategy.get_warm_up_period_timedelta = Mock(return_value=timedelta(days=150))
        now = datetime(2025, 6, 1, 12, 0, 0)
        result = compute_paper_trading_fetch_start_date(now, strategy, BarTimeInterval.DAY)
        expected = (now - timedelta(days=150)).strftime('%Y-%m-%d')
        assert result == expected
        strategy.get_warm_up_period_timedelta.assert_called_once_with(BarTimeInterval.DAY)

    def test_uses_lstm_lookback_when_it_exceeds_warmup_window(self):
        strategy = MagicMock()
        strategy.get_warm_up_period_timedelta = Mock(return_value=timedelta(days=30))
        now = datetime(2025, 6, 1, 12, 0, 0)
        result = compute_paper_trading_fetch_start_date(now, strategy, BarTimeInterval.DAY)
        expected = (now - timedelta(days=DEFAULT_PAPER_TRADING_LSTM_LOOKBACK_DAYS)).strftime('%Y-%m-%d')
        assert result == expected

    def test_passes_bar_interval_for_hourly_bars(self):
        strategy = MagicMock()
        strategy.get_warm_up_period_timedelta = Mock(return_value=timedelta(days=15))
        now = datetime(2025, 6, 1, 12, 0, 0)
        compute_paper_trading_fetch_start_date(now, strategy, BarTimeInterval.HOUR)
        strategy.get_warm_up_period_timedelta.assert_called_once_with(BarTimeInterval.HOUR)

class TestPaperTradingEngineFromConfig:
    """Test PaperTradingEngine.from_config() factory method."""

    @patch('algo_trading_engine.trade.OptionsHandler')
    @patch('algo_trading_engine.trade.DataRetriever')
    def test_from_config_sets_symbol_on_strategy_instance(self, mock_data_retriever, _mock_options_handler):
        """Test that symbol is set on strategy when strategy instance is provided."""
        mock_strategy = MagicMock()
        mock_strategy.set_data = Mock()
        mock_strategy.indicators = []
        mock_strategy.options_handler = None
        mock_strategy.warm_up_period = 0
        mock_strategy.get_warm_up_period_timedelta = Mock(return_value=timedelta(0))
        mock_retriever_instance = MagicMock()
        mock_retriever_instance.treasury_rates = None
        mock_retriever_instance.fetch_data_for_period.return_value = pd.DataFrame({'Close': [500.0] * 10, 'Open': [499.0] * 10, 'High': [501.0] * 10, 'Low': [498.0] * 10}, index=pd.date_range('2025-01-01', periods=10, freq='D'))
        mock_data_retriever.return_value = mock_retriever_instance
        config = PaperTradingConfig(symbol='QQQ', strategy_type=mock_strategy, api_key='test_key', use_free_tier=True)
        engine = PaperTradingEngine.from_config(config)
        assert engine.strategy == mock_strategy
        assert mock_strategy.symbol == 'QQQ'

    @patch('algo_trading_engine.trade.compute_paper_trading_fetch_start_date', return_value='2024-06-01')
    @patch('algo_trading_engine.trade.OptionsHandler')
    @patch('algo_trading_engine.trade.DataRetriever')
    def test_from_config_passes_computed_fetch_start_to_retriever_and_fetch(self, mock_data_retriever, _mock_options_handler, mock_compute_start):
        """Fetch window uses the date from compute_paper_trading_fetch_start_date."""
        mock_strategy = MagicMock()
        mock_strategy.set_data = Mock()
        mock_strategy.indicators = []
        mock_strategy.options_handler = None
        mock_strategy.warm_up_period = 0
        mock_strategy.get_warm_up_period_timedelta = Mock(return_value=timedelta(0))
        mock_retriever_instance = MagicMock()
        mock_retriever_instance.treasury_rates = None
        mock_retriever_instance.fetch_data_for_period.return_value = pd.DataFrame({'Close': [500.0] * 10, 'Open': [499.0] * 10, 'High': [501.0] * 10, 'Low': [498.0] * 10}, index=pd.date_range('2025-01-01', periods=10, freq='D'))
        mock_data_retriever.return_value = mock_retriever_instance
        config = PaperTradingConfig(symbol='QQQ', strategy_type=mock_strategy, api_key='test_key', use_free_tier=True)
        PaperTradingEngine.from_config(config)
        mock_compute_start.assert_called_once()
        mock_data_retriever.assert_called_once()
        assert 'lstm_start_date' not in mock_data_retriever.call_args[1]
        mock_retriever_instance.fetch_data_for_period.assert_called_once_with('2024-06-01')

    @patch("algo_trading_engine.trade.apply_live_quote")
    @patch("algo_trading_engine.trade.market_now", return_value=datetime(2026, 11, 11, 10, 30))
    @patch("algo_trading_engine.trade.OptionsHandler")
    @patch("algo_trading_engine.trade.DataRetriever")
    def test_from_config_applies_live_quote_before_set_data(
        self,
        mock_data_retriever,
        _mock_options_handler,
        _mock_market_now,
        mock_apply_live,
    ):
        frame = pd.DataFrame(
            {
                "Close": [500.0] * 2,
                "Open": [499.0] * 2,
                "High": [501.0] * 2,
                "Low": [498.0] * 2,
            },
            index=pd.to_datetime(["2026-11-10", "2026-11-11"]),
        )
        mock_strategy = MagicMock()
        mock_strategy.set_data = Mock()
        mock_strategy.indicators = []
        mock_strategy.options_handler = None
        mock_strategy.warm_up_period = 0
        mock_strategy.get_warm_up_period_timedelta = Mock(return_value=timedelta(0))
        mock_retriever_instance = MagicMock()
        mock_retriever_instance.treasury_rates = None
        mock_retriever_instance.fetch_data_for_period.return_value = frame
        mock_retriever_instance.get_live_price.return_value = 505.0
        mock_data_retriever.return_value = mock_retriever_instance
        mock_apply_live.side_effect = lambda data, *_a, **_k: data
        config = PaperTradingConfig(
            symbol="SPY", strategy_type=mock_strategy, api_key="test_key", use_free_tier=True
        )
        PaperTradingEngine.from_config(config)
        mock_apply_live.assert_called_once()
        mock_strategy.set_data.assert_called_once()

    @patch('algo_trading_engine.trade.OptionsHandler')
    @patch('algo_trading_engine.trade.DataRetriever')
    @patch('algo_trading_engine.trade.TradingEngine.precompute_indicators')
    def test_from_config_precomputes_indicators(self, mock_precompute, mock_data_retriever, _mock_options_handler):
        """from_config must precompute indicators when constructing the engine."""
        mock_strategy = MagicMock()
        mock_strategy.set_data = Mock()
        mock_strategy.indicators = []
        mock_strategy.options_handler = None
        mock_strategy.warm_up_period = 0
        mock_strategy.get_warm_up_period_timedelta = Mock(return_value=timedelta(0))
        mock_retriever_instance = MagicMock()
        mock_retriever_instance.treasury_rates = None
        mock_retriever_instance.fetch_data_for_period.return_value = pd.DataFrame({'Close': [500.0] * 10, 'Open': [499.0] * 10, 'High': [501.0] * 10, 'Low': [498.0] * 10}, index=pd.date_range('2025-01-01', periods=10, freq='D'))
        mock_data_retriever.return_value = mock_retriever_instance
        config = PaperTradingConfig(symbol='SPY', strategy_type=mock_strategy, api_key='test_key', use_free_tier=True)
        PaperTradingEngine.from_config(config)
        mock_strategy.set_data.assert_called_once()
        mock_precompute.assert_called_once()

class TestCustomDecisionStore:
    """Test that a custom DecisionStore passed via PaperTradingConfig is used by the engine."""

    def test_engine_uses_custom_decision_store_from_config(self, mock_strategy, mock_options_handler, monkeypatch, tmp_path):
        """When PaperTradingConfig.decision_store is set, the engine must use it
        instead of creating a default JsonDecisionStore."""
        monkeypatch.chdir(tmp_path)
        custom_store = MagicMock(spec=DecisionStore)
        custom_store.get_open_positions.return_value = []
        config = PaperTradingConfig(symbol='SPY', strategy_type='credit_spread', api_key='test_api_key', use_free_tier=True, decision_store=custom_store)
        engine = PaperTradingEngine(strategy=mock_strategy, config=config, options_handler=mock_options_handler)
        with patch.object(mock_strategy, 'get_current_underlying_price', return_value=500.0), patch('algo_trading_engine.data_retriever.DataRetriever.get_live_price', return_value=500.0), patch('algo_trading_engine.trade.InteractiveStrategyRecommender') as mock_rec_cls:
            mock_recommender = Mock()
            mock_recommender.run.return_value = None
            mock_rec_cls.return_value = mock_recommender
            success = engine.run()
            assert success is True
            mock_rec_cls.assert_called_once()
            _, call_kwargs = mock_rec_cls.call_args
            if not call_kwargs:
                call_args = mock_rec_cls.call_args[0]
                assert call_args[1] is custom_store, 'Recommender should receive the custom DecisionStore'
            else:
                assert call_kwargs.get('decision_store', mock_rec_cls.call_args[0][1]) is custom_store
            custom_store.get_open_positions.assert_called_once_with(symbol='SPY', strategy_name='credit_spread')

    def test_engine_falls_back_to_json_store_when_none(self, mock_strategy, mock_options_handler, monkeypatch, tmp_path):
        """When PaperTradingConfig.decision_store is None, the engine creates
        a default JsonDecisionStore."""
        monkeypatch.chdir(tmp_path)
        config = PaperTradingConfig(symbol='SPY', strategy_type='credit_spread', api_key='test_api_key', use_free_tier=True, decision_store=None)
        engine = PaperTradingEngine(strategy=mock_strategy, config=config, options_handler=mock_options_handler)
        with patch.object(mock_strategy, 'get_current_underlying_price', return_value=500.0), patch('algo_trading_engine.data_retriever.DataRetriever.get_live_price', return_value=500.0), patch('algo_trading_engine.trade.InteractiveStrategyRecommender') as mock_rec_cls:
            mock_recommender = Mock()
            mock_recommender.run.return_value = None
            mock_rec_cls.return_value = mock_recommender
            success = engine.run()
            assert success is True
            call_args = mock_rec_cls.call_args[0]
            assert isinstance(call_args[1], JsonDecisionStore), 'Recommender should receive a JsonDecisionStore when config.decision_store is None'

class TestAutoYesPassthrough:
    """Test that PaperTradingConfig.auto_yes is forwarded to InteractiveStrategyRecommender."""

    @pytest.mark.parametrize('auto_yes_value', [True, False])
    def test_engine_passes_auto_yes_to_recommender(self, mock_strategy, mock_options_handler, monkeypatch, tmp_path, auto_yes_value):
        monkeypatch.chdir(tmp_path)
        custom_store = MagicMock(spec=DecisionStore)
        custom_store.get_open_positions.return_value = []
        config = PaperTradingConfig(symbol='SPY', strategy_type='credit_spread', api_key='test_api_key', use_free_tier=True, decision_store=custom_store, auto_yes=auto_yes_value)
        engine = PaperTradingEngine(strategy=mock_strategy, config=config, options_handler=mock_options_handler)
        with patch.object(mock_strategy, 'get_current_underlying_price', return_value=500.0), patch('algo_trading_engine.data_retriever.DataRetriever.get_live_price', return_value=500.0), patch('algo_trading_engine.trade.InteractiveStrategyRecommender') as mock_rec_cls:
            mock_recommender = Mock()
            mock_recommender.run.return_value = None
            mock_rec_cls.return_value = mock_recommender
            success = engine.run()
            assert success is True
            _, call_kwargs = mock_rec_cls.call_args
            assert call_kwargs['auto_yes'] is auto_yes_value

class TestStrategyCallbacksFromPaperTradingEngine:
    """Verify that strategy lifecycle callbacks fire through PaperTradingEngine.

    These tests let the real InteractiveStrategyRecommender run (no class-level
    mock) so we exercise the full path:
        PaperTradingEngine.run() -> InteractiveStrategyRecommender.run()
        -> strategy.on_new_date() -> capture callbacks -> process recommendation
        -> strategy.on_add_position_success / on_remove_position_success
    """

    def _build_engine(self, strategy, tmp_path, decision_store=None, auto_yes=True):
        """Helper: build a PaperTradingEngine with minimal plumbing."""
        config = PaperTradingConfig(symbol='SPY', strategy_type='credit_spread', api_key='test_api_key', use_free_tier=True, decision_store=decision_store or JsonDecisionStore(base_dir=str(tmp_path / 'predictions')), auto_yes=auto_yes)
        handler = MagicMock()
        handler.symbol = 'SPY'
        return PaperTradingEngine(strategy=strategy, config=config, options_handler=handler)

    def _make_strategy_mock(self):
        strategy = MagicMock()
        strategy.symbol = 'SPY'
        strategy.__class__ = type('CreditSpreadStrategy', (), {})
        dates = pd.date_range(start='2025-01-01', end='2025-08-08', freq='D')
        strategy.data = pd.DataFrame({'Close': [500.0] * len(dates)}, index=dates)
        strategy.profit_target = None
        strategy.stop_loss = None
        strategy.indicators = []
        return strategy

    def test_on_add_position_success_called_after_accepted_open(self, monkeypatch, tmp_path):
        """PaperTradingEngine -> real recommender -> on_add_position_success fires."""
        monkeypatch.chdir(tmp_path)
        atm = _make_option('A', 500, '2025-09-06', 'put', 2.0)
        otm = _make_option('B', 495, '2025-09-06', 'put', 1.0)
        strategy = self._make_strategy_mock()
        from algo_trading_engine.vo import create_position as _cp

        def fake_on_new_date(date_arg, positions, add_position, _remove_position):
            if len(positions) == 0:
                pos = _cp(symbol='SPY', expiration_date=datetime(2025, 9, 6), strategy_type=StrategyType.PUT_CREDIT_SPREAD, strike_price=500.0, entry_date=date_arg, entry_price=1.05, legs=(atm, otm))
                pos.set_quantity(1)
                add_position(pos)
        strategy.on_new_date = fake_on_new_date
        store = JsonDecisionStore(base_dir=str(tmp_path / 'predictions'))
        engine = self._build_engine(strategy, tmp_path, decision_store=store)
        with patch.object(strategy, 'get_current_underlying_price', return_value=500.0):
            success = engine.run()
        assert success is True
        strategy.on_add_position_success.assert_called_once()
        accepted_pos = strategy.on_add_position_success.call_args[0][0]
        assert accepted_pos.entry_price == 1.05

    def test_on_remove_position_success_called_after_accepted_close(self, monkeypatch, tmp_path):
        """PaperTradingEngine -> real recommender -> on_remove_position_success fires."""
        monkeypatch.chdir(tmp_path)
        legs = (_make_option('A', 500, '2025-09-06', 'call', 2.0, 500), _make_option('B', 505, '2025-09-06', 'call', 1.0, 500))
        store = JsonDecisionStore(base_dir=str(tmp_path / 'predictions'))
        proposal = ProposedPositionRequestDTO(symbol='SPY', strategy_type=StrategyType.CALL_CREDIT_SPREAD, legs=legs, credit=1.0, width=5.0, probability_of_profit=0.6, confidence=0.6, expiration_date='2025-09-06', created_at=datetime(2025, 7, 1).isoformat(), strategy_name='credit_spread')
        decided_at = datetime(2025, 7, 1).isoformat()
        rec_id = generate_decision_id(proposal, decided_at)
        record = DecisionResponseDTO(id=rec_id, proposal=proposal, decided_at=decided_at, rationale='init', quantity=1, entry_price=1.0)
        store.append_decision(record)
        strategy = self._make_strategy_mock()

        def fake_on_new_date(date_arg, positions, _add_position, remove_position):
            for pos in positions:
                remove_position(date_arg, pos, 0.5, 500.0, [200, 300])
        strategy.on_new_date = fake_on_new_date
        engine = self._build_engine(strategy, tmp_path, decision_store=store)
        with patch.object(strategy, 'get_current_underlying_price', return_value=500.0):
            success = engine.run()
        assert success is True
        strategy.on_remove_position_success.assert_called_once()
        args = strategy.on_remove_position_success.call_args[0]
        assert args[2] == 0.5
        assert args[3] == 500.0
        assert args[4] == [200, 300]

    def test_on_add_position_success_not_called_when_no_recommendation(self, monkeypatch, tmp_path):
        """Callback must not fire when the strategy produces no position."""
        monkeypatch.chdir(tmp_path)
        strategy = self._make_strategy_mock()
        strategy.on_new_date = lambda date, positions, add_pos, remove_pos: None
        store = JsonDecisionStore(base_dir=str(tmp_path / 'predictions'))
        engine = self._build_engine(strategy, tmp_path, decision_store=store)
        with patch.object(strategy, 'get_current_underlying_price', return_value=500.0):
            success = engine.run()
        assert success is True
        strategy.on_add_position_success.assert_not_called()


class _RecordingIndicator(Indicator):
    def __init__(self, name="recording", should_fail=False):
        super().__init__(name)
        self.should_fail = should_fail
        self.updates = []

    @property
    def warm_up_period(self) -> int:
        return 1

    def precompute(self, data):
        return None

    def update(self, date, data):
        self.updates.append((date, data))
        if self.should_fail:
            raise ValueError(f"{self.name} failed")

    def print(self):
        return None


class _IndicatorStrategy(Strategy):
    def on_new_date(self, date, positions, add_position, remove_position):
        return None

    def on_end(self, positions, remove_position, date):
        return None

    def validate_data(self, data):
        return True


class TestPaperTradingIndicatorUpdate:
    def _engine(self, indicator):
        strategy = _IndicatorStrategy()
        strategy.add_indicator(indicator)
        dates = pd.date_range("2024-01-01", periods=5, freq="D")
        strategy.set_data(pd.DataFrame({"Close": [1, 2, 3, 4, 5]}, index=dates))
        config = PaperTradingConfig(symbol="SPY", strategy_type="credit_spread")
        return PaperTradingEngine(strategy, config, options_handler=MagicMock())

    def test_updates_a_timestamp_missing_from_the_precomputed_series(self):
        indicator = _RecordingIndicator()
        engine = self._engine(indicator)
        run_date = datetime(2024, 6, 1, 15, 30)
        engine.update_indicators(run_date)
        assert indicator.updates[0][0] == run_date
        assert indicator.updates[0][1] is engine.strategy.data

    def test_skips_a_timestamp_already_precomputed(self):
        indicator = _RecordingIndicator()
        engine = self._engine(indicator)
        run_date = datetime(2024, 1, 3)
        indicator._values[run_date] = 1.0
        engine.update_indicators(run_date)
        assert indicator.updates == []

    def test_stops_on_the_first_failure(self):
        first = _RecordingIndicator(name="first")
        failing = _RecordingIndicator(name="failing", should_fail=True)
        later = _RecordingIndicator(name="later")
        strategy = _IndicatorStrategy()
        strategy.add_indicator(first)
        strategy.add_indicator(failing)
        strategy.add_indicator(later)
        dates = pd.date_range("2024-01-01", periods=3, freq="D")
        strategy.set_data(pd.DataFrame({"Close": [1, 2, 3]}, index=dates))
        engine = PaperTradingEngine(
            strategy,
            PaperTradingConfig(symbol="SPY", strategy_type="credit_spread"),
            options_handler=MagicMock(),
        )
        with pytest.raises(IndicatorUpdateError, match="failing"):
            engine.update_indicators(datetime(2024, 6, 1))
        assert len(first.updates) == 1
        assert later.updates == []

    def test_run_updates_indicators_before_the_recommender(self, monkeypatch, tmp_path):
        monkeypatch.chdir(tmp_path)
        indicator = _RecordingIndicator()
        strategy = _IndicatorStrategy()
        strategy.symbol = "SPY"
        strategy.add_indicator(indicator)
        dates = pd.date_range("2024-01-01", periods=3, freq="D")
        strategy.set_data(pd.DataFrame({"Close": [1, 2, 3]}, index=dates))
        config = PaperTradingConfig(
            symbol="SPY",
            strategy_type="credit_spread",
            api_key="test_api_key",
            use_free_tier=True,
            decision_store=JsonDecisionStore(base_dir=str(tmp_path / "predictions")),
            auto_yes=True,
        )
        engine = PaperTradingEngine(strategy, config, options_handler=MagicMock())
        seen = {}

        def _run(date):
            seen["updated_before_run"] = indicator.updates[0][0] == date

        with patch("algo_trading_engine.trade.InteractiveStrategyRecommender") as recommender_cls:
            recommender_cls.return_value.run.side_effect = _run
            recommender_cls.return_value.get_open_positions_status.return_value = []
            assert engine.run() is True
        assert seen["updated_before_run"] is True
