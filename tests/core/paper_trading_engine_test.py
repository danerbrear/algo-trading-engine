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

from algo_trading_engine.core.engine import PaperTradingEngine
from algo_trading_engine.models.config import PaperTradingConfig
from algo_trading_engine.prediction.decision_store import (
    DecisionStore,
    JsonDecisionStore, 
    ProposedPositionRequestDTO, 
    DecisionResponseDTO,
    generate_decision_id
)
from algo_trading_engine.common.models import StrategyType
from algo_trading_engine.common.models import Option


def _make_option(symbol: str, strike: float, expiration: str, opt_type: str, last: float, volume: int = 100) -> Option:
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
        'volume': volume,
    })


@pytest.fixture
def mock_strategy():
    """Create a mock strategy for testing."""
    strategy = MagicMock()
    strategy.symbol = 'SPY'
    
    # Create a proper class name for strategy name extraction
    strategy.__class__ = type('CreditSpreadStrategy', (), {})
    
    # Mock data attribute
    dates = pd.date_range(start='2025-01-01', end='2025-08-08', freq='D')
    data = pd.DataFrame({
        'Close': [500.0] * len(dates),
        'Open': [499.0] * len(dates),
        'High': [501.0] * len(dates),
        'Low': [498.0] * len(dates),
    }, index=dates)
    strategy.data = data
    
    # Mock methods
    strategy.set_data = Mock()
    strategy.recommend_open_position.return_value = None  # Default: no recommendation
    strategy.recommend_close_positions.return_value = []  # Default: no closes
    
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
    return PaperTradingConfig(
        symbol='SPY',
        strategy_type='credit_spread',
        api_key='test_api_key',
        use_free_tier=True,
    )


@pytest.fixture
def decision_store_with_allocations(tmp_path):
    """Create a decision store and initialize capital allocations config."""
    # Create predictions/decisions directory for store
    decisions_path = tmp_path / "predictions" / "decisions"
    decisions_path.mkdir(parents=True, exist_ok=True)
    
    store = JsonDecisionStore(base_dir=str(decisions_path.parent))
    
    # Create capital allocations config file
    config_path = tmp_path / "config" / "strategies"
    config_path.mkdir(parents=True, exist_ok=True)
    config_file = config_path / "capital_allocations.json"
    
    import json
    config_data = {
        "strategies": {
            "credit_spread": {
                "allocated_capital": 10000.0,
                "max_risk_percentage": 0.05
            }
        }
    }
    with open(config_file, 'w') as f:
        json.dump(config_data, f)
    
    return store, str(config_file)


def test_paper_trading_engine_run_no_open_positions(
    mock_strategy, 
    mock_options_handler, 
    paper_trading_config,
    decision_store_with_allocations,
    monkeypatch,
    tmp_path
):
    """Test PaperTradingEngine.run() when there are no open positions.
    
    This test verifies that the engine runs successfully when there are no
    existing open positions to check.
    """
    store, config_file = decision_store_with_allocations
    
    # Change directory context for the test
    monkeypatch.chdir(tmp_path)
    
    # Create engine
    engine = PaperTradingEngine(
        strategy=mock_strategy,
        config=paper_trading_config,
        options_handler=mock_options_handler
    )
    
    # Mock to avoid API calls: strategy price lookup, recommender, and DataRetriever
    with patch.object(mock_strategy, 'get_current_underlying_price', return_value=500.0), \
         patch('algo_trading_engine.common.data_retriever.DataRetriever.get_live_price', return_value=500.0), \
         patch('algo_trading_engine.prediction.recommendation_engine.InteractiveStrategyRecommender') as mock_recommender_class:
        mock_recommender = Mock()
        mock_recommender.get_open_positions_status.return_value = []
        mock_recommender.run.return_value = None
        mock_recommender_class.return_value = mock_recommender
        
        # Run engine
        success = engine.run()
        
        # Verify execution
        assert success is True
        mock_recommender_class.assert_called_once()
        mock_recommender.run.assert_called_once()
        # get_open_positions_status should NOT be called when there are no open positions
        mock_recommender.get_open_positions_status.assert_not_called()


def test_paper_trading_engine_run_with_open_positions(
    mock_strategy, 
    mock_options_handler, 
    paper_trading_config,
    decision_store_with_allocations,
    monkeypatch,
    tmp_path
):
    """Test PaperTradingEngine.run() when there are existing open positions.
    
    This test would have caught the bug where 'recommender' was used before
    being defined when open positions exist.
    """
    store, config_file = decision_store_with_allocations
    
    # Change directory context for the test
    monkeypatch.chdir(tmp_path)
    
    # Add an open position to a store at the default path that the engine will use
    # The engine creates JsonDecisionStore() with default "predictions/decisions" path
    default_store = JsonDecisionStore()  # Uses default path relative to tmp_path
    
    legs = (
        _make_option('SPY_PUT_500', 500, '2025-09-06', 'put', 2.0),
        _make_option('SPY_PUT_495', 495, '2025-09-06', 'put', 1.0),
    )
    proposal = ProposedPositionRequestDTO(
        symbol='SPY',
        strategy_type=StrategyType.PUT_CREDIT_SPREAD,
        legs=legs,
        credit=1.0,
        width=5.0,
        probability_of_profit=0.68,
        confidence=0.7,
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
        rationale='test position',
        quantity=1,
        entry_price=1.0,
    )
    default_store.append_decision(record)
    
    # Verify position was added
    open_positions = default_store.get_open_positions(symbol='SPY')
    assert len(open_positions) == 1, "Should have one open position"
    
    # Create engine
    engine = PaperTradingEngine(
        strategy=mock_strategy,
        config=paper_trading_config,
        options_handler=mock_options_handler
    )
    
    # Mock to avoid API calls: strategy price lookup, recommender, and DataRetriever
    with patch.object(mock_strategy, 'get_current_underlying_price', return_value=500.0), \
         patch('algo_trading_engine.common.data_retriever.DataRetriever.get_live_price', return_value=500.0), \
         patch('algo_trading_engine.prediction.recommendation_engine.InteractiveStrategyRecommender') as mock_recommender_class:
        mock_recommender = Mock()
        
        # Mock get_open_positions_status to return status info
        mock_recommender.get_open_positions_status.return_value = [{
            'symbol': 'SPY',
            'strategy_type': 'PUT_CREDIT_SPREAD',
            'quantity': 1,
            'entry_price': 1.0,
            'exit_price': 0.5,
            'pnl_dollars': 50.0,
            'pnl_percent': 0.50,
            'days_held': 38,
            'dte': 29,
        }]
        mock_recommender.run.return_value = None
        mock_recommender_class.return_value = mock_recommender
        
        # Run engine - this would fail with UnboundLocalError before the fix
        success = engine.run()
        
        # Verify execution
        assert success is True
        mock_recommender_class.assert_called_once()
        # CRITICAL: get_open_positions_status must be called AFTER recommender is created
        # This is the key assertion that would have caught the bug!
        mock_recommender.get_open_positions_status.assert_called_once()
        mock_recommender.run.assert_called_once()


def test_paper_trading_engine_run_with_recommendation_engine_failure(
    mock_strategy, 
    mock_options_handler, 
    paper_trading_config,
    decision_store_with_allocations,
    monkeypatch,
    tmp_path
):
    """Test PaperTradingEngine.run() handles recommendation engine failures gracefully."""
    store, config_file = decision_store_with_allocations
    
    # Change directory context for the test
    monkeypatch.chdir(tmp_path)
    
    # Create engine
    engine = PaperTradingEngine(
        strategy=mock_strategy,
        config=paper_trading_config,
        options_handler=mock_options_handler
    )
    
    # Mock to avoid API calls: strategy price lookup, recommender, and DataRetriever
    with patch.object(mock_strategy, 'get_current_underlying_price', return_value=500.0), \
         patch('algo_trading_engine.common.data_retriever.DataRetriever.get_live_price', return_value=500.0), \
         patch('algo_trading_engine.prediction.recommendation_engine.InteractiveStrategyRecommender') as mock_recommender_class:
        mock_recommender = Mock()
        mock_recommender.run.side_effect = Exception("Test error")
        mock_recommender_class.return_value = mock_recommender
        
        # Run engine
        success = engine.run()
        
        # Verify execution failed gracefully
        assert success is False


def test_paper_trading_engine_run_without_options_handler(
    mock_strategy, 
    paper_trading_config,
    decision_store_with_allocations,
    monkeypatch,
    tmp_path
):
    """Test PaperTradingEngine.run() fails gracefully without options handler."""
    store, config_file = decision_store_with_allocations
    
    # Change directory context for the test
    monkeypatch.chdir(tmp_path)
    
    # Ensure mock_strategy doesn't have options_handler attribute
    if hasattr(mock_strategy, 'options_handler'):
        delattr(mock_strategy, 'options_handler')
    
    # Create engine without options handler
    engine = PaperTradingEngine(
        strategy=mock_strategy,
        config=paper_trading_config,
        options_handler=None
    )
    
    # Run engine
    success = engine.run()
    
    # Verify execution failed gracefully
    assert success is False


def test_paper_trading_engine_run_missing_capital_config(
    mock_strategy, 
    mock_options_handler, 
    paper_trading_config,
    tmp_path,
    monkeypatch
):
    """Test PaperTradingEngine.run() handles missing capital config gracefully."""
    # Create store without capital allocations config
    store = JsonDecisionStore(base_dir=str(tmp_path))
    
    # Change directory context for the test
    monkeypatch.chdir(tmp_path)
    
    # Create engine
    engine = PaperTradingEngine(
        strategy=mock_strategy,
        config=paper_trading_config,
        options_handler=mock_options_handler
    )
    
    # Mock to avoid API calls: strategy price lookup, recommender, and DataRetriever
    with patch.object(mock_strategy, 'get_current_underlying_price', return_value=500.0), \
         patch('algo_trading_engine.common.data_retriever.DataRetriever.get_live_price', return_value=500.0), \
         patch('algo_trading_engine.prediction.recommendation_engine.InteractiveStrategyRecommender') as mock_recommender_class:
        mock_recommender = Mock()
        mock_recommender.get_open_positions_status.return_value = []
        mock_recommender.run.return_value = None
        mock_recommender_class.return_value = mock_recommender
        
        success = engine.run()
        
        # Should succeed after initializing default config
        assert success is True


def test_paper_trading_engine_strategy_name_extraction(
    mock_options_handler, 
    paper_trading_config,
    tmp_path,
    monkeypatch
):
    """Test that strategy name is correctly extracted from strategy class."""
    # Test various strategy class names
    test_cases = [
        ('CreditSpreadStrategy', 'credit_spread'),
        ('VelocitySignalMomentumStrategy', 'velocity_signal_momentum'),
        ('MyCustomStrategy', 'my_custom'),
    ]
    
    for class_name, expected_name in test_cases:
        # Create mock strategy with specific class name
        strategy = MagicMock()
        strategy.symbol = 'SPY'
        strategy.__class__ = type(class_name, (), {})
        
        dates = pd.date_range(start='2025-01-01', end='2025-08-08', freq='D')
        strategy.data = pd.DataFrame({
            'Close': [500.0] * len(dates),
        }, index=dates)
        
        # Create engine
        engine = PaperTradingEngine(
            strategy=strategy,
            config=paper_trading_config,
            options_handler=mock_options_handler
        )
        
        # Extract strategy name
        strategy_name = engine._get_strategy_name_from_class()
        
        # For velocity_signal_momentum, the mapping should convert it to velocity_momentum
        if expected_name == 'velocity_signal_momentum':
            # Check that it's either velocity_signal_momentum or velocity_momentum
            # depending on mapping
            assert strategy_name in ['velocity_signal_momentum', 'velocity_momentum']
        else:
            assert strategy_name == expected_name


class TestPaperTradingEngineFromConfig:
    """Test PaperTradingEngine.from_config() factory method."""

    @patch('algo_trading_engine.common.options_handler.OptionsHandler')
    @patch('algo_trading_engine.common.data_retriever.DataRetriever')
    def test_from_config_sets_symbol_on_strategy_instance(
        self, mock_data_retriever, mock_options_handler
    ):
        """Test that symbol is set on strategy when strategy instance is provided."""
        mock_strategy = MagicMock()
        mock_strategy.set_data = Mock()
        mock_strategy.options_handler = None
        mock_strategy.warm_up_period = 0

        mock_retriever_instance = MagicMock()
        mock_retriever_instance.treasury_rates = None
        mock_retriever_instance.fetch_data_for_period.return_value = pd.DataFrame({
            'Close': [500.0] * 10,
            'Open': [499.0] * 10,
            'High': [501.0] * 10,
            'Low': [498.0] * 10,
        }, index=pd.date_range('2025-01-01', periods=10, freq='D'))
        mock_data_retriever.return_value = mock_retriever_instance

        config = PaperTradingConfig(
            symbol='QQQ',
            strategy_type=mock_strategy,
            api_key='test_key',
            use_free_tier=True,
        )

        engine = PaperTradingEngine.from_config(config)

        assert engine.strategy == mock_strategy
        assert mock_strategy.symbol == 'QQQ'


class TestCustomDecisionStore:
    """Test that a custom DecisionStore passed via PaperTradingConfig is used by the engine."""

    def test_engine_uses_custom_decision_store_from_config(
        self, mock_strategy, mock_options_handler, monkeypatch, tmp_path
    ):
        """When PaperTradingConfig.decision_store is set, the engine must use it
        instead of creating a default JsonDecisionStore."""
        monkeypatch.chdir(tmp_path)

        custom_store = MagicMock(spec=DecisionStore)
        custom_store.get_open_positions.return_value = []

        config = PaperTradingConfig(
            symbol='SPY',
            strategy_type='credit_spread',
            api_key='test_api_key',
            use_free_tier=True,
            decision_store=custom_store,
        )

        engine = PaperTradingEngine(
            strategy=mock_strategy,
            config=config,
            options_handler=mock_options_handler,
        )

        with patch.object(mock_strategy, 'get_current_underlying_price', return_value=500.0), \
             patch('algo_trading_engine.common.data_retriever.DataRetriever.get_live_price', return_value=500.0), \
             patch('algo_trading_engine.prediction.recommendation_engine.InteractiveStrategyRecommender') as mock_rec_cls:
            mock_recommender = Mock()
            mock_recommender.run.return_value = None
            mock_rec_cls.return_value = mock_recommender

            success = engine.run()

            assert success is True

            # The custom store (not a default JsonDecisionStore) should have been
            # passed to CapitalManager and the recommender.
            mock_rec_cls.assert_called_once()
            _, call_kwargs = mock_rec_cls.call_args
            if not call_kwargs:
                call_args = mock_rec_cls.call_args[0]
                assert call_args[1] is custom_store, "Recommender should receive the custom DecisionStore"
            else:
                assert call_kwargs.get('decision_store', mock_rec_cls.call_args[0][1]) is custom_store

            custom_store.get_open_positions.assert_called_once_with(symbol='SPY')

    def test_engine_falls_back_to_json_store_when_none(
        self, mock_strategy, mock_options_handler, monkeypatch, tmp_path
    ):
        """When PaperTradingConfig.decision_store is None, the engine creates
        a default JsonDecisionStore."""
        monkeypatch.chdir(tmp_path)

        config = PaperTradingConfig(
            symbol='SPY',
            strategy_type='credit_spread',
            api_key='test_api_key',
            use_free_tier=True,
            decision_store=None,
        )

        engine = PaperTradingEngine(
            strategy=mock_strategy,
            config=config,
            options_handler=mock_options_handler,
        )

        with patch.object(mock_strategy, 'get_current_underlying_price', return_value=500.0), \
             patch('algo_trading_engine.common.data_retriever.DataRetriever.get_live_price', return_value=500.0), \
             patch('algo_trading_engine.prediction.recommendation_engine.InteractiveStrategyRecommender') as mock_rec_cls:
            mock_recommender = Mock()
            mock_recommender.run.return_value = None
            mock_rec_cls.return_value = mock_recommender

            success = engine.run()

            assert success is True

            # A JsonDecisionStore should have been created and passed
            call_args = mock_rec_cls.call_args[0]
            assert isinstance(call_args[1], JsonDecisionStore), (
                "Recommender should receive a JsonDecisionStore when config.decision_store is None"
            )
