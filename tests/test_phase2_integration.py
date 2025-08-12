"""
Tests for Phase 2: Enhanced Current Date Volume Validation in BacktestEngine.
"""

import pytest
from datetime import datetime
from unittest.mock import Mock, patch

from src.backtest.main import BacktestEngine
from src.backtest.config import VolumeConfig, VolumeStats
from src.strategies.credit_spread_minimal import CreditSpreadStrategy
from src.backtest.models import Position, StrategyType
from src.common.models import Option, OptionType


class TestPhase2Integration:
    """Test the Phase 2 implementation of enhanced current date volume validation."""
    
    def setup_method(self):
        """Set up test fixtures."""
        # Create mock data
        self.mock_data = Mock()
        self.mock_data.index = [datetime(2024, 1, 15), datetime(2024, 2, 15)]
        self.mock_data.iloc = {0: {'Close': 500.0}, -1: {'Close': 510.0}}
        self.mock_data.loc = {datetime(2024, 1, 15): {'Close': 500.0}, datetime(2024, 2, 15): {'Close': 510.0}}
        
        # Create mock strategy
        self.mock_lstm_model = Mock()
        self.mock_lstm_scaler = Mock()
        self.mock_options_handler = Mock()
        
        self.strategy = CreditSpreadStrategy(
            lstm_model=self.mock_lstm_model,
            lstm_scaler=self.mock_lstm_scaler,
            options_handler=self.mock_options_handler
        )
        
        # Create test options
        self.option1 = Option(
            symbol="SPY240315C00500000",
            ticker="SPY",
            expiration="2024-03-15",
            strike=500.0,
            option_type=OptionType.CALL,
            last_price=1.50,
            volume=15
        )
        
        self.option2 = Option(
            symbol="SPY240315C00510000",
            ticker="SPY",
            expiration="2024-03-15",
            strike=510.0,
            option_type=OptionType.CALL,
            last_price=0.75,
            volume=8
        )
        
        # Create test position
        self.position = Position(
            symbol="SPY",
            expiration_date=datetime(2024, 3, 15),
            strategy_type=StrategyType.CALL_CREDIT_SPREAD,
            strike_price=500.0,
            entry_date=datetime(2024, 1, 15),
            entry_price=0.75,
            spread_options=[self.option1, self.option2]
        )
        self.position.set_quantity(1)  # Set quantity after creation
        
        # Create volume config
        self.volume_config = VolumeConfig(
            enable_volume_validation=True,
            min_volume=10,
            skip_closure_on_insufficient_volume=True
        )
        
        # Create backtest engine
        self.engine = BacktestEngine(
            data=self.mock_data,
            strategy=self.strategy,
            initial_capital=100000,
            volume_config=self.volume_config
        )
        
        # Add position to engine
        self.engine.positions.append(self.position)
        
        self.test_date = datetime(2024, 2, 15)
    
    def test_remove_position_with_sufficient_current_volume(self):
        """Test position closure with sufficient current date volume."""
        # Mock current volumes that are sufficient
        current_volumes = [25, 15]  # Both above min_volume of 10
        
        # Call _remove_position with current volumes
        self.engine._remove_position(
            date=self.test_date,
            position=self.position,
            exit_price=0.50,
            current_volumes=current_volumes
        )
        
        # Verify position was closed (removed from positions list)
        assert len(self.engine.positions) == 0
        assert self.engine.capital != 100000  # Capital should have changed
    
    def test_remove_position_with_insufficient_current_volume(self):
        """Test position closure with insufficient current date volume."""
        # Mock current volumes that are insufficient
        current_volumes = [5, 3]  # Both below min_volume of 10
        
        # Call _remove_position with current volumes
        self.engine._remove_position(
            date=self.test_date,
            position=self.position,
            exit_price=0.50,
            current_volumes=current_volumes
        )
        
        # Verify position was NOT closed (still in positions list)
        assert len(self.engine.positions) == 1
        assert self.engine.capital == 100000  # Capital should not have changed
        
        # Verify volume stats were updated
        assert self.engine.volume_stats.positions_rejected_closure_volume == 1
        assert self.engine.volume_stats.skipped_closures == 1
    
    def test_remove_position_with_mixed_current_volume(self):
        """Test position closure with mixed current date volume (some sufficient, some insufficient)."""
        # Mock current volumes with mixed results
        current_volumes = [15, 5]  # First sufficient, second insufficient
        
        # Call _remove_position with current volumes
        self.engine._remove_position(
            date=self.test_date,
            position=self.position,
            exit_price=0.50,
            current_volumes=current_volumes
        )
        
        # Verify position was NOT closed (still in positions list)
        assert len(self.engine.positions) == 1
        assert self.engine.capital == 100000  # Capital should not have changed
        
        # Verify volume stats were updated
        assert self.engine.volume_stats.positions_rejected_closure_volume == 1
        assert self.engine.volume_stats.skipped_closures == 1
    
    def test_remove_position_with_none_current_volumes(self):
        """Test position closure with None current volumes (no volume data available)."""
        # Mock current volumes as None
        current_volumes = [None, None]
        
        # Call _remove_position with current volumes
        self.engine._remove_position(
            date=self.test_date,
            position=self.position,
            exit_price=0.50,
            current_volumes=current_volumes
        )
        
        # Verify position was NOT closed (still in positions list)
        assert len(self.engine.positions) == 1
        assert self.engine.capital == 100000  # Capital should not have changed
        
        # Verify volume stats were updated
        assert self.engine.volume_stats.positions_rejected_closure_volume == 1
        assert self.engine.volume_stats.skipped_closures == 1
    
    def test_remove_position_with_volume_validation_disabled(self):
        """Test position closure with volume validation disabled."""
        # Create engine with volume validation disabled
        engine_disabled = BacktestEngine(
            data=self.mock_data,
            strategy=self.strategy,
            initial_capital=100000,
            volume_config=VolumeConfig(min_volume=10, enable_volume_validation=False)
        )
        engine_disabled.positions.append(self.position)
        
        # Mock current volumes that are insufficient
        current_volumes = [5, 3]  # Both below min_volume of 10
        
        # Call _remove_position with current volumes
        engine_disabled._remove_position(
            date=self.test_date,
            position=self.position,
            exit_price=0.50,
            current_volumes=current_volumes
        )
        
        # Verify position was closed (volume validation disabled)
        assert len(engine_disabled.positions) == 0
        assert engine_disabled.capital != 100000  # Capital should have changed
    
    def test_remove_position_without_current_volumes(self):
        """Test position closure without current volumes (backward compatibility)."""
        # Call _remove_position without current volumes
        self.engine._remove_position(
            date=self.test_date,
            position=self.position,
            exit_price=0.50
            # No current_volumes parameter
        )
        
        # Verify position was closed (no volume validation when no volumes provided)
        assert len(self.engine.positions) == 0
        assert self.engine.capital != 100000  # Capital should have changed
    
    
    
    def test_strategy_on_end_integration(self):
        """Test integration between strategy on_end and enhanced volume validation."""
        # Mock the get_current_volumes_for_position method
        self.strategy.get_current_volumes_for_position = Mock(return_value=[25, 15])
        
        # Mock the options_data
        self.strategy.options_data = {
            '2024-02-15': Mock()  # Mock option chain data
        }
        
        # Mock calculate_exit_price
        self.position.calculate_exit_price = Mock(return_value=0.50)
        
        # Create a mock remove_position function to capture calls
        mock_remove_position = Mock()
        
        # Call strategy on_end
        self.strategy.on_end([self.position], mock_remove_position, self.test_date)
        
        # Verify get_current_volumes_for_position was called
        self.strategy.get_current_volumes_for_position.assert_called_once_with(
            self.position, 
            self.test_date
        )
        
        # Verify remove_position was called with current_volumes
        mock_remove_position.assert_called_once()
        call_args = mock_remove_position.call_args
        assert call_args[0][0] == self.test_date  # date
        assert call_args[0][1] == self.position   # position
        assert call_args[0][2] == 0.50           # exit_price
        assert call_args[1]['current_volumes'] == [25, 15]  # current_volumes 