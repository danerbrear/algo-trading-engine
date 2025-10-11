"""
Tests for enhanced current date volume validation in strategy.
"""

from datetime import datetime
from unittest.mock import Mock

from src.strategies.credit_spread_minimal import CreditSpreadStrategy
from src.backtest.models import Position, StrategyType
from src.common.models import Option, OptionType


class TestStrategyCurrentDateVolumeValidation:
    """Test the new get_current_volumes_for_position method."""
    
    def setup_method(self):
        """Set up test fixtures."""
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
        
        self.test_date = datetime(2024, 2, 15)
    
    def test_get_current_volumes_for_position_success(self):
        """Test successful volume data fetching for position."""
        # Mock the options handler to return fresh volume data
        fresh_option1 = Mock()
        fresh_option1.volume = 25
        
        fresh_option2 = Mock()
        fresh_option2.volume = 12
        
        self.mock_options_handler.get_specific_option_contract.side_effect = [
            fresh_option1,  # For option1
            fresh_option2   # For option2
        ]
        
        # Call the method
        current_volumes = self.strategy.get_current_volumes_for_position(
            self.position, 
            self.test_date
        )
        
        # Verify results
        assert current_volumes == [25, 12]
        assert self.mock_options_handler.get_specific_option_contract.call_count == 2
        
        # Verify the calls were made with correct parameters
        calls = self.mock_options_handler.get_specific_option_contract.call_args_list
        assert calls[0][0] == (500.0, "2024-03-15", "call", self.test_date)
        assert calls[1][0] == (510.0, "2024-03-15", "call", self.test_date)
    
    def test_get_current_volumes_for_position_no_volume_data(self):
        """Test handling when no volume data is available."""
        # Mock the options handler to return None for volume data
        fresh_option1 = Mock()
        fresh_option1.volume = None
        
        fresh_option2 = Mock()
        fresh_option2.volume = None
        
        self.mock_options_handler.get_specific_option_contract.side_effect = [
            fresh_option1,  # For option1
            fresh_option2   # For option2
        ]
        
        # Call the method
        current_volumes = self.strategy.get_current_volumes_for_position(
            self.position, 
            self.test_date
        )
        
        # Verify results
        assert current_volumes == [None, None]
    
    def test_get_current_volumes_for_position_api_failure(self):
        """Test handling of API failures during volume data fetching."""
        # Mock the options handler to raise an exception
        self.mock_options_handler.get_specific_option_contract.side_effect = Exception("API Error")
        
        # Call the method
        current_volumes = self.strategy.get_current_volumes_for_position(
            self.position, 
            self.test_date
        )
        
        # Verify results - should return None for both options due to exception
        assert current_volumes == [None, None]
    
    def test_get_current_volumes_for_position_mixed_results(self):
        """Test handling of mixed results (some success, some failure)."""
        # Mock the options handler to return mixed results
        fresh_option1 = Mock()
        fresh_option1.volume = 30
        
        fresh_option2 = Mock()
        fresh_option2.volume = None
        
        self.mock_options_handler.get_specific_option_contract.side_effect = [
            fresh_option1,  # For option1 - success
            fresh_option2   # For option2 - no volume data
        ]
        
        # Call the method
        current_volumes = self.strategy.get_current_volumes_for_position(
            self.position, 
            self.test_date
        )
        
        # Verify results
        assert current_volumes == [30, None]
    
    def test_get_current_volumes_for_position_empty_position(self):
        """Test handling of position with no spread options."""
        # Create position with no spread options
        empty_position = Position(
            symbol="SPY",
            expiration_date=datetime(2024, 3, 15),
            strategy_type=StrategyType.CALL_CREDIT_SPREAD,
            strike_price=500.0,
            entry_date=datetime(2024, 1, 15),
            entry_price=0.75,
            spread_options=[]  # Empty list
        )
        
        # Call the method
        current_volumes = self.strategy.get_current_volumes_for_position(
            empty_position, 
            self.test_date
        )
        
        # Verify results
        assert current_volumes == []
        # Verify no API calls were made
        self.mock_options_handler.get_specific_option_contract.assert_not_called()
    
    def test_get_current_volumes_for_position_none_options_handler(self):
        """Test handling when options_handler is None."""
        # Create strategy without options_handler
        strategy_no_handler = CreditSpreadStrategy(
            lstm_model=self.mock_lstm_model,
            lstm_scaler=self.mock_lstm_scaler,
            options_handler=None
        )
        
        # Call the method - should handle gracefully and return None values
        current_volumes = strategy_no_handler.get_current_volumes_for_position(
            self.position, 
            self.test_date
        )
        
        # Verify results - should return None for both options due to AttributeError
        assert current_volumes == [None, None] 