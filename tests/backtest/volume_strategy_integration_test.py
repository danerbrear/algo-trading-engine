"""
Tests for enhanced current date volume validation in strategy.
"""

from datetime import datetime, date
from decimal import Decimal
from unittest.mock import Mock

from algo_trading_engine.strategies.credit_spread_minimal import CreditSpreadStrategy
from algo_trading_engine.backtest.models import Position, StrategyType
from algo_trading_engine.common.models import Option, OptionType
from algo_trading_engine.common.options_dtos import OptionContractDTO, OptionBarDTO, StrikePrice, ExpirationDate


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
        # Create mock contracts and bars for new API
        from algo_trading_engine.common.options_dtos import StrikeRangeDTO, ExpirationRangeDTO
        
        contract1 = OptionContractDTO(
            ticker='O:SPY240315C00500000',
            underlying_ticker='SPY',
            contract_type=OptionType.CALL,
            strike_price=StrikePrice(Decimal('500.0')),
            expiration_date=ExpirationDate(date(2024, 3, 15)),
            exercise_style='american',
            shares_per_contract=100
        )
        contract2 = OptionContractDTO(
            ticker='O:SPY240315C00510000',
            underlying_ticker='SPY',
            contract_type=OptionType.CALL,
            strike_price=StrikePrice(Decimal('510.0')),
            expiration_date=ExpirationDate(date(2024, 3, 15)),
            exercise_style='american',
            shares_per_contract=100
        )
        
        bar1 = OptionBarDTO(
            ticker='O:SPY240315C00500000',
            timestamp=self.test_date,
            open_price=Decimal('1.50'),
            high_price=Decimal('1.55'),
            low_price=Decimal('1.45'),
            close_price=Decimal('1.50'),
            volume=25,
            volume_weighted_avg_price=Decimal('1.50'),
            number_of_transactions=100,
            adjusted=True
        )
        bar2 = OptionBarDTO(
            ticker='O:SPY240315C00510000',
            timestamp=self.test_date,
            open_price=Decimal('0.75'),
            high_price=Decimal('0.80'),
            low_price=Decimal('0.70'),
            close_price=Decimal('0.75'),
            volume=12,
            volume_weighted_avg_price=Decimal('0.75'),
            number_of_transactions=50,
            adjusted=True
        )
        
        # Mock get_contract_list_for_date to return contracts based on strike
        def get_contracts_side_effect(*args, **kwargs):
            strike_val = float(kwargs['strike_range'].min_strike.value)
            if strike_val == 500.0:
                return [contract1]
            elif strike_val == 510.0:
                return [contract2]
            return []
        
        self.mock_options_handler.get_contract_list_for_date.side_effect = get_contracts_side_effect
        
        # Mock get_option_bar to return bars based on contract
        def get_bar_side_effect(contract, date):
            if contract.strike_price.value == Decimal('500.0'):
                return bar1
            elif contract.strike_price.value == Decimal('510.0'):
                return bar2
            return None
        
        self.mock_options_handler.get_option_bar.side_effect = get_bar_side_effect
        
        # Call the method
        current_volumes = self.strategy.get_current_volumes_for_position(
            self.position, 
            self.test_date
        )
        
        # Verify results
        assert current_volumes == [25, 12]
        assert self.mock_options_handler.get_contract_list_for_date.call_count == 2
        assert self.mock_options_handler.get_option_bar.call_count == 2
    
    def test_get_current_volumes_for_position_no_volume_data(self):
        """Test handling when no volume data is available."""
        # Mock contracts but return None for bars (no volume data)
        from algo_trading_engine.common.options_dtos import StrikeRangeDTO, ExpirationRangeDTO, StrikePrice, ExpirationDate
        
        contract1 = OptionContractDTO(
            ticker='O:SPY240315C00500000',
            underlying_ticker='SPY',
            contract_type=OptionType.CALL,
            strike_price=StrikePrice(Decimal('500.0')),
            expiration_date=ExpirationDate(date(2024, 3, 15)),
            exercise_style='american',
            shares_per_contract=100
        )
        contract2 = OptionContractDTO(
            ticker='O:SPY240315C00510000',
            underlying_ticker='SPY',
            contract_type=OptionType.CALL,
            strike_price=StrikePrice(Decimal('510.0')),
            expiration_date=ExpirationDate(date(2024, 3, 15)),
            exercise_style='american',
            shares_per_contract=100
        )
        
        def get_contracts_side_effect(*args, **kwargs):
            strike_val = float(kwargs['strike_range'].min_strike.value)
            if strike_val == 500.0:
                return [contract1]
            elif strike_val == 510.0:
                return [contract2]
            return []
        
        self.mock_options_handler.get_contract_list_for_date.side_effect = get_contracts_side_effect
        self.mock_options_handler.get_option_bar.return_value = None  # No bar data
        
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
        self.mock_options_handler.get_contract_list_for_date.side_effect = Exception("API Error")
        
        # Call the method
        current_volumes = self.strategy.get_current_volumes_for_position(
            self.position, 
            self.test_date
        )
        
        # Verify results - should return None for both options due to exception
        assert current_volumes == [None, None]
    
    def test_get_current_volumes_for_position_mixed_results(self):
        """Test handling of mixed results (some success, some failure)."""
        # Create mock contracts
        from algo_trading_engine.common.options_dtos import StrikeRangeDTO, ExpirationRangeDTO, StrikePrice, ExpirationDate
        
        contract1 = OptionContractDTO(
            ticker='O:SPY240315C00500000',
            underlying_ticker='SPY',
            contract_type=OptionType.CALL,
            strike_price=StrikePrice(Decimal('500.0')),
            expiration_date=ExpirationDate(date(2024, 3, 15)),
            exercise_style='american',
            shares_per_contract=100
        )
        contract2 = OptionContractDTO(
            ticker='O:SPY240315C00510000',
            underlying_ticker='SPY',
            contract_type=OptionType.CALL,
            strike_price=StrikePrice(Decimal('510.0')),
            expiration_date=ExpirationDate(date(2024, 3, 15)),
            exercise_style='american',
            shares_per_contract=100
        )
        
        bar1 = OptionBarDTO(
            ticker='O:SPY240315C00500000',
            timestamp=self.test_date,
            open_price=Decimal('1.50'),
            high_price=Decimal('1.55'),
            low_price=Decimal('1.45'),
            close_price=Decimal('1.50'),
            volume=30,
            volume_weighted_avg_price=Decimal('1.50'),
            number_of_transactions=100,
            adjusted=True
        )
        
        def get_contracts_side_effect(*args, **kwargs):
            strike_val = float(kwargs['strike_range'].min_strike.value)
            if strike_val == 500.0:
                return [contract1]
            elif strike_val == 510.0:
                return [contract2]
            return []
        
        def get_bar_side_effect(contract, date):
            if contract.strike_price.value == Decimal('500.0'):
                return bar1  # Success
            elif contract.strike_price.value == Decimal('510.0'):
                return None  # No bar data
            return None
        
        self.mock_options_handler.get_contract_list_for_date.side_effect = get_contracts_side_effect
        self.mock_options_handler.get_option_bar.side_effect = get_bar_side_effect
        
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
        self.mock_options_handler.get_contract_list_for_date.assert_not_called()
        self.mock_options_handler.get_option_bar.assert_not_called()
    
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
