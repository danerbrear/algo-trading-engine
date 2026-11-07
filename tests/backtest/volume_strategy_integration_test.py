"""
Tests for enhanced current date volume validation in strategy.
"""
from datetime import datetime
from unittest.mock import Mock
from algo_trading_engine._internal.strategies.credit_spread_minimal import CreditSpreadStrategy
from algo_trading_engine.vo import create_position
from algo_trading_engine.enums import StrategyType
from algo_trading_engine.vo import Option
from algo_trading_engine.enums import OptionType

class TestStrategyCurrentDateVolumeValidation:
    """Test the new get_current_volumes_for_position method."""

    def setup_method(self):
        """Set up test fixtures."""
        self.mock_lstm_model = Mock()
        self.mock_lstm_scaler = Mock()
        self.mock_options_handler = Mock()
        self.get_contract_list_for_date = self.mock_options_handler.get_contract_list_for_date
        self.get_option_bar = self.mock_options_handler.get_option_bar
        self.get_options_chain = self.mock_options_handler.get_options_chain
        self.strategy = CreditSpreadStrategy(get_contract_list_for_date=self.get_contract_list_for_date, get_option_bar=self.get_option_bar, get_options_chain=self.get_options_chain, lstm_model=self.mock_lstm_model, lstm_scaler=self.mock_lstm_scaler, symbol='SPY', options_handler=self.mock_options_handler)
        self.option1 = Option(symbol='SPY240315C00500000', ticker='SPY', timestamp=datetime(2024, 1, 15), expiration='2024-03-15', strike=500.0, option_type=OptionType.CALL, last_price=1.5, volume=15)
        self.option2 = Option(symbol='SPY240315C00510000', ticker='SPY', timestamp=datetime(2024, 1, 15), expiration='2024-03-15', strike=510.0, option_type=OptionType.CALL, last_price=0.75, volume=8)
        self.position = create_position(symbol='SPY', expiration_date=datetime(2024, 3, 15), strategy_type=StrategyType.CALL_CREDIT_SPREAD, strike_price=500.0, entry_date=datetime(2024, 1, 15), entry_price=0.75, legs=[self.option1, self.option2])
        self.test_date = datetime(2024, 2, 15)

    def test_get_current_volumes_for_position_success(self):
        """Volumes come from the quotes already stored on the legs."""
        current_volumes = self.strategy.get_current_volumes_for_position(self.position, self.test_date)
        assert current_volumes == [15, 8]
        self.mock_options_handler.get_contract_list_for_date.assert_not_called()
        self.mock_options_handler.get_option_bar.assert_not_called()

    def test_get_current_volumes_for_position_no_volume_data(self):
        """A leg with no volume contributes None."""
        self.position.legs[0] = Option(symbol='SPY240315C00500000', ticker='SPY', timestamp=datetime(2024, 1, 15), expiration='2024-03-15', strike=500.0, option_type=OptionType.CALL, last_price=1.5, volume=None)
        self.position.legs[1] = Option(symbol='SPY240315C00510000', ticker='SPY', timestamp=datetime(2024, 1, 15), expiration='2024-03-15', strike=510.0, option_type=OptionType.CALL, last_price=0.75, volume=None)
        current_volumes = self.strategy.get_current_volumes_for_position(self.position, self.test_date)
        assert current_volumes == [None, None]

    def test_get_current_volumes_for_position_empty_position(self):
        """Test handling of position with no spread options."""
        empty_position = create_position(symbol='SPY', expiration_date=datetime(2024, 3, 15), strategy_type=StrategyType.CALL_CREDIT_SPREAD, strike_price=500.0, entry_date=datetime(2024, 1, 15), entry_price=0.75, legs=[])
        current_volumes = self.strategy.get_current_volumes_for_position(empty_position, self.test_date)
        assert current_volumes == []
        self.mock_options_handler.get_contract_list_for_date.assert_not_called()
        self.mock_options_handler.get_option_bar.assert_not_called()
