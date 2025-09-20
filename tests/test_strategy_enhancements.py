"""
Unit tests for enhanced strategy methods
"""
from datetime import datetime
from unittest.mock import Mock, patch
import pandas as pd

from src.strategies.velocity_signal_momentum_strategy import VelocitySignalMomentumStrategy
from src.strategies.credit_spread_minimal import CreditSpreadStrategy
from src.common.models import Option, OptionType
from src.backtest.models import Position, StrategyType


class TestVelocitySignalMomentumStrategyEnhancements:
    """Test cases for enhanced VelocitySignalMomentumStrategy methods"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.options_handler = Mock()
        self.strategy = VelocitySignalMomentumStrategy(
            options_handler=self.options_handler,
            start_date_offset=60
        )
        
        # Create sample data
        dates = pd.date_range(start='2024-01-01', periods=100, freq='D')
        self.sample_data = pd.DataFrame({
            'Close': [100 + i * 0.5 for i in range(100)],
            'Open': [100 + i * 0.5 for i in range(100)],
            'High': [101 + i * 0.5 for i in range(100)],
            'Low': [99 + i * 0.5 for i in range(100)],
            'Volume': [1000000] * 100
        }, index=dates)
        
        self.strategy.set_data(self.sample_data, {})
    
    def test_get_current_underlying_price_historical_date(self):
        """Test _get_current_underlying_price with historical date"""
        test_date = datetime(2024, 1, 15)
        price = self.strategy._get_current_underlying_price(test_date)
        
        # Should return the close price for that date
        expected_price = self.sample_data.loc[test_date, 'Close']
        assert price == expected_price
    
    @patch('src.strategies.velocity_signal_momentum_strategy.datetime')
    def test_get_current_underlying_price_current_date_with_live_price(self, mock_datetime):
        """Test _get_current_underlying_price with current date and live price available"""
        # Mock current date
        current_date = datetime(2024, 1, 20)
        mock_datetime.now.return_value = current_date
        
        # Mock data retriever with live price
        mock_data_retriever = Mock()
        mock_data_retriever.get_live_price.return_value = 150.0
        self.strategy.data_retriever = mock_data_retriever
        
        price = self.strategy._get_current_underlying_price(current_date)
        
        assert price == 150.0
        mock_data_retriever.get_live_price.assert_called_once()
    
    @patch('src.strategies.velocity_signal_momentum_strategy.datetime')
    def test_get_current_underlying_price_current_date_fallback(self, mock_datetime):
        """Test _get_current_underlying_price with current date but no live price"""
        # Mock current date that's NOT in the sample data
        current_date = datetime(2024, 5, 1)  # This date is not in the sample data
        mock_datetime.now.return_value = current_date
        
        # Mock data retriever without live price
        mock_data_retriever = Mock()
        mock_data_retriever.get_live_price.return_value = None
        self.strategy.data_retriever = mock_data_retriever
        
        # Also mock the options_handler to prevent fallback
        self.strategy.options_handler = None
        
        price = self.strategy._get_current_underlying_price(current_date)
        
        # Should return None since no live price and date not in data
        assert price is None
    
    def test_recalculate_moving_averages(self):
        """Test _recalculate_moving_averages method"""
        # Ensure data has moving averages
        self.strategy._recalculate_moving_averages()
        
        # Check that moving averages are calculated
        assert 'SMA_15' in self.strategy.data.columns
        assert 'SMA_30' in self.strategy.data.columns
        assert 'MA_Velocity_15_30' in self.strategy.data.columns
        assert 'Velocity_Changes' in self.strategy.data.columns
        
        # Check that SMA_15 is calculated correctly for valid periods
        valid_sma_15 = self.strategy.data['SMA_15'].dropna()
        assert len(valid_sma_15) > 0
        
        # Check that SMA_30 is calculated correctly for valid periods
        valid_sma_30 = self.strategy.data['SMA_30'].dropna()
        assert len(valid_sma_30) > 0
    
    def test_check_backward_trend_success_upward_trend(self):
        """Test _check_backward_trend_success with upward trend"""
        # Create data with clear upward trend
        dates = pd.date_range(start='2024-01-01', periods=50, freq='D')
        upward_data = pd.DataFrame({
            'Close': [100 + i * 2 for i in range(50)],  # Clear upward trend
            'Open': [100 + i * 2 for i in range(50)],
            'High': [101 + i * 2 for i in range(50)],
            'Low': [99 + i * 2 for i in range(50)],
            'Volume': [1000000] * 50
        }, index=dates)
        
        # Test with signal at index 20 (should detect upward trend)
        success, duration, return_val = self.strategy._check_backward_trend_success(
            upward_data, 20, 'up', min_duration=3, max_duration=10
        )
        
        assert success is True
        assert duration > 0
        assert return_val > 0
    
    def test_check_backward_trend_success_no_trend(self):
        """Test _check_backward_trend_success with no clear trend"""
        # Create data with no clear trend
        dates = pd.date_range(start='2024-01-01', periods=50, freq='D')
        flat_data = pd.DataFrame({
            'Close': [100] * 50,  # No trend
            'Open': [100] * 50,
            'High': [101] * 50,
            'Low': [99] * 50,
            'Volume': [1000000] * 50
        }, index=dates)
        
        # Test with signal at index 20
        success, duration, return_val = self.strategy._check_backward_trend_success(
            flat_data, 20, 'up', min_duration=3, max_duration=10
        )
        
        assert success is False
        assert duration == 0
        assert return_val == 0.0
    
    def test_check_trend_success_always_uses_backward(self):
        """Test that _check_trend_success always calls _check_backward_trend_success"""
        with patch.object(self.strategy, '_check_backward_trend_success') as mock_backward:
            mock_backward.return_value = (True, 5, 0.05)
            
            result = self.strategy._check_trend_success(
                self.sample_data, 20, 'up', min_duration=3, max_duration=10
            )
            
            # Check that the method was called with the correct arguments
            mock_backward.assert_called_once()
            call_args = mock_backward.call_args
            assert call_args[0][0].equals(self.sample_data)  # First argument (data)
            assert call_args[0][1] == 20  # Second argument (signal_index)
            assert call_args[0][2] == 'up'  # Third argument (trend_type)
            assert call_args[0][3] == 3  # Fourth argument (min_duration)
            assert call_args[0][4] == 10  # Fifth argument (max_duration)
            assert result == (True, 5, 0.05)


class TestCreditSpreadStrategyEnhancements:
    """Test cases for enhanced CreditSpreadStrategy methods"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.lstm_model = Mock()
        self.lstm_scaler = Mock()
        self.options_handler = Mock()
        
        self.strategy = CreditSpreadStrategy(
            lstm_model=self.lstm_model,
            lstm_scaler=self.lstm_scaler,
            options_handler=self.options_handler,
            start_date_offset=0
        )
        
        # Create sample data
        dates = pd.date_range(start='2024-01-01', periods=100, freq='D')
        self.sample_data = pd.DataFrame({
            'Close': [100 + i * 0.5 for i in range(100)],
            'Open': [100 + i * 0.5 for i in range(100)],
            'High': [101 + i * 0.5 for i in range(100)],
            'Low': [99 + i * 0.5 for i in range(100)],
            'Volume': [1000000] * 100
        }, index=dates)
        
        self.strategy.set_data(self.sample_data, {})
    
    def test_get_current_underlying_price_historical_date(self):
        """Test _get_current_underlying_price with historical date"""
        test_date = datetime(2024, 1, 15)
        price = self.strategy._get_current_underlying_price(test_date)
        
        # Should return the close price for that date
        expected_price = self.sample_data.loc[test_date, 'Close']
        assert price == expected_price
    
    @patch('src.strategies.credit_spread_minimal.datetime')
    def test_get_current_underlying_price_current_date_with_live_price(self, mock_datetime):
        """Test _get_current_underlying_price with current date and live price available"""
        # Mock current date
        current_date = datetime(2024, 1, 20)
        mock_datetime.now.return_value = current_date
        
        # Mock data retriever with live price
        mock_data_retriever = Mock()
        mock_data_retriever.get_live_price.return_value = 150.0
        self.strategy.data_retriever = mock_data_retriever
        
        price = self.strategy._get_current_underlying_price(current_date)
        
        assert price == 150.0
        mock_data_retriever.get_live_price.assert_called_once()
    
    def test_get_current_volumes_for_position_success(self):
        """Test get_current_volumes_for_position with successful API calls"""
        # Create mock position with options
        option1 = Option(
            ticker='SPY',
            symbol='SPY240119C00450000',
            strike=450.0,
            expiration='2024-01-19',
            option_type=OptionType.CALL,
            last_price=5.0,
            volume=100
        )
        option2 = Option(
            ticker='SPY',
            symbol='SPY240119C00455000',
            strike=455.0,
            expiration='2024-01-19',
            option_type=OptionType.CALL,
            last_price=3.0,
            volume=150
        )
        
        position = Position(
            symbol='SPY',
            expiration_date=datetime(2024, 1, 19),
            strategy_type=StrategyType.CALL_CREDIT_SPREAD,
            strike_price=450.0,
            entry_date=datetime(2024, 1, 15),
            entry_price=2.0,
            spread_options=[option1, option2]
        )
        
        # Mock options handler to return fresh option data
        fresh_option1 = Option(
            ticker='SPY',
            symbol='SPY240119C00450000',
            strike=450.0,
            expiration='2024-01-19',
            option_type=OptionType.CALL,
            last_price=5.0,
            volume=200  # Updated volume
        )
        fresh_option2 = Option(
            ticker='SPY',
            symbol='SPY240119C00455000',
            strike=455.0,
            expiration='2024-01-19',
            option_type=OptionType.CALL,
            last_price=3.0,
            volume=250  # Updated volume
        )
        
        self.options_handler.get_specific_option_contract.side_effect = [fresh_option1, fresh_option2]
        
        test_date = datetime(2024, 1, 16)
        volumes = self.strategy.get_current_volumes_for_position(position, test_date)
        
        assert volumes == [200, 250]
        assert self.options_handler.get_specific_option_contract.call_count == 2
    
    def test_get_current_volumes_for_position_api_failure(self):
        """Test get_current_volumes_for_position with API failures"""
        # Create mock position with options
        option1 = Option(
            ticker='SPY',
            symbol='SPY240119C00450000',
            strike=450.0,
            expiration='2024-01-19',
            option_type=OptionType.CALL,
            last_price=5.0,
            volume=100
        )
        
        position = Position(
            symbol='SPY',
            expiration_date=datetime(2024, 1, 19),
            strategy_type=StrategyType.CALL_CREDIT_SPREAD,
            strike_price=450.0,
            entry_date=datetime(2024, 1, 15),
            entry_price=2.0,
            spread_options=[option1]
        )
        
        # Mock options handler to raise exception
        self.options_handler.get_specific_option_contract.side_effect = Exception("API Error")
        
        test_date = datetime(2024, 1, 16)
        volumes = self.strategy.get_current_volumes_for_position(position, test_date)
        
        assert volumes == [None]
    
    def test_get_current_volumes_for_position_no_volume_data(self):
        """Test get_current_volumes_for_position when fresh option has no volume"""
        # Create mock position with options
        option1 = Option(
            ticker='SPY',
            symbol='SPY240119C00450000',
            strike=450.0,
            expiration='2024-01-19',
            option_type=OptionType.CALL,
            last_price=5.0,
            volume=100
        )
        
        position = Position(
            symbol='SPY',
            expiration_date=datetime(2024, 1, 19),
            strategy_type=StrategyType.CALL_CREDIT_SPREAD,
            strike_price=450.0,
            entry_date=datetime(2024, 1, 15),
            entry_price=2.0,
            spread_options=[option1]
        )
        
        # Mock options handler to return option without volume
        fresh_option1 = Option(
            ticker='SPY',
            symbol='SPY240119C00450000',
            strike=450.0,
            expiration='2024-01-19',
            option_type=OptionType.CALL,
            last_price=5.0,
            volume=None  # No volume data
        )
        
        self.options_handler.get_specific_option_contract.return_value = fresh_option1
        
        test_date = datetime(2024, 1, 16)
        volumes = self.strategy.get_current_volumes_for_position(position, test_date)
        
        assert volumes == [None]


class TestDataRetrieverLivePrice:
    """Test cases for DataRetriever get_live_price method"""
    
    def setup_method(self):
        """Set up test fixtures"""
        from src.common.data_retriever import DataRetriever
        self.data_retriever = DataRetriever(
            symbol='SPY',
            use_free_tier=True,
            quiet_mode=True
        )
    
    @patch('src.common.data_retriever.yf.Ticker')
    def test_get_live_price_yfinance_fallback(self, mock_ticker):
        """Test get_live_price with yfinance fallback"""
        # Mock yfinance ticker
        mock_ticker_instance = Mock()
        mock_ticker_instance.info = {
            'currentPrice': 450.0,
            'regularMarketPrice': 450.0,
            'previousClose': 445.0
        }
        mock_ticker.return_value = mock_ticker_instance
        
        price = self.data_retriever.get_live_price('SPY')
        
        assert price == 450.0
        mock_ticker.assert_called_once_with('SPY')
    
    @patch('src.common.data_retriever.yf.Ticker')
    def test_get_live_price_yfinance_no_current_price(self, mock_ticker):
        """Test get_live_price with yfinance when currentPrice is None"""
        # Mock yfinance ticker with no currentPrice
        mock_ticker_instance = Mock()
        mock_ticker_instance.info = {
            'currentPrice': None,
            'regularMarketPrice': 450.0,
            'previousClose': 445.0
        }
        mock_ticker.return_value = mock_ticker_instance
        
        price = self.data_retriever.get_live_price('SPY')
        
        assert price == 450.0  # Should fall back to regularMarketPrice
    
    @patch('src.common.data_retriever.yf.Ticker')
    def test_get_live_price_yfinance_fallback_to_previous_close(self, mock_ticker):
        """Test get_live_price with yfinance falling back to previousClose"""
        # Mock yfinance ticker with no currentPrice or regularMarketPrice
        mock_ticker_instance = Mock()
        mock_ticker_instance.info = {
            'currentPrice': None,
            'regularMarketPrice': None,
            'previousClose': 445.0
        }
        mock_ticker.return_value = mock_ticker_instance
        
        price = self.data_retriever.get_live_price('SPY')
        
        assert price == 445.0  # Should fall back to previousClose
    
    @patch('src.common.data_retriever.yf.Ticker')
    def test_get_live_price_yfinance_no_data(self, mock_ticker):
        """Test get_live_price with yfinance when no price data is available"""
        # Mock yfinance ticker with no price data
        mock_ticker_instance = Mock()
        mock_ticker_instance.info = {
            'currentPrice': None,
            'regularMarketPrice': None,
            'previousClose': None
        }
        mock_ticker.return_value = mock_ticker_instance
        
        price = self.data_retriever.get_live_price('SPY')
        
        assert price is None
    
    @patch('src.common.data_retriever.yf.Ticker')
    def test_get_live_price_yfinance_exception(self, mock_ticker):
        """Test get_live_price with yfinance exception"""
        # Mock yfinance ticker to raise exception
        mock_ticker.side_effect = Exception("Network error")
        
        price = self.data_retriever.get_live_price('SPY')
        
        assert price is None
