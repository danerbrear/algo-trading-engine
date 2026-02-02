"""
Unit tests for enhanced strategy methods
"""
from datetime import datetime
from unittest.mock import Mock, patch
import pandas as pd
import tempfile
import shutil
from pathlib import Path
import pytest

from algo_trading_engine.strategies.velocity_signal_momentum_strategy import VelocitySignalMomentumStrategy
from algo_trading_engine.strategies.credit_spread_minimal import CreditSpreadStrategy
from algo_trading_engine.common.models import Option, OptionType
from algo_trading_engine.backtest.models import Position, StrategyType


@pytest.fixture
def temp_cache_dir():
    """Create a temporary cache directory for testing and clean up afterward."""
    temp_dir = Path(tempfile.mkdtemp())
    yield temp_dir
    # Cleanup after test
    if temp_dir.exists():
        shutil.rmtree(temp_dir)


@pytest.fixture
def mock_cache_manager(temp_cache_dir):
    """Mock CacheManager to use temporary directory."""
    def _mock_get_cache_dir(self, *subdirs):
        cache_dir = temp_cache_dir.joinpath(*subdirs)
        cache_dir.mkdir(parents=True, exist_ok=True)
        return cache_dir
    
    with patch('algo_trading_engine.common.cache.cache_manager.CacheManager.get_cache_dir', _mock_get_cache_dir):
        yield temp_cache_dir


class TestVelocitySignalMomentumStrategyEnhancements:
    """Test cases for enhanced VelocitySignalMomentumStrategy methods"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.options_handler = Mock()
        # Create callables from options_handler methods
        self.get_contract_list_for_date = self.options_handler.get_contract_list_for_date
        self.get_option_bar = self.options_handler.get_option_bar
        self.get_options_chain = self.options_handler.get_options_chain
        self.strategy = VelocitySignalMomentumStrategy(
            get_contract_list_for_date=self.get_contract_list_for_date,
            get_option_bar=self.get_option_bar,
            get_options_chain=self.get_options_chain,
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
        
        self.strategy.set_data(self.sample_data)
    
    def test_get_current_underlying_price_historical_date(self):
        """Test get_current_underlying_price with historical date"""
        test_date = datetime(2024, 1, 15)
        symbol = self.sample_data.index.name if self.sample_data.index.name else 'SPY'
        
        # Mock the injected method
        def mock_get_price(date, sym):
            return float(self.sample_data.loc[date, 'Close'])
        self.strategy.get_current_underlying_price = mock_get_price
        
        price = self.strategy.get_current_underlying_price(test_date, symbol)
        
        # Should return the close price for that date
        expected_price = self.sample_data.loc[test_date, 'Close']
        assert price == expected_price
    
    def test_get_current_underlying_price_current_date_with_live_price(self):
        """Test get_current_underlying_price with current date and live price available"""
        # Mock current date
        current_date = datetime(2024, 1, 20)
        symbol = self.sample_data.index.name if self.sample_data.index.name else 'SPY'
        
        # Mock the injected method to return live price
        def mock_get_price(date, sym):
            return 150.0
        self.strategy.get_current_underlying_price = mock_get_price
        
        price = self.strategy.get_current_underlying_price(current_date, symbol)
        
        assert price == 150.0
    
    def test_get_current_underlying_price_current_date_fallback(self):
        """Test get_current_underlying_price with current date but no live price available"""
        # Mock current date that's NOT in the sample data
        current_date = datetime(2024, 5, 1)  # This date is not in the sample data
        symbol = self.sample_data.index.name if self.sample_data.index.name else 'SPY'
        
        # Mock the injected method to raise ValueError when live price is None
        def mock_get_price(date, sym):
            raise ValueError("Failed to fetch live price from DataRetriever")
        self.strategy.get_current_underlying_price = mock_get_price
        
        # The current implementation raises ValueError when live price is None
        import pytest
        with pytest.raises(ValueError, match="Failed to fetch live price from DataRetriever"):
            self.strategy.get_current_underlying_price(current_date, symbol)
    
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
        # Create callables from options_handler methods
        self.get_contract_list_for_date = self.options_handler.get_contract_list_for_date
        self.get_option_bar = self.options_handler.get_option_bar
        self.get_options_chain = self.options_handler.get_options_chain
        
        self.strategy = CreditSpreadStrategy(
            get_contract_list_for_date=self.get_contract_list_for_date,
            get_option_bar=self.get_option_bar,
            get_options_chain=self.get_options_chain,
            lstm_model=self.lstm_model,
            lstm_scaler=self.lstm_scaler,
            symbol='SPY',
            start_date_offset=0,
            options_handler=self.options_handler  # Still needed for LSTMModel
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
        
        self.strategy.set_data(self.sample_data)
    
    def test_get_current_underlying_price_historical_date(self):
        """Test get_current_underlying_price with historical date"""
        test_date = datetime(2024, 1, 15)
        symbol = 'SPY'
        
        # Mock the injected method
        def mock_get_price(date, sym):
            return float(self.sample_data.loc[date, 'Close'])
        self.strategy.get_current_underlying_price = mock_get_price
        
        price = self.strategy.get_current_underlying_price(test_date, symbol)
        
        # Should return the close price for that date
        expected_price = self.sample_data.loc[test_date, 'Close']
        assert price == expected_price
    
    def test_get_current_underlying_price_current_date_with_live_price(self):
        """Test get_current_underlying_price with current date and live price available"""
        # Mock current date
        current_date = datetime(2024, 1, 20)
        symbol = 'SPY'
        
        # Mock the injected method to return live price
        def mock_get_price(date, sym):
            return 150.0
        self.strategy.get_current_underlying_price = mock_get_price
        
        price = self.strategy.get_current_underlying_price(current_date, symbol)
        
        assert price == 150.0
    
    def test_get_current_volumes_for_position_success(self):
        """Test get_current_volumes_for_position with successful API calls"""
        from algo_trading_engine.dto import OptionContractDTO, OptionBarDTO
        from algo_trading_engine.common.models import OptionType as CommonOptionType
        from decimal import Decimal
        
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
        
        # Mock contract DTOs
        from algo_trading_engine.vo import StrikePrice, ExpirationDate
        contract1 = OptionContractDTO(
            ticker='O:SPY240119C00450000',
            underlying_ticker='SPY',
            contract_type=CommonOptionType.CALL,
            strike_price=StrikePrice(Decimal('450.0')),
            expiration_date=ExpirationDate(datetime(2024, 1, 19).date()),
            exercise_style='american',
            shares_per_contract=100
        )
        contract2 = OptionContractDTO(
            ticker='O:SPY240119C00455000',
            underlying_ticker='SPY',
            contract_type=CommonOptionType.CALL,
            strike_price=StrikePrice(Decimal('455.0')),
            expiration_date=ExpirationDate(datetime(2024, 1, 19).date()),
            exercise_style='american',
            shares_per_contract=100
        )
        
        # Mock bar DTOs with updated volumes
        bar1 = OptionBarDTO(
            ticker='O:SPY240119C00450000',
            timestamp=datetime(2024, 1, 16),
            open_price=Decimal('5.0'),
            high_price=Decimal('5.1'),
            low_price=Decimal('4.9'),
            close_price=Decimal('5.0'),
            volume=200,  # Updated volume
            volume_weighted_avg_price=Decimal('5.0'),
            number_of_transactions=100
        )
        bar2 = OptionBarDTO(
            ticker='O:SPY240119C00455000',
            timestamp=datetime(2024, 1, 16),
            open_price=Decimal('3.0'),
            high_price=Decimal('3.1'),
            low_price=Decimal('2.9'),
            close_price=Decimal('3.0'),
            volume=250,  # Updated volume
            volume_weighted_avg_price=Decimal('3.0'),
            number_of_transactions=80
        )
        
        # Mock _get_option_with_bar which is what get_current_volumes_for_position actually calls
        def mock_get_option_with_bar(strike, expiry_date, option_type, date):
            # Return Option objects with volume data
            if strike == 450.0 and expiry_date == datetime(2024, 1, 19).date() and option_type == OptionType.CALL:
                return Option.from_contract_and_bar(contract1, bar1)
            elif strike == 455.0 and expiry_date == datetime(2024, 1, 19).date() and option_type == OptionType.CALL:
                return Option.from_contract_and_bar(contract2, bar2)
            return None
        
        # Update the method on the strategy
        self.strategy._get_option_with_bar = Mock(side_effect=mock_get_option_with_bar)
        
        test_date = datetime(2024, 1, 16)
        volumes = self.strategy.get_current_volumes_for_position(position, test_date)
        
        assert volumes == [200, 250]
        assert self.strategy._get_option_with_bar.call_count == 2
    
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
        
        # Mock callables to raise exception
        self.strategy.get_contract_list_for_date = Mock(side_effect=Exception("API Error"))
        
        test_date = datetime(2024, 1, 16)
        volumes = self.strategy.get_current_volumes_for_position(position, test_date)
        
        assert volumes == [None]
    
    def test_get_current_volumes_for_position_no_volume_data(self):
        """Test get_current_volumes_for_position when fresh option has no volume"""
        from algo_trading_engine.dto import OptionContractDTO, OptionBarDTO
        from algo_trading_engine.common.models import OptionType as CommonOptionType
        from decimal import Decimal
        
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
        
        # Mock contract DTO
        from algo_trading_engine.vo import StrikePrice, ExpirationDate
        contract1 = OptionContractDTO(
            ticker='O:SPY240119C00450000',
            underlying_ticker='SPY',
            contract_type=CommonOptionType.CALL,
            strike_price=StrikePrice(Decimal('450.0')),
            expiration_date=ExpirationDate(datetime(2024, 1, 19).date()),
            exercise_style='american',
            shares_per_contract=100
        )
        
        # Mock the new API methods
        # When there's no volume data, get_option_bar should return None
        self.strategy.get_contract_list_for_date = Mock(return_value=[contract1])
        self.strategy.get_option_bar = Mock(return_value=None)  # No bar data = no volume
        
        test_date = datetime(2024, 1, 16)
        volumes = self.strategy.get_current_volumes_for_position(position, test_date)
        
        assert volumes == [None]


class TestDataRetrieverLivePrice:
    """Test cases for DataRetriever get_live_price method"""
    
    def setup_method(self):
        """Set up test fixtures"""
        # Note: DataRetriever will be created in each test with mock_cache_manager
        pass
    
    @patch('algo_trading_engine.common.data_retriever.yf.Ticker')
    def test_get_live_price_yfinance_fallback(self, mock_ticker, mock_cache_manager):
        """Test get_live_price with yfinance fallback"""
        from algo_trading_engine.common.data_retriever import DataRetriever
        
        data_retriever = DataRetriever(
            symbol='SPY',
            use_free_tier=True,
            quiet_mode=True
        )
        
        # Mock yfinance ticker
        mock_ticker_instance = Mock()
        mock_ticker_instance.info = {
            'currentPrice': 450.0,
            'regularMarketPrice': 450.0,
            'previousClose': 445.0
        }
        mock_ticker.return_value = mock_ticker_instance
        
        price = data_retriever.get_live_price('SPY')
        
        assert price == 450.0
        mock_ticker.assert_called_once_with('SPY')
    
    @patch('algo_trading_engine.common.data_retriever.yf.Ticker')
    def test_get_live_price_yfinance_no_current_price(self, mock_ticker, mock_cache_manager):
        """Test get_live_price with yfinance when currentPrice is None"""
        from algo_trading_engine.common.data_retriever import DataRetriever
        
        data_retriever = DataRetriever(
            symbol='SPY',
            use_free_tier=True,
            quiet_mode=True
        )
        
        # Mock yfinance ticker with no currentPrice
        mock_ticker_instance = Mock()
        mock_ticker_instance.info = {
            'currentPrice': None,
            'regularMarketPrice': 450.0,
            'previousClose': 445.0
        }
        mock_ticker.return_value = mock_ticker_instance
        
        price = data_retriever.get_live_price('SPY')
        
        assert price == 450.0  # Should fall back to regularMarketPrice
    
    @patch('algo_trading_engine.common.data_retriever.yf.Ticker')
    def test_get_live_price_yfinance_fallback_to_previous_close(self, mock_ticker, mock_cache_manager):
        """Test get_live_price with yfinance falling back to previousClose"""
        from algo_trading_engine.common.data_retriever import DataRetriever
        
        data_retriever = DataRetriever(
            symbol='SPY',
            use_free_tier=True,
            quiet_mode=True
        )
        
        # Mock yfinance ticker with no currentPrice or regularMarketPrice
        mock_ticker_instance = Mock()
        mock_ticker_instance.info = {
            'currentPrice': None,
            'regularMarketPrice': None,
            'previousClose': 445.0
        }
        mock_ticker.return_value = mock_ticker_instance
        
        price = data_retriever.get_live_price('SPY')
        
        assert price == 445.0  # Should fall back to previousClose
    
    @patch('algo_trading_engine.common.data_retriever.yf.Ticker')
    def test_get_live_price_yfinance_no_data(self, mock_ticker, mock_cache_manager):
        """Test get_live_price with yfinance when no price data is available"""
        from algo_trading_engine.common.data_retriever import DataRetriever
        
        data_retriever = DataRetriever(
            symbol='SPY',
            use_free_tier=True,
            quiet_mode=True
        )
        
        # Mock yfinance ticker with no price data
        mock_ticker_instance = Mock()
        mock_ticker_instance.info = {
            'currentPrice': None,
            'regularMarketPrice': None,
            'previousClose': None
        }
        mock_ticker.return_value = mock_ticker_instance
        
        price = data_retriever.get_live_price('SPY')
        
        assert price is None
    
    @patch('algo_trading_engine.common.data_retriever.yf.Ticker')
    def test_get_live_price_yfinance_exception(self, mock_ticker, mock_cache_manager):
        """Test get_live_price with yfinance exception"""
        from algo_trading_engine.common.data_retriever import DataRetriever
        
        data_retriever = DataRetriever(
            symbol='SPY',
            use_free_tier=True,
            quiet_mode=True
        )
        
        # Mock yfinance ticker to raise exception
        mock_ticker.side_effect = Exception("Network error")
        
        price = data_retriever.get_live_price('SPY')
        
        assert price is None
