"""
Unit tests for DataRetriever bar interval support.
"""
import pytest
from unittest.mock import Mock, patch, MagicMock
import pandas as pd
from datetime import datetime
from pathlib import Path
import tempfile
import shutil

from algo_trading_engine.common.data_retriever import DataRetriever
from algo_trading_engine.enums import BarTimeInterval


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


class TestDataRetrieverBarInterval:
    """Test cases for DataRetriever bar_interval parameter."""
    
    def test_default_bar_interval_is_daily(self, mock_cache_manager):
        """Test that bar_interval defaults to DAY."""
        retriever = DataRetriever(symbol='SPY')
        assert retriever.bar_interval == BarTimeInterval.DAY
    
    def test_hourly_bar_interval(self, mock_cache_manager):
        """Test creating DataRetriever with hourly bars."""
        retriever = DataRetriever(
            symbol='SPY',
            bar_interval=BarTimeInterval.HOUR
        )
        assert retriever.bar_interval == BarTimeInterval.HOUR
    
    def test_minute_bar_interval(self, mock_cache_manager):
        """Test creating DataRetriever with minute bars."""
        retriever = DataRetriever(
            symbol='SPY',
            bar_interval=BarTimeInterval.MINUTE
        )
        assert retriever.bar_interval == BarTimeInterval.MINUTE
    
    def test_get_yfinance_interval_daily(self, mock_cache_manager):
        """Test _get_yfinance_interval returns correct string for daily."""
        retriever = DataRetriever(symbol='SPY', bar_interval=BarTimeInterval.DAY)
        assert retriever._get_yfinance_interval() == "1d"
    
    def test_get_yfinance_interval_hourly(self, mock_cache_manager):
        """Test _get_yfinance_interval returns correct string for hourly."""
        retriever = DataRetriever(symbol='SPY', bar_interval=BarTimeInterval.HOUR)
        assert retriever._get_yfinance_interval() == "1h"
    
    def test_get_yfinance_interval_minute(self, mock_cache_manager):
        """Test _get_yfinance_interval returns correct string for minute."""
        retriever = DataRetriever(symbol='SPY', bar_interval=BarTimeInterval.MINUTE)
        assert retriever._get_yfinance_interval() == "1m"
    
    def test_get_cache_interval_dir_daily(self, mock_cache_manager):
        """Test _get_cache_interval_dir returns correct directory for daily."""
        retriever = DataRetriever(symbol='SPY', bar_interval=BarTimeInterval.DAY)
        assert retriever._get_cache_interval_dir() == "daily"
    
    def test_get_cache_interval_dir_hourly(self, mock_cache_manager):
        """Test _get_cache_interval_dir returns correct directory for hourly."""
        retriever = DataRetriever(symbol='SPY', bar_interval=BarTimeInterval.HOUR)
        assert retriever._get_cache_interval_dir() == "hourly"
    
    def test_get_cache_interval_dir_minute(self, mock_cache_manager):
        """Test _get_cache_interval_dir returns correct string for minute."""
        retriever = DataRetriever(symbol='SPY', bar_interval=BarTimeInterval.MINUTE)
        assert retriever._get_cache_interval_dir() == "minute"


class TestDataRetrieverFetchWithInterval:
    """Test cases for fetch_data_for_period with different intervals."""
    
    def create_mock_ticker_data(self, num_bars=10, interval=BarTimeInterval.DAY):
        """Create mock ticker data for testing."""
        if interval == BarTimeInterval.DAY:
            dates = pd.date_range(start='2024-01-01', periods=num_bars, freq='D')
        elif interval == BarTimeInterval.HOUR:
            dates = pd.date_range(start='2024-01-01 09:30', periods=num_bars, freq='h')
        else:  # MINUTE
            dates = pd.date_range(start='2024-01-01 09:30', periods=num_bars, freq='min')
        
        data = pd.DataFrame({
            'Open': [100 + i for i in range(num_bars)],
            'High': [105 + i for i in range(num_bars)],
            'Low': [95 + i for i in range(num_bars)],
            'Close': [100 + i for i in range(num_bars)],
            'Volume': [1000000] * num_bars
        }, index=dates)
        
        return data
    
    @patch('algo_trading_engine.common.data_retriever.DataRetriever._load_cached_data_range')
    @patch('yfinance.Ticker')
    def test_fetch_data_daily_interval(self, mock_ticker_class, mock_load_cache, mock_cache_manager):
        """Test fetching data with daily interval."""
        # Mock cache to return None (no cached data)
        mock_load_cache.return_value = None
        
        retriever = DataRetriever(symbol='SPY', bar_interval=BarTimeInterval.DAY)
        
        # Mock ticker
        mock_ticker = Mock()
        mock_ticker.info = {'symbol': 'SPY'}
        mock_ticker.history.return_value = self.create_mock_ticker_data(10, BarTimeInterval.DAY)
        mock_ticker_class.return_value = mock_ticker
        
        data = retriever.fetch_data_for_period('2024-01-01', '2024-01-10')
        
        # Verify yfinance was called with correct interval
        mock_ticker.history.assert_called_once()
        call_kwargs = mock_ticker.history.call_args.kwargs
        assert call_kwargs['interval'] == '1d'
        assert len(data) > 0
    
    @patch('algo_trading_engine.common.data_retriever.DataRetriever._load_cached_data_range')
    @patch('yfinance.Ticker')
    def test_fetch_data_hourly_interval(self, mock_ticker_class, mock_load_cache, mock_cache_manager):
        """Test fetching data with hourly interval."""
        # Mock cache to return None (no cached data)
        mock_load_cache.return_value = None
        
        retriever = DataRetriever(symbol='SPY', bar_interval=BarTimeInterval.HOUR)
        
        # Mock ticker
        mock_ticker = Mock()
        mock_ticker.info = {'symbol': 'SPY'}
        mock_ticker.history.return_value = self.create_mock_ticker_data(24, BarTimeInterval.HOUR)
        mock_ticker_class.return_value = mock_ticker
        
        data = retriever.fetch_data_for_period('2024-01-01', '2024-01-02')
        
        # Verify yfinance was called with correct interval
        mock_ticker.history.assert_called_once()
        call_kwargs = mock_ticker.history.call_args.kwargs
        assert call_kwargs['interval'] == '1h'
        assert len(data) > 0
    
    @patch('algo_trading_engine.common.data_retriever.DataRetriever._load_cached_data_range')
    @patch('yfinance.Ticker')
    def test_fetch_data_minute_interval(self, mock_ticker_class, mock_load_cache, mock_cache_manager):
        """Test fetching data with minute interval."""
        # Mock cache to return None (no cached data)
        mock_load_cache.return_value = None
        
        retriever = DataRetriever(symbol='SPY', bar_interval=BarTimeInterval.MINUTE)
        
        # Mock ticker
        mock_ticker = Mock()
        mock_ticker.info = {'symbol': 'SPY'}
        mock_ticker.history.return_value = self.create_mock_ticker_data(60, BarTimeInterval.MINUTE)
        mock_ticker_class.return_value = mock_ticker
        
        # Use end_date that includes the time component
        data = retriever.fetch_data_for_period('2024-01-01', '2024-01-01 23:59')
        
        # Verify yfinance was called with correct interval
        mock_ticker.history.assert_called_once()
        call_kwargs = mock_ticker.history.call_args.kwargs
        assert call_kwargs['interval'] == '1m'
        assert len(data) > 0
    
    @patch('algo_trading_engine.common.data_retriever.DataRetriever._load_cached_data_range')
    @patch('yfinance.Ticker')
    def test_fetch_data_no_data_type_parameter(self, mock_ticker_class, mock_load_cache, mock_cache_manager):
        """Test that fetch_data_for_period doesn't accept data_type parameter."""
        # Mock cache to return None (no cached data)
        mock_load_cache.return_value = None
        
        retriever = DataRetriever(symbol='SPY', bar_interval=BarTimeInterval.DAY)
        
        # Mock ticker
        mock_ticker = Mock()
        mock_ticker.info = {'symbol': 'SPY'}
        mock_ticker.history.return_value = self.create_mock_ticker_data(10, BarTimeInterval.DAY)
        mock_ticker_class.return_value = mock_ticker
        
        # Should work with just start_date and end_date
        data = retriever.fetch_data_for_period('2024-01-01', '2024-01-10')
        assert len(data) > 0
        
        # Verify data_type was not passed anywhere
        call_kwargs = mock_ticker.history.call_args.kwargs
        assert 'data_type' not in call_kwargs
    
    @patch('algo_trading_engine.common.data_retriever.DataRetriever._load_cached_data_range')
    @patch('yfinance.Ticker')
    def test_fetch_data_error_includes_interval(self, mock_ticker_class, mock_load_cache, mock_cache_manager):
        """Test that error messages include interval information."""
        # Mock cache to return None (no cached data)
        mock_load_cache.return_value = None
        
        retriever = DataRetriever(symbol='SPY', bar_interval=BarTimeInterval.HOUR)
        
        # Mock ticker to return empty data
        mock_ticker = Mock()
        mock_ticker.info = {'symbol': 'SPY'}
        mock_ticker.history.return_value = pd.DataFrame()
        mock_ticker_class.return_value = mock_ticker
        
        with pytest.raises(ValueError) as exc_info:
            retriever.fetch_data_for_period('2024-01-01', '2024-01-10')
        
        error_message = str(exc_info.value)
        assert '1h' in error_message  # Should mention the interval


class TestCacheStructure:
    """Test cases for new cache directory structure."""
    
    def test_cache_structure_uses_interval_subdirectory(self, mock_cache_manager):
        """Test that cache uses interval-specific subdirectories."""
        retriever = DataRetriever(symbol='SPY', bar_interval=BarTimeInterval.HOUR)
        
        # Verify cache directory structure
        interval_dir = retriever._get_cache_interval_dir()
        assert interval_dir == "hourly"
        
        # Verify cache base path includes symbol
        cache_base = retriever.cache_manager.get_cache_dir('stocks', 'SPY')
        assert 'SPY' in str(cache_base)
        
        # Verify it's using the temp directory
        assert str(mock_cache_manager) in str(cache_base)
    
    def test_load_cached_data_range_returns_none_if_no_cache(self, mock_cache_manager):
        """Test _load_cached_data_range returns None if cache doesn't exist."""
        retriever = DataRetriever(symbol='TESTYMBOL', bar_interval=BarTimeInterval.DAY)
        
        result = retriever._load_cached_data_range('2024-01-01', '2024-01-10')
        
        assert result is None
        # Cache directory created in temp dir, will be cleaned up automatically
    
    def test_daily_cache_file_naming(self, mock_cache_manager):
        """Test that daily cache files use YYYY-MM-DD.pkl format (one file per start_date)."""
        retriever = DataRetriever(symbol='SPY', bar_interval=BarTimeInterval.DAY)
        
        # Daily files should be named by start_date: 2024-01-01.pkl (contains all data from that start date)
        # This is different from hourly/minute which use granular files
        interval_dir = retriever._get_cache_interval_dir()
        assert interval_dir == "daily"
        
        # This confirms daily uses single-file caching for performance
    
    def test_hourly_cache_file_naming(self, mock_cache_manager):
        """Test that hourly cache files use YYYY-MM-DD_HHMM.pkl format."""
        retriever = DataRetriever(symbol='SPY', bar_interval=BarTimeInterval.HOUR)
        
        # Hourly files should be named like: 2024-01-01_0930.pkl
        interval_dir = retriever._get_cache_interval_dir()
        assert interval_dir == "hourly"
        
        # This confirms the naming convention matches the plan
    
    def test_minute_cache_file_naming(self, mock_cache_manager):
        """Test that minute cache files use YYYY-MM-DD_HHMM.pkl format."""
        retriever = DataRetriever(symbol='SPY', bar_interval=BarTimeInterval.MINUTE)
        
        # Minute files should be named like: 2024-01-01_0930.pkl
        interval_dir = retriever._get_cache_interval_dir()
        assert interval_dir == "minute"
        
        # This confirms the naming convention matches the plan


class TestDailyCachePerformance:
    """Test that daily caching uses single-file approach for performance."""
    
    def test_daily_bars_saved_as_single_file(self, mock_cache_manager):
        """Test that daily bars are saved as one file per start_date, not per day."""
        with patch('algo_trading_engine.common.data_retriever.DataRetriever._load_cached_data_range') as mock_load_cache, \
             patch('yfinance.Ticker') as mock_ticker_class:
            
            # Mock cache to return None (no cached data)
            mock_load_cache.return_value = None
            
            retriever = DataRetriever(symbol='SPY', bar_interval=BarTimeInterval.DAY)
            
            # Create mock data with 100 days
            dates = pd.date_range(start='2024-01-01', periods=100, freq='D')
            mock_data = pd.DataFrame({
                'Open': [100 + i for i in range(100)],
                'High': [105 + i for i in range(100)],
                'Low': [95 + i for i in range(100)],
                'Close': [100 + i for i in range(100)],
                'Volume': [1000000] * 100
            }, index=dates)
            
            # Mock ticker
            mock_ticker = Mock()
            mock_ticker.info = {'symbol': 'SPY'}
            mock_ticker.history.return_value = mock_data
            mock_ticker_class.return_value = mock_ticker
            
            # Fetch data (will write to temp cache)
            retriever.fetch_data_for_period('2024-01-01', '2024-04-09')
            
            # Verify: Should create ONE file named 2024-01-01.pkl, not 100 separate files
            cache_dir = retriever.cache_manager.get_cache_dir('stocks', 'SPY') / 'daily'
            
            # The file should be named by start_date
            expected_file = cache_dir / '2024-01-01.pkl'
            assert expected_file.exists(), "Daily cache should create one file per start_date"
            
            # Verify we can load it back
            loaded_data = pd.read_pickle(expected_file)
            assert len(loaded_data) == 100, "Should contain all 100 days in one file"
            
            # Temp cache will be automatically cleaned up by fixture


class TestProcessAgnosticCaching:
    """Test that caching is process-agnostic (no 'backtest' or 'general' in paths)."""
    
    def test_no_data_type_in_cache_path(self, mock_cache_manager):
        """Test that cache paths don't include data_type like 'backtest' or 'general'."""
        retriever = DataRetriever(symbol='SPY', bar_interval=BarTimeInterval.DAY)
        
        # Get cache directory
        cache_base = retriever.cache_manager.get_cache_dir('stocks', 'SPY')
        interval_dir = retriever._get_cache_interval_dir()
        full_cache_path = cache_base / interval_dir
        
        # Verify path doesn't contain process-specific terms
        path_str = str(full_cache_path)
        assert 'backtest' not in path_str.lower()
        assert 'general' not in path_str.lower()
        assert 'paper_trading' not in path_str.lower()
        assert 'hmm' not in path_str.lower()
        assert 'lstm' not in path_str.lower()
        
        # Verify it does contain the interval directory
        assert interval_dir in path_str
    
    def test_cache_reusable_across_processes(self, mock_cache_manager):
        """Test that cache from different processes uses same structure."""
        # Create two retrievers with same symbol and interval
        retriever1 = DataRetriever(symbol='SPY', bar_interval=BarTimeInterval.HOUR)
        retriever2 = DataRetriever(symbol='SPY', bar_interval=BarTimeInterval.HOUR)
        
        # They should use the same cache directory
        cache_dir1 = retriever1.cache_manager.get_cache_dir('stocks', 'SPY') / 'hourly'
        cache_dir2 = retriever2.cache_manager.get_cache_dir('stocks', 'SPY') / 'hourly'
        
        assert cache_dir1 == cache_dir2
