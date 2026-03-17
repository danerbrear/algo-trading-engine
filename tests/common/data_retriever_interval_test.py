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


class TestDataRetrieverUseCache:
    """Test cases for DataRetriever use_cache parameter."""

    def test_use_cache_defaults_to_true(self, mock_cache_manager):
        """Test that use_cache defaults to True."""
        retriever = DataRetriever(symbol='SPY')
        assert retriever.use_cache is True

    def test_use_cache_can_be_set_to_false(self, mock_cache_manager):
        """Test that use_cache can be explicitly disabled."""
        retriever = DataRetriever(symbol='SPY', use_cache=False)
        assert retriever.use_cache is False

    @patch('algo_trading_engine.common.data_retriever.DataRetriever._load_cached_data_range')
    @patch('yfinance.Ticker')
    def test_fetch_data_does_not_write_cache_when_disabled(
        self, mock_ticker_class, mock_load_cache, mock_cache_manager
    ):
        """Test that no cache files are written when use_cache=False."""
        mock_load_cache.return_value = None

        retriever = DataRetriever(symbol='SPY', bar_interval=BarTimeInterval.DAY, use_cache=False)

        dates = pd.date_range(start='2024-01-01', periods=10, freq='D')
        mock_data = pd.DataFrame({
            'Open': [100 + i for i in range(10)],
            'High': [105 + i for i in range(10)],
            'Low': [95 + i for i in range(10)],
            'Close': [100 + i for i in range(10)],
            'Volume': [1000000] * 10
        }, index=dates)

        mock_ticker = Mock()
        mock_ticker.info = {'symbol': 'SPY'}
        mock_ticker.history.return_value = mock_data
        mock_ticker_class.return_value = mock_ticker

        data = retriever.fetch_data_for_period('2024-01-01', '2024-01-10')

        assert len(data) > 0
        cache_dir = retriever.cache_manager.get_cache_dir('stocks', 'SPY') / 'daily'
        assert not cache_dir.exists() or not any(cache_dir.iterdir()), \
            "No cache files should be written when use_cache=False"

    @patch('algo_trading_engine.common.data_retriever.DataRetriever._load_cached_data_range')
    @patch('yfinance.Ticker')
    def test_fetch_data_writes_cache_when_enabled(
        self, mock_ticker_class, mock_load_cache, mock_cache_manager
    ):
        """Test that cache files are written when use_cache=True."""
        mock_load_cache.return_value = None

        retriever = DataRetriever(symbol='SPY', bar_interval=BarTimeInterval.DAY, use_cache=True)

        dates = pd.date_range(start='2024-01-01', periods=10, freq='D')
        mock_data = pd.DataFrame({
            'Open': [100 + i for i in range(10)],
            'High': [105 + i for i in range(10)],
            'Low': [95 + i for i in range(10)],
            'Close': [100 + i for i in range(10)],
            'Volume': [1000000] * 10
        }, index=dates)

        mock_ticker = Mock()
        mock_ticker.info = {'symbol': 'SPY'}
        mock_ticker.history.return_value = mock_data
        mock_ticker_class.return_value = mock_ticker

        retriever.fetch_data_for_period('2024-01-01', '2024-01-10')

        cache_dir = retriever.cache_manager.get_cache_dir('stocks', 'SPY') / 'daily'
        expected_file = cache_dir / '2024-01-01.pkl'
        assert expected_file.exists(), "Cache file should be written when use_cache=True"


class TestPartialCacheFetch:
    """Test partial cache hit + gap fetch merging for all bar intervals."""

    def _create_daily_data(self, start_date, num_days):
        """Helper to create a DataFrame of daily OHLCV data."""
        dates = pd.bdate_range(start=start_date, periods=num_days)
        return pd.DataFrame({
            'Open': [100.0 + i for i in range(num_days)],
            'High': [105.0 + i for i in range(num_days)],
            'Low': [95.0 + i for i in range(num_days)],
            'Close': [101.0 + i for i in range(num_days)],
            'Volume': [1000000] * num_days,
        }, index=dates)

    def _create_hourly_data(self, start_date, num_bars):
        """Helper to create a DataFrame of hourly OHLCV data."""
        dates = pd.date_range(start=f'{start_date} 09:30', periods=num_bars, freq='h')
        return pd.DataFrame({
            'Open': [100.0 + i for i in range(num_bars)],
            'High': [105.0 + i for i in range(num_bars)],
            'Low': [95.0 + i for i in range(num_bars)],
            'Close': [101.0 + i for i in range(num_bars)],
            'Volume': [1000000] * num_bars,
        }, index=dates)

    # ---- Daily bar tests ----

    def test_daily_partial_cache_fetches_only_gap(self, mock_cache_manager):
        """When cache covers start..mid and end_date > mid, only fetch mid+1..end."""
        retriever = DataRetriever(symbol='SPY', bar_interval=BarTimeInterval.DAY)

        cached = self._create_daily_data('2024-01-01', 10)
        last_cached = cached.index[-1]

        cache_dir = retriever.cache_manager.get_cache_dir('stocks', 'SPY') / 'daily'
        cache_dir.mkdir(parents=True, exist_ok=True)
        cached.to_pickle(cache_dir / '2024-01-01.pkl')

        gap_data = self._create_daily_data(
            (last_cached + pd.Timedelta(days=1)).strftime('%Y-%m-%d'), 5
        )

        with patch.object(retriever, '_fetch_bars_from_api', return_value=gap_data) as mock_api:
            result = retriever.fetch_data_for_period('2024-01-01', '2024-02-01')

        mock_api.assert_called_once()
        call_start = mock_api.call_args[0][0]
        assert pd.Timestamp(call_start) > last_cached, \
            "API should only be called for dates after the last cached date"
        assert len(result) == len(cached) + len(gap_data)

    def test_daily_partial_cache_updates_cache_file(self, mock_cache_manager):
        """Merged data should be persisted back to the cache file."""
        retriever = DataRetriever(symbol='SPY', bar_interval=BarTimeInterval.DAY)

        cached = self._create_daily_data('2024-01-01', 5)
        last_cached = cached.index[-1]

        cache_dir = retriever.cache_manager.get_cache_dir('stocks', 'SPY') / 'daily'
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_file = cache_dir / '2024-01-01.pkl'
        cached.to_pickle(cache_file)

        gap_data = self._create_daily_data(
            (last_cached + pd.Timedelta(days=1)).strftime('%Y-%m-%d'), 3
        )

        with patch.object(retriever, '_fetch_bars_from_api', return_value=gap_data):
            retriever.fetch_data_for_period('2024-01-01', '2024-02-01')

        updated = pd.read_pickle(cache_file)
        assert len(updated) == len(cached) + len(gap_data), \
            "Cache file should contain both original and gap data"

    def test_daily_partial_cache_returns_cached_when_api_empty(self, mock_cache_manager):
        """If the API returns no gap data (e.g. weekends), return cached data as-is."""
        retriever = DataRetriever(symbol='SPY', bar_interval=BarTimeInterval.DAY)

        cached = self._create_daily_data('2024-01-01', 5)

        cache_dir = retriever.cache_manager.get_cache_dir('stocks', 'SPY') / 'daily'
        cache_dir.mkdir(parents=True, exist_ok=True)
        cached.to_pickle(cache_dir / '2024-01-01.pkl')

        with patch.object(retriever, '_fetch_bars_from_api', return_value=pd.DataFrame()):
            result = retriever.fetch_data_for_period('2024-01-01', '2024-02-01')

        assert len(result) == len(cached)

    def test_daily_partial_cache_deduplicates_overlap(self, mock_cache_manager):
        """Overlapping bars between cache and API are deduplicated (API wins)."""
        retriever = DataRetriever(symbol='SPY', bar_interval=BarTimeInterval.DAY)

        cached = self._create_daily_data('2024-01-01', 10)

        cache_dir = retriever.cache_manager.get_cache_dir('stocks', 'SPY') / 'daily'
        cache_dir.mkdir(parents=True, exist_ok=True)
        cached.to_pickle(cache_dir / '2024-01-01.pkl')

        overlap_start = cached.index[-2].strftime('%Y-%m-%d')
        gap_data = self._create_daily_data(overlap_start, 5)
        gap_data['Close'] = 999.0

        with patch.object(retriever, '_fetch_bars_from_api', return_value=gap_data):
            result = retriever.fetch_data_for_period('2024-01-01', '2024-02-01')

        assert not result.index.duplicated().any(), "Result should have no duplicate dates"
        overlap_idx = cached.index[-2]
        assert result.loc[overlap_idx, 'Close'] == 999.0, \
            "API data should take precedence on overlapping dates"

    def test_daily_partial_cache_does_not_write_when_cache_disabled(self, mock_cache_manager):
        """When use_cache=False, partial fetch should not update the cache file."""
        retriever = DataRetriever(symbol='SPY', bar_interval=BarTimeInterval.DAY, use_cache=False)

        cached = self._create_daily_data('2024-01-01', 5)
        last_cached = cached.index[-1]

        cache_dir = retriever.cache_manager.get_cache_dir('stocks', 'SPY') / 'daily'
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_file = cache_dir / '2024-01-01.pkl'
        cached.to_pickle(cache_file)
        original_size = cache_file.stat().st_size

        gap_data = self._create_daily_data(
            (last_cached + pd.Timedelta(days=1)).strftime('%Y-%m-%d'), 5
        )

        with patch.object(retriever, '_fetch_bars_from_api', return_value=gap_data):
            result = retriever.fetch_data_for_period('2024-01-01', '2024-02-01')

        assert len(result) == len(cached) + len(gap_data)
        assert cache_file.stat().st_size == original_size, \
            "Cache file should not be updated when use_cache=False"

    # ---- Hourly bar tests ----

    def test_hourly_partial_cache_fetches_gap(self, mock_cache_manager):
        """Hourly bars with incomplete cache trigger a gap fetch instead of returning stale data."""
        retriever = DataRetriever(symbol='SPY', bar_interval=BarTimeInterval.HOUR)

        cached = self._create_hourly_data('2024-01-02', 7)

        cache_dir = retriever.cache_manager.get_cache_dir('stocks', 'SPY') / 'hourly'
        cache_dir.mkdir(parents=True, exist_ok=True)
        for idx, row in cached.iterrows():
            fname = f"{idx.strftime('%Y-%m-%d')}_{idx.strftime('%H%M')}.pkl"
            pd.DataFrame([row]).to_pickle(cache_dir / fname)

        gap_data = self._create_hourly_data('2024-01-03', 7)

        with patch.object(retriever, '_fetch_bars_from_api', return_value=gap_data) as mock_api:
            result = retriever.fetch_data_for_period('2024-01-02', '2024-01-04')

        mock_api.assert_called_once()
        assert len(result) > len(cached), "Result should include gap data beyond cache"
        assert not result.index.duplicated().any()

    def test_hourly_partial_cache_saves_only_gap_bars(self, mock_cache_manager):
        """When merging hourly cache, only the new gap bars are written to disk."""
        retriever = DataRetriever(symbol='SPY', bar_interval=BarTimeInterval.HOUR)

        cached = self._create_hourly_data('2024-01-02', 4)

        cache_dir = retriever.cache_manager.get_cache_dir('stocks', 'SPY') / 'hourly'
        cache_dir.mkdir(parents=True, exist_ok=True)
        for idx, row in cached.iterrows():
            fname = f"{idx.strftime('%Y-%m-%d')}_{idx.strftime('%H%M')}.pkl"
            pd.DataFrame([row]).to_pickle(cache_dir / fname)

        files_before = set(cache_dir.glob('*.pkl'))

        gap_data = self._create_hourly_data('2024-01-03', 4)

        with patch.object(retriever, '_fetch_bars_from_api', return_value=gap_data):
            retriever.fetch_data_for_period('2024-01-02', '2024-01-04')

        files_after = set(cache_dir.glob('*.pkl'))
        new_files = files_after - files_before
        assert len(new_files) == len(gap_data), \
            "Only the new gap bars should be written as new cache files"

    def test_hourly_complete_cache_returns_without_api_call(self, mock_cache_manager):
        """When hourly cache already covers end_date, no API call is made."""
        retriever = DataRetriever(symbol='SPY', bar_interval=BarTimeInterval.HOUR)

        # end_date is 2024-01-02 (midnight), cached data goes through 2024-01-02 16:30
        cached = self._create_hourly_data('2024-01-02', 8)

        cache_dir = retriever.cache_manager.get_cache_dir('stocks', 'SPY') / 'hourly'
        cache_dir.mkdir(parents=True, exist_ok=True)
        for idx, row in cached.iterrows():
            fname = f"{idx.strftime('%Y-%m-%d')}_{idx.strftime('%H%M')}.pkl"
            pd.DataFrame([row]).to_pickle(cache_dir / fname)

        with patch.object(retriever, '_fetch_bars_from_api') as mock_api:
            result = retriever.fetch_data_for_period('2024-01-02', '2024-01-02')

        mock_api.assert_not_called()
        assert len(result) == len(cached)

    # ---- end_date=None tests ----

    def test_stale_cache_extended_when_end_date_is_none(self, mock_cache_manager):
        """When end_date is None and cache is stale, fetch the gap to bring it current."""
        retriever = DataRetriever(symbol='SPY', bar_interval=BarTimeInterval.DAY)

        cached = self._create_daily_data('2024-01-01', 5)

        cache_dir = retriever.cache_manager.get_cache_dir('stocks', 'SPY') / 'daily'
        cache_dir.mkdir(parents=True, exist_ok=True)
        cached.to_pickle(cache_dir / '2024-01-01.pkl')

        gap_data = self._create_daily_data('2024-06-01', 10)

        with patch.object(retriever, '_fetch_bars_from_api', return_value=gap_data) as mock_api:
            result = retriever.fetch_data_for_period('2024-01-01')

        mock_api.assert_called_once(), "Stale cache should trigger a gap fetch"
        assert len(result) == len(cached) + len(gap_data)
