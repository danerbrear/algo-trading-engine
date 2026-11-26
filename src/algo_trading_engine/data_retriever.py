import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from typing import ClassVar, Dict, Optional

from algo_trading_engine._internal.common.cache.cache_manager import CacheManager
from algo_trading_engine._internal.util.market_clock import (
    bars_through,
    market_index,
    market_now,
    market_time,
    to_market_dates,
    to_market_index,
)
from algo_trading_engine.logging import get_logger
from algo_trading_engine.vo import TreasuryRates
from algo_trading_engine.enums import BarTimeInterval


class DataRetriever:
    """Fetches and caches market data; does not bind frames or symbols on the instance."""

    _live_price_cache: ClassVar[Dict[str, float]] = {}

    def __init__(self, use_cache: bool = True):
        """Initialize DataRetriever.

        Args:
            use_cache: Whether to write fetched data to the local filesystem cache
        """
        self.use_cache = use_cache
        self.cache_manager = CacheManager(create_dirs=use_cache)

    @classmethod
    def clear_live_price_cache(cls) -> None:
        """Clear cached live prices. Intended for tests."""
        cls._live_price_cache.clear()

    def load_treasury_rates(self, start_date: datetime, end_date: Optional[datetime] = None) -> Optional[TreasuryRates]:
        """
        Load treasury rates from cache for the specified date range.

        Returns:
            TreasuryRates when data is found, otherwise None
        """
        if end_date is None:
            end_date = start_date

        get_logger().info(f"Loading treasury rates for date range: {start_date.date()} to {end_date.date()}")

        treasury_data = self.cache_manager.load_date_from_cache(start_date, "_treasury_rates", "treasury", "rates")

        if treasury_data is not None:
            get_logger().info(f"Loaded treasury rates for {start_date.date()}")
            return TreasuryRates(treasury_data)

        return self._find_closest_treasury_data(start_date)

    def fetch_data_for_period(
        self,
        symbol: str,
        bar_interval: BarTimeInterval,
        start_date: str,
        end_date: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Fetch data for a specific period with caching.

        When cached data exists but does not extend to the requested end_date,
        only the missing tail is fetched from the API and merged with the cache.
        """
        cached_data = self._load_cached_data_range(symbol, bar_interval, start_date, end_date)
        end_inclusive = self._effective_end_inclusive(end_date)

        if cached_data is not None and not cached_data.empty:
            effective_end = end_date if end_date is not None else end_inclusive.strftime("%Y-%m-%d")
            if not self._cache_is_complete(cached_data, end_date, end_inclusive):
                return self._fetch_gap_and_merge(symbol, bar_interval, cached_data, start_date, effective_end)

            interval_str = self._get_yfinance_interval(bar_interval)
            get_logger().info(f"Loaded {len(cached_data)} cached {interval_str} bars")
            return cached_data

        interval_str = self._get_yfinance_interval(bar_interval)

        get_logger().info(f"Fetching {interval_str} bars for {symbol} from {start_date} onwards...")

        ticker = yf.Ticker(symbol)

        info = ticker.info
        if not info or len(info) == 0:
            raise ValueError(f"Invalid symbol or no info available for {symbol}")

        start_date_ts = pd.Timestamp(start_date)
        yf_end = self._yfinance_end_exclusive(end_inclusive)

        data = ticker.history(start=start_date_ts, end=yf_end, interval=interval_str)

        if data.empty:
            raise ValueError(
                f"No data retrieved for {symbol} from {start_date} with interval '{interval_str}'. "
                f"This could be due to:\n"
                f"  1. Invalid date range for this interval\n"
                f"  2. Network/API connectivity issues\n"
                f"  3. Rate limiting from Yahoo Finance"
            )

        data.index = self._provider_index_to_market(data.index, bar_interval)
        get_logger().info(f"Fetched {len(data)} {interval_str} bars from {data.index[0]} to {data.index[-1]}")

        data = data.loc[(data.index >= start_date_ts) & (data.index <= end_inclusive)].copy()

        if data.empty:
            raise ValueError(f"No data available after filtering for date range {start_date} to {end_inclusive}")

        if self.use_cache:
            self._save_data_to_cache(symbol, bar_interval, data, start_date)

        return data

    # TODO: Move out of this file - DataRetriever should be reserved for data fetching only, not calculation.
    def calculate_features_for_data(self, data: pd.DataFrame, window=20):
        """Calculate technical features for a given dataset"""
        min_required_samples = max(window, 50) + 10

        if len(data) < min_required_samples:
            raise ValueError(
                f"Insufficient data for feature calculation: {len(data)} samples available, "
                f"need at least {min_required_samples} samples. "
                f"This ensures proper calculation of technical indicators like SMA50, RSI, and MACD."
            )

        data["Returns"] = data["Close"].pct_change()
        data["Log_Returns"] = np.log(data["Close"] / data["Close"].shift(1))

        data["Volatility"] = data["Returns"].rolling(window=window, min_periods=1).std()
        data["High_Low_Range"] = (data["High"] - data["Low"]) / data["Close"]

        data["SMA20"] = data["Close"].rolling(window=window, min_periods=1).mean()
        data["SMA50"] = data["Close"].rolling(window=50, min_periods=1).mean()
        data["Price_to_SMA20"] = data["Close"] / data["SMA20"]
        data["SMA20_to_SMA50"] = data["SMA20"] / data["SMA50"]

        data["RSI"] = self._calculate_rsi(data["Close"], window)
        macd_line, macd_signal = self._calculate_macd(data["Close"])
        data["MACD_Hist"] = macd_line - macd_signal

        data["SMA20_to_SMA50_Lag1"] = data["SMA20_to_SMA50"].shift(1)
        data["SMA20_to_SMA50_Lag5"] = data["SMA20_to_SMA50"].shift(5)
        data["SMA20_to_SMA50_MA5"] = data["SMA20_to_SMA50"].rolling(window=5, min_periods=1).mean()
        data["SMA20_to_SMA50_MA10"] = data["SMA20_to_SMA50"].rolling(window=10, min_periods=1).mean()
        data["SMA20_to_SMA50_Std5"] = data["SMA20_to_SMA50"].rolling(window=5, min_periods=1).std()
        data["SMA20_to_SMA50_Momentum"] = data["SMA20_to_SMA50"] - data["SMA20_to_SMA50"].shift(5)

        data["RSI_Lag1"] = data["RSI"].shift(1)
        data["RSI_Lag5"] = data["RSI"].shift(5)
        data["RSI_MA5"] = data["RSI"].rolling(window=5, min_periods=1).mean()
        data["RSI_MA10"] = data["RSI"].rolling(window=10, min_periods=1).mean()
        data["RSI_Std5"] = data["RSI"].rolling(window=5, min_periods=1).std()
        data["RSI_Momentum"] = data["RSI"] - data["RSI"].shift(5)
        data["RSI_Overbought"] = (data["RSI"] > 70).astype(int)
        data["RSI_Oversold"] = (data["RSI"] < 30).astype(int)

        data["Volume_SMA"] = data["Volume"].rolling(window=window, min_periods=1).mean()
        data["Volume_Ratio"] = data["Volume"] / data["Volume_SMA"]
        data["OBV"] = self._calculate_obv(data["Close"], data["Volume"])

        original_length = len(data)
        data.dropna(inplace=True)

        if len(data) == 0:
            raise ValueError(
                "All data was dropped due to NaN values after feature calculation. "
                "This indicates data quality issues or insufficient data for proper technical analysis."
            )

        if len(data) < original_length * 0.8:
            get_logger().warning(
                f"Lost {original_length - len(data)} samples due to NaN values, keeping {len(data)} samples"
            )

        get_logger().info(f"Calculated features for {len(data)} samples")

    def get_live_price(self, symbol: str) -> Optional[float]:
        """Fetch live price for the current date (cached per symbol for the process)."""
        if symbol in self._live_price_cache:
            cached_price = self._live_price_cache[symbol]
            get_logger().debug(f"Using cached live price for {symbol}: ${cached_price:.2f}")
            return cached_price

        try:
            get_logger().info(f"Fetching live price for {symbol} using yfinance...")
            ticker = yf.Ticker(symbol)
            info = ticker.info

            if "currentPrice" in info and info["currentPrice"] is not None:
                live_price = float(info["currentPrice"])
                get_logger().info(f"Live price for {symbol} (from yfinance): ${live_price:.2f}")
            elif "regularMarketPrice" in info and info["regularMarketPrice"] is not None:
                live_price = float(info["regularMarketPrice"])
                get_logger().info(f"Live price for {symbol} (from yfinance regular market): ${live_price:.2f}")
            elif "previousClose" in info and info["previousClose"] is not None:
                live_price = float(info["previousClose"])
                get_logger().warning(f"Using previous close for {symbol} (from yfinance): ${live_price:.2f}")
            else:
                get_logger().warning(f"No price data available from yfinance for {symbol}")
                return None

            self._live_price_cache[symbol] = live_price
            return live_price

        except Exception as yfinance_error:
            get_logger().error(f"Error fetching live price from yfinance for {symbol}: {str(yfinance_error)}")
            return None

    @staticmethod
    def _as_timestamp(value: datetime | str) -> pd.Timestamp:
        parsed = pd.Timestamp(value)
        if not isinstance(parsed, pd.Timestamp) or parsed is pd.NaT:
            raise ValueError(f"Invalid timestamp: {value!r}")
        return parsed

    @staticmethod
    def _read_cached_bars(cache_file: Path) -> pd.DataFrame:
        bars = pd.read_pickle(cache_file)
        if not isinstance(bars, pd.DataFrame):
            raise TypeError(f"{cache_file.name} does not hold a bar DataFrame")
        market_index(bars)
        return bars

    @staticmethod
    def _effective_end_inclusive(end_date: Optional[str]) -> pd.Timestamp:
        if end_date is None:
            return DataRetriever._as_timestamp(market_now())
        ts = DataRetriever._as_timestamp(end_date)
        if len(end_date.strip()) == 10:
            day = ts.date()
            return DataRetriever._as_timestamp(datetime(day.year, day.month, day.day, 23, 59, 59, 999999))
        return ts

    @staticmethod
    def _yfinance_end_exclusive(end_inclusive: pd.Timestamp) -> pd.Timestamp:
        day = end_inclusive.date() + timedelta(days=1)
        return DataRetriever._as_timestamp(datetime(day.year, day.month, day.day))

    @staticmethod
    def _end_spans_today(end_inclusive: pd.Timestamp) -> bool:
        return end_inclusive.date() >= market_now().date()

    def _cache_is_complete(
        self,
        cached_data: pd.DataFrame,
        end_date: Optional[str],
        end_inclusive: pd.Timestamp,
    ) -> bool:
        if self._end_spans_today(end_inclusive):
            return False
        last = market_time(market_index(cached_data)[-1])
        if end_date is not None and len(end_date.strip()) == 10:
            return last.date() >= end_inclusive.date()
        return last >= end_inclusive

    @staticmethod
    def _provider_index_to_market(index: pd.Index, bar_interval: BarTimeInterval) -> pd.DatetimeIndex:
        if bar_interval == BarTimeInterval.DAY:
            return to_market_dates(index)
        return to_market_index(index)

    @staticmethod
    def _get_yfinance_interval(bar_interval: BarTimeInterval) -> str:
        interval_map = {
            BarTimeInterval.MINUTE: "1m",
            BarTimeInterval.HOUR: "1h",
            BarTimeInterval.DAY: "1d",
        }
        return interval_map[bar_interval]

    @staticmethod
    def _get_cache_interval_dir(bar_interval: BarTimeInterval) -> str:
        interval_dir_map = {
            BarTimeInterval.MINUTE: "minute",
            BarTimeInterval.HOUR: "hourly",
            BarTimeInterval.DAY: "daily",
        }
        return interval_dir_map[bar_interval]

    def _find_closest_treasury_data(self, target_date: datetime) -> Optional[TreasuryRates]:
        treasury_dir = self.cache_manager.get_cache_dir("treasury", "rates")
        available_files = list(treasury_dir.glob("*_treasury_rates.pkl"))

        if not available_files:
            get_logger().warning("No treasury rate files found in cache")
            return None

        closest_file = None
        min_date_diff = float("inf")

        for file_path in available_files:
            try:
                date_str = file_path.stem.split("_")[0]
                file_date = datetime.strptime(date_str, "%Y-%m-%d")
                date_diff = abs((target_date - file_date).days)

                if date_diff < min_date_diff:
                    min_date_diff = date_diff
                    closest_file = file_path
            except (ValueError, IndexError):
                continue

        if closest_file:
            treasury_data = self.cache_manager.load_from_cache(closest_file.name, "treasury", "rates")
            if treasury_data is not None:
                get_logger().info(
                    f"Loaded closest treasury rates from {closest_file.name} (diff: {min_date_diff} days)"
                )
                return TreasuryRates(treasury_data)

        get_logger().warning("Could not find any treasury rate files")
        return None

    def _load_cached_data_range(
        self,
        symbol: str,
        bar_interval: BarTimeInterval,
        start_date: str,
        end_date: Optional[str] = None,
    ) -> Optional[pd.DataFrame]:
        if end_date is None:
            end_date = market_now().strftime("%Y-%m-%d")

        interval_dir = self._get_cache_interval_dir(bar_interval)
        cache_base = self.cache_manager.get_cache_dir("stocks", symbol)
        cache_dir = cache_base / interval_dir

        if not cache_dir.exists():
            return None

        if bar_interval == BarTimeInterval.DAY:
            cache_file = cache_dir / f"{start_date}.pkl"
            if cache_file.exists():
                try:
                    return bars_through(self._read_cached_bars(cache_file), self._effective_end_inclusive(end_date))
                except Exception as e:
                    get_logger().warning(f"Failed to load {cache_file.name}: {e}")
                    return None
            return None

        cache_files = sorted(cache_dir.glob("*.pkl"))

        if not cache_files:
            return None

        dfs = []
        start_ts = self._as_timestamp(start_date)
        end_ts = self._effective_end_inclusive(end_date)

        for cache_file in cache_files:
            try:
                file_date_str = cache_file.stem.split("_")[0]
                file_date = self._as_timestamp(file_date_str)

                if start_ts <= file_date <= end_ts:
                    dfs.append(self._read_cached_bars(cache_file))
            except Exception as e:
                get_logger().warning(f"Failed to load {cache_file.name}: {e}")
                return None

        if not dfs:
            return None

        return pd.concat(dfs).sort_index()

    def _fetch_bars_from_api(
        self, symbol: str, bar_interval: BarTimeInterval, start_date: str, end_date: str
    ) -> pd.DataFrame:
        interval_str = self._get_yfinance_interval(bar_interval)
        ticker = yf.Ticker(symbol)

        start_date_ts = pd.Timestamp(start_date)
        end_inclusive = self._effective_end_inclusive(end_date)
        yf_end = self._yfinance_end_exclusive(end_inclusive)

        data = ticker.history(
            start=start_date_ts,
            end=yf_end,
            interval=interval_str,
        )

        if data.empty:
            return pd.DataFrame()

        data.index = self._provider_index_to_market(data.index, bar_interval)
        data = data.loc[(data.index >= start_date_ts) & (data.index <= end_inclusive)].copy()

        return data

    def _save_data_to_cache(
        self, symbol: str, bar_interval: BarTimeInterval, data: pd.DataFrame, start_date: str
    ) -> None:
        if data.empty:
            return
        index = market_index(data)

        interval_dir = self._get_cache_interval_dir(bar_interval)
        interval_str = self._get_yfinance_interval(bar_interval)
        cache_base = self.cache_manager.get_cache_dir("stocks", symbol)
        cache_dir = cache_base / interval_dir
        cache_dir.mkdir(parents=True, exist_ok=True)

        today = market_now().date()
        data = data.loc[[ts.date() != today for ts in index]].copy()
        if data.empty:
            return

        if bar_interval == BarTimeInterval.DAY:
            cache_file = cache_dir / f"{start_date}.pkl"
            data.to_pickle(cache_file)
            get_logger().info(f"Cached {len(data)} daily bars to {interval_dir}/{start_date}.pkl")
        else:
            for label, row in data.iterrows():
                bar = market_time(label)
                cache_file = cache_dir / f"{bar:%Y-%m-%d}_{bar:%H%M}.pkl"
                pd.DataFrame([row]).to_pickle(cache_file)
            get_logger().info(f"Cached {len(data)} {interval_str} bars to {interval_dir}/")

    def _fetch_gap_and_merge(
        self,
        symbol: str,
        bar_interval: BarTimeInterval,
        cached_data: pd.DataFrame,
        start_date: str,
        end_date: str,
    ) -> pd.DataFrame:
        last_cached = market_time(market_index(cached_data)[-1])
        interval_str = self._get_yfinance_interval(bar_interval)

        if bar_interval == BarTimeInterval.DAY:
            gap_start = (last_cached + timedelta(days=1)).strftime("%Y-%m-%d")
        else:
            gap_start = last_cached.strftime("%Y-%m-%d")

        get_logger().info(
            f"Partial cache hit: {len(cached_data)} {interval_str} bars through "
            f"{last_cached.date()}, fetching remaining data from {gap_start} to {end_date}"
        )

        gap_data = self._fetch_bars_from_api(symbol, bar_interval, gap_start, end_date)

        if gap_data.empty:
            get_logger().info("No additional data available from API for the gap period")
            return cached_data

        get_logger().info(f"Fetched {len(gap_data)} additional {interval_str} bars from API")

        merged_data = pd.concat([cached_data, gap_data]).sort_index()
        merged_data = merged_data.loc[~merged_data.index.duplicated(keep="last")]

        if self.use_cache:
            if bar_interval == BarTimeInterval.DAY:
                self._save_data_to_cache(symbol, bar_interval, merged_data, start_date)
            else:
                self._save_data_to_cache(symbol, bar_interval, gap_data, start_date)

        return merged_data

    def _calculate_rsi(self, prices, window=14):
        if len(prices) < window + 1:
            raise ValueError(
                f"Insufficient data for RSI calculation: {len(prices)} samples available, "
                f"need at least {window + 1} samples for window size {window}."
            )

        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window, min_periods=1).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window, min_periods=1).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    def _calculate_macd(self, prices, fast=12, slow=26, signal=9):
        exp1 = prices.ewm(span=fast, adjust=False).mean()
        exp2 = prices.ewm(span=slow, adjust=False).mean()
        macd = exp1 - exp2
        signal_line = macd.ewm(span=signal, adjust=False).mean()
        return macd, signal_line

    def _calculate_obv(self, prices, volume):
        price_change = prices.diff()
        obv = pd.Series(index=prices.index, dtype=float)
        obv.iloc[0] = volume.iloc[0]

        for i in range(1, len(prices)):
            if price_change.iloc[i] > 0:
                obv.iloc[i] = obv.iloc[i - 1] + volume.iloc[i]
            elif price_change.iloc[i] < 0:
                obv.iloc[i] = obv.iloc[i - 1] - volume.iloc[i]
            else:
                obv.iloc[i] = obv.iloc[i - 1]

        return obv
