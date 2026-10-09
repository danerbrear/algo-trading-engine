"""
Trading engine interfaces and implementations.

This module provides the abstract base class for trading engines
and concrete implementations for backtesting and paper trading.
"""

from abc import ABC, abstractmethod
from typing import Callable, List, Optional, Union
from datetime import datetime, timedelta
import pandas as pd

from algo_trading_engine import DataRetriever, Strategy
from algo_trading_engine.logging import get_logger
from algo_trading_engine.enums import BarTimeInterval, UniversalCloseCondition
from algo_trading_engine.models import EngineConfig
from algo_trading_engine.models.config import PaperTradingConfig
from algo_trading_engine.dto import OptionBarDTO, OptionContractDTO

# Minimum calendar lookback for LSTM / feature history in paper trading (aligned with prior default).
DEFAULT_PAPER_TRADING_LSTM_LOOKBACK_DAYS = 120


def compute_paper_trading_fetch_start_date(
    now: datetime,
    strategy: Strategy,
    bar_interval: BarTimeInterval,
    lstm_lookback_days: int = DEFAULT_PAPER_TRADING_LSTM_LOOKBACK_DAYS,
) -> str:
    """
    First calendar date (YYYY-MM-DD) for historical data fetch in paper trading.

    Mirrors backtest behavior: extend the window backward by
    ``strategy.get_warm_up_period_timedelta(bar_interval)`` from the effective
    end date (here, ``now``), and ensure at least ``lstm_lookback_days`` of
    history for models that need a fixed window.
    """
    lstm_start = now - timedelta(days=lstm_lookback_days)
    warmup_start = now - strategy.get_warm_up_period_timedelta(bar_interval)
    earliest = min(lstm_start, warmup_start)
    return earliest.strftime("%Y-%m-%d")


def make_rt_option_bar(
    options_handler,
) -> Callable[['OptionContractDTO'], Optional[OptionBarDTO]]:
    """
    Build a near-real-time option bar callable for live/paper trading.

    The returned callable fetches current prices via the Polygon snapshot endpoint, which is
    inherently "now" and takes no date. It returns None when no snapshot price is available
    (no historical /aggs fallback). Callers branch on Strategy.use_snapshot_for_current_bar,
    which BacktestConfig sets False and PaperTradingConfig sets True.
    """

    def _get_rt_option_bar(contract: 'OptionContractDTO') -> Optional[OptionBarDTO]:
        return options_handler.get_option_snapshot(contract)

    return _get_rt_option_bar


class TradingEngine(ABC):
    """
    Abstract base class for trading engines.
    
    Both backtesting and paper trading engines implement this interface,
    allowing for unified usage patterns.
    """

    def __init__(self, strategy: Strategy, data: pd.DataFrame, config: EngineConfig, bar_interval: 'BarTimeInterval' = None):
        self._strategy = strategy
        self._data = data
        self._config = config
        self.bar_interval = bar_interval
        
        strategy.get_current_underlying_price = self._get_current_underlying_price
    
    @abstractmethod
    def run(self) -> bool:
        """
        Execute the trading simulation.
        
        Returns:
            True if execution completed successfully, False otherwise
        """
    
    @abstractmethod
    def get_positions(self) -> List['Position']:
        """
        Get current open positions.
        
        Returns:
            List of currently open Position objects
        """
    
    @property
    def data(self) -> pd.DataFrame:
        """Get the market data."""
        return self._data
    
    @property
    @abstractmethod
    def strategy(self) -> Strategy:
        """Get the strategy being used by this engine."""
    
    @classmethod
    @abstractmethod
    def from_config(cls, config: Union['BacktestConfig', PaperTradingConfig]) -> 'TradingEngine':
        """
        Create trading engine from configuration.
        
        Factory method that handles all data fetching, strategy creation, and setup.
        Child projects only need to provide configuration.
        
        Args:
            config: Configuration DTO (BacktestConfig or PaperTradingConfig)
            
        Returns:
            Configured TradingEngine instance ready to run
            
        Raises:
            ValueError: If configuration is invalid or data fetching fails
        """

    def _get_current_underlying_price(self, date: datetime, symbol: str) -> Optional[float]:
        """
        Fetch and return the live price if the date is the current date, otherwise return last_price for the date.
        
        This method is injected into strategies so they can get current underlying prices
        without needing to manage DataRetriever themselves.
        
        Args:
            date: Date to get price for
            symbol: Symbol to fetch price for (e.g., 'SPY')
            
        Returns:
            Current underlying price as float, or None if unavailable
            
        Raises:
            ValueError: If live price fetch fails and date is current date
        """
        current_date = datetime.now().date()
        if date.date() == current_date:
            try:
                use_cache = True
                if hasattr(self, '_config') and hasattr(self._config, 'use_cache'):
                    use_cache = self._config.use_cache

                data_retriever = DataRetriever(symbol=symbol, use_cache=use_cache)
                live_price = data_retriever.get_live_price()
            except Exception as e:
                raise ValueError(f'Failed to fetch live price from DataRetriever: {e}') from e

            if live_price is not None:
                return live_price
            else:
                raise ValueError("Failed to fetch live price from DataRetriever.")
        else:
            # Historical date - return Close price from data
            try:
                return float(self.data.loc[date]['Close'])
            except (KeyError, IndexError):
                # If exact date not found, try to get closest available date
                try:
                    return float(self.data.loc[self.data.index <= date]['Close'].iloc[-1])
                except (IndexError, KeyError) as e:
                    raise ValueError(f"Could not find price data for date {date.date()}") from e
    
        
    def get_current_volumes_for_position(self, position: 'Position', date: Optional[datetime] = None) -> list[int]:
        """
        Fetch current date volume data for all options in a position using options_retriever.
        """
        if date is None:
            date = datetime.now()
            get_logger().warning("get_current_volumes_for_position was called without date; using datetime.now()")
        current_volumes = []
        
        # Check if position has spread_options
        if not hasattr(position, 'spread_options') or position.spread_options is None:
            return current_volumes

        get_current_option_bar = self._strategy.get_current_option_bar

        for option in position.spread_options:
            try:
                # Real-time snapshot when live, historical /aggs otherwise (handled by the strategy helper)
                bar_data = get_current_option_bar(option, date)

                if bar_data is None:
                    get_logger().warning(f"No option bar data available for {option.ticker} on {date.date()}")
                    current_volumes.append(None)
                    continue

                if bar_data and hasattr(bar_data, 'volume') and bar_data.volume is not None:
                    current_volumes.append(bar_data.volume)
                    get_logger().debug(f"Fetched volume data for {option.ticker} on {date.date()}: {bar_data.volume}")
                else:
                    current_volumes.append(None)
                    get_logger().warning(f"No volume data available for {option.ticker} on {date.date()}")

            except Exception as e:
                get_logger().warning(f"Error fetching volume data for {option.symbol}: {e}")
                current_volumes.append(None)
        return current_volumes
    
    def _get_valuation_bar(self, option: 'OptionContractDTO', date: datetime) -> Optional[OptionBarDTO]:
        """
        Fetch a bar for valuing a position leg.

        Delegates to Strategy.get_current_option_bar: near-real-time snapshot when live,
        historical /aggs (at the configured bar interval) during backtests.
        """
        return self.strategy.get_current_option_bar(option, date, timespan=self.bar_interval)

    def compute_exit_price(self, position: 'Position', date: datetime) -> Optional[float]:
        """
        Compute exit price for a position on a specific date.
        
        Args:
            position: Position to compute exit price for
            date: Date to compute exit price on
            
        Returns:
            Exit price or None if unavailable
        """
        try:
            if not position.spread_options:
                get_logger().warning("Position has no spread options")
                return None

            underlying_price = self.strategy.get_current_underlying_price(
                date,
                getattr(self.strategy, 'symbol', None),
            )

            if len(position.spread_options) == 1:
                option = position.spread_options[0]
                bar = self._get_valuation_bar(option, date)
                if not bar:
                    get_logger().warning(f"No bar data available for option: {option.ticker} on {date.date()}")
                    return None
                exit_price = position.calculate_exit_price_from_bars(bar, bar, underlying_price)
                return float(exit_price) if exit_price is not None else None

            atm_option, otm_option = position.spread_options
            atm_bar = self._get_valuation_bar(atm_option, date)
            otm_bar = self._get_valuation_bar(otm_option, date)
            if not atm_bar or not otm_bar:
                get_logger().warning(
                    "Missing bar data for spread options {} and {} on {}; attempting robust resolution",
                    atm_option.ticker,
                    otm_option.ticker,
                    date.date(),
                )
            exit_price = position.calculate_exit_price_from_bars(atm_bar, otm_bar, underlying_price)
            return float(exit_price) if exit_price is not None else None

        except Exception as e:
            get_logger().warning(f"Error computing exit price: {e}")
            return None
    
    def check_univeral_close_conditions(
        self,
        date: datetime,
        remove_position: Optional[
            Callable[[datetime, "Position", float, Optional[float], Optional[list[int]]], None]
        ] = None,
    ) -> None:
        """
        Check if the position should be closed due to universal close conditions.

        ``remove_position`` closes the position when a condition is met. When omitted, a met
        condition is logged and the position is left open.
        """
        # Get symbol from strategy if available, otherwise default to 'SPY'
        symbol = getattr(self.strategy, 'symbol', 'SPY')
        current_underlying_price = self.strategy.get_current_underlying_price(date, symbol)
        for position in self.get_positions():
            # Get current volumes for this specific position
            current_volumes = self.get_current_volumes_for_position(position, date)

            # Compute exit price for profit target and stop loss checks
            exit_price = self.compute_exit_price(position, date)

            def close_position(
                close_price: float,
                underlying_price: Optional[float] = None,
                position: "Position" = position,
                current_volumes: Optional[list[int]] = current_volumes,
            ) -> None:
                if remove_position is None:
                    get_logger().warning(
                        f"Close condition met for {position} but no remove_position callable was provided"
                    )
                    return
                remove_position(date, position, close_price, underlying_price, current_volumes)

            if self._should_close_due_to_assignment(position, date):
                get_logger().info(f"Position {position.__str__()} expired or near expiration (days to exp: {position.get_days_to_expiration(date)})")
                close_position(0.0, current_underlying_price)
            elif self._should_close_due_to_profit_target(position, exit_price):
                get_logger().info(f"Profit target hit for {position.__str__()} at exit {exit_price}")
                close_position(exit_price if exit_price is not None else 0.0)
            elif self._should_close_due_to_stop(position, exit_price):
                get_logger().info(f"Stop loss hit for {position.__str__()} at exit {exit_price}")
                close_position(exit_price if exit_price is not None else 0.0)
    
    def _should_close_due_to_assignment(self, position: 'Position', date: datetime) -> bool:
        if UniversalCloseCondition.ASSIGNMENT not in self.strategy.universal_close_conditions:
            return False
        try:
            return position.get_days_to_expiration(date) < 1
        except Exception:
            return False

    def _should_close_due_to_profit_target(self, position: 'Position', exit_price: Optional[float]) -> bool:
        if UniversalCloseCondition.PROFIT_TARGET not in self.strategy.universal_close_conditions:
            return False
        if exit_price is None or self.strategy.profit_target is None:
            return False
        return position.profit_target_hit(self.strategy.profit_target, exit_price)

    def _should_close_due_to_stop(self, position: 'Position', exit_price: Optional[float]) -> bool:
        if UniversalCloseCondition.STOP_LOSS not in self.strategy.universal_close_conditions:
            return False
        if exit_price is None or self.strategy.stop_loss is None:
            return False
        return position.stop_loss_hit(self.strategy.stop_loss, exit_price)
