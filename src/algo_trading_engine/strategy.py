"""
Strategy interface for trading strategies.

This module provides the abstract base class that all trading strategies must implement.
"""

from abc import ABC, abstractmethod
from datetime import datetime, timedelta
import math
from typing import (
    Any,
    Callable,
    ClassVar,
    Iterable,
    List,
    Optional,
    Protocol,
    Sequence,
    TYPE_CHECKING,
)
import pandas as pd
from pandera import DataFrameSchema

from algo_trading_engine._internal.util.market_clock import market_index, market_time
from algo_trading_engine.logging import get_logger
from algo_trading_engine.data_retriever import DataRetriever
from algo_trading_engine.vo import TreasuryRates, Universe
from algo_trading_engine.vo.universe import (
    PostLoadCallback,
    UniverseSet,
    default_ohlcv_schema,
    identity_post_load,
)
from algo_trading_engine.indicators.indicator import Indicator
from algo_trading_engine.enums import BarTimeInterval, UniversalCloseCondition

if TYPE_CHECKING:
    from algo_trading_engine.plotting.config import PlotConfig
    from algo_trading_engine.vo import Position
    from algo_trading_engine.dto import OptionBarDTO


class _NotImplementedCallback:
    """Callable stub that is not a function, so instance access does not bind it."""

    def __call__(self, *_args, **_kwargs):
        raise NotImplementedError("This logic must be overridden.")


class IndicatorUpdateError(RuntimeError):
    """Raised when an indicator fails to update for the current date.

    Strategy signals derived from a stale indicator are not trustworthy, so this
    aborts the run rather than letting it report success with stale values.
    """


class Strategy(ABC):
    """
    Abstract base class for trading strategies.

    All trading strategies must inherit from this class and implement
    the required abstract methods.

    Options Trading Callables:
        Strategies that trade options can expect the following callables as instance
        attributes. Each defaults to a raising stub until the strategy builder or engine
        replaces it:

        - get_contract_list_for_date: Get list of option contracts for an underlying and date
        - get_option_bar: Get historical bar data for an option contract on a specific date (/aggs)
        - get_rt_option_bar: Get near-real-time bar data for an option contract via Polygon
          snapshot. Injected for paper trading. Use ``use_snapshot_for_current_bar`` to choose
          snapshot pricing instead of historical pricing.
        - get_options_chain: Get the full options chain for a symbol on a specific date
        - get_current_volumes_for_position: Get current volumes for an open position (position, date)
        - compute_exit_price: Compute the exit price for a position on a specific date
        - get_position_size: Size a position from available capital
        - get_current_underlying_price: Underlying price for a date and symbol, assigned by the trading engine

    RemovePositionCallback is the close callback passed to ``on_new_date`` and ``on_end``.
    """

    class RemovePositionCallback(Protocol):
        """Close callback injected by a trading engine.

        ``underlying_price`` and ``current_volumes`` are optional, matching
        ``BacktestEngine._remove_position``.
        """

        def __call__(
            self,
            date: datetime,
            position: "Position",
            exit_price: float,
            underlying_price: Optional[float] = None,
            current_volumes: Optional[Sequence[Optional[int]]] = None,
        ) -> None:
            """Close ``position`` at ``exit_price``."""

    universe_set: UniverseSet
    universe_schema: ClassVar[DataFrameSchema] = default_ohlcv_schema()

    # Callables for options trading strategies. Default to a raising stub until injected.
    # Callable[..., Any] matches keyword calls and injected handlers without changing the stub.
    get_contract_list_for_date: Callable[..., Any] = _NotImplementedCallback()
    get_option_bar: Callable[..., Any] = _NotImplementedCallback()
    get_rt_option_bar: Callable[..., Any] = _NotImplementedCallback()
    get_options_chain: Callable[..., Any] = _NotImplementedCallback()
    get_current_volumes_for_position: Callable[..., Any] = _NotImplementedCallback()
    compute_exit_price: Callable[..., Any] = _NotImplementedCallback()
    get_position_size: Callable[..., Any] = _NotImplementedCallback()
    get_current_underlying_price: Callable[[datetime, str], Optional[float]] = _NotImplementedCallback()

    def __init__(
        self,
        profit_target: float | None = None,
        stop_loss: float | None = None,
        universal_close_conditions: Optional[Iterable[UniversalCloseCondition]] = None,
        use_snapshot_for_current_bar: bool = False,
    ):
        """
        Initialize the strategy.

        Args:
            profit_target: Optional profit target percentage (e.g., 0.5 for 50%)
            stop_loss: Optional stop loss percentage (e.g., 0.6 for 60%)
            universal_close_conditions: Engine-level close conditions to apply after
                on_new_date. Defaults to all members when omitted.
            use_snapshot_for_current_bar: When True, current option bars come from the
                Polygon snapshot. BacktestConfig sets this False and PaperTradingConfig
                sets this True when a strategy instance is supplied.
        """
        self.profit_target = profit_target
        self.stop_loss = stop_loss
        self.use_snapshot_for_current_bar = use_snapshot_for_current_bar
        if universal_close_conditions is None:
            self.universal_close_conditions = frozenset(UniversalCloseCondition)
        else:
            self.universal_close_conditions = frozenset(universal_close_conditions)
        self._data_override: Optional[pd.DataFrame] = None
        self.treasury_data: Optional[TreasuryRates] = None
        self.plot_config: Optional["PlotConfig"] = None
        self.indicators: List[Indicator] = []
        self.option_bar_timespan = BarTimeInterval.DAY
        universe_set = getattr(type(self), "universe_set", None)
        if not universe_set:
            raise TypeError(f"{type(self).__name__} must define a non-empty class attribute universe_set")
        self.universe = Universe(
            universe_set,
            schema=type(self).universe_schema,
            post_load=type(self).universe_post_load(DataRetriever(use_cache=False)),
        )

    @property
    def warm_up_period(self) -> int:
        """
        Number of initial bars used only to update indicators; the backtest engine does
        not call on_new_date until bar index ``warm_up_period`` (the first bar after
        this warm-up window). Derived from the most demanding indicator attached to the
        strategy.
        """
        if not self.indicators:
            return 0
        return max(indicator.warm_up_period for indicator in self.indicators)

    @abstractmethod
    def on_new_date(
        self,
        date: datetime,
        positions: tuple["Position", ...],
        add_position: Callable[["Position"], None],
        remove_position: "Strategy.RemovePositionCallback",
    ) -> None:
        """
        Called for each trading day to execute strategy logic.

        Args:
            date: Current trading date
            positions: Tuple of currently open positions
            add_position: Callback function to add a new position
            remove_position: Callback function to remove/close a position
                Signature: (date, position, exit_price, underlying_price=None, current_volumes=None)
        """
        if not self._update_positions(date, positions):
            get_logger().warning("Error updating positions.")

    @abstractmethod
    def on_end(
        self,
        positions: Sequence["Position"],
        remove_position: "Strategy.RemovePositionCallback",
        date: datetime,
    ) -> None:
        """
        Called at the end of backtest/paper trading to close remaining positions.

        Args:
            positions: Tuple of currently open positions
            remove_position: Callback function to remove/close a position
            date: Final date
        """

    @abstractmethod
    def validate_data(self, data: pd.DataFrame) -> bool:
        """
        Validate that the provided data meets the strategy's requirements.

        Args:
            data: DataFrame with market data and features

        Returns:
            True if data is valid, False otherwise
        """

    def get_current_option_bar(
        self,
        symbol: str,
        ticker: str,
        date: datetime,
        timespan: BarTimeInterval = BarTimeInterval.DAY,
    ) -> Optional["OptionBarDTO"]:
        """
        Fetch an option bar for current valuation.

        When ``use_snapshot_for_current_bar`` is set, use the near-real-time Polygon snapshot
        (dateless). Otherwise use the historical /aggs bar for ``date`` at the given ``timespan``.
        """
        if self.use_snapshot_for_current_bar and self.get_rt_option_bar:
            return self.get_rt_option_bar(symbol, ticker)

        if self.get_option_bar:
            return self.get_option_bar(symbol, ticker, date, 1, timespan)

        return None

    def get_warm_up_period_timedelta(self, bar_interval: BarTimeInterval) -> timedelta:
        """
        Time delta representing the warm-up period.
        """
        warm_up_period = (
            self.warm_up_period * 2.0
        )  # 2.0 is a factor to account for the fact that the warm-up period is a bar count, not a wall-clock time
        if bar_interval == BarTimeInterval.DAY:
            return timedelta(days=warm_up_period)
        elif bar_interval == BarTimeInterval.HOUR:
            if warm_up_period <= 0:
                return timedelta(0)
            trading_days = math.ceil(warm_up_period / 7)
            return timedelta(days=trading_days)
        else:
            raise ValueError(f"Unsupported bar interval: {bar_interval}")

    def add_indicator(self, indicator: Indicator) -> None:
        """
        Add an indicator to the strategy.

        This allows indicators to be added after strategy initialization,
        which is useful for dynamic indicator configuration.

        Args:
            indicator: Indicator instance to add

        Example:
            strategy = MyStrategy()
            atr = ATRIndicator(period=14)
            strategy.add_indicator(atr)
        """
        if not isinstance(indicator, Indicator):
            raise TypeError(f"Expected Indicator instance, got {type(indicator).__name__}")
        self.indicators.append(indicator)

    def get_indicator(self, indicator_class: type) -> Optional[Indicator]:
        """
        Get an indicator by class type.

        Args:
            indicator_class: The indicator class to search for (e.g., ATRIndicator)

        Returns:
            Indicator instance if found, None otherwise

        Example:
            atr = self.get_indicator(ATRIndicator)
            if atr:
                current_atr = atr.value
        """
        for indicator in self.indicators:
            if isinstance(indicator, indicator_class):
                return indicator
        return None

    def get_indicator_by_name(self, name: str) -> Optional[Indicator]:
        """
        Get an indicator by its name.

        Useful when multiple indicators of the same type exist (e.g. SMA_20 and SMA_50).

        Args:
            name: The indicator name to search for (e.g., "SMA_20", "ATR")

        Returns:
            Indicator instance if found, None otherwise

        Example:
            sma_20 = self.get_indicator_by_name("SMA_20")
            sma_50 = self.get_indicator_by_name("SMA_50")
        """
        for indicator in self.indicators:
            if indicator.name == name:
                return indicator
        return None

    def on_remove_position_success(
        self,
        date: datetime,
        position: "Position",
        exit_price: float,
        underlying_price: float | None = None,
        current_volumes: Optional[Sequence[Optional[int]]] = None,
    ):
        """
        Callback function for when a position is removed successfully via the TradingEngine.

        Args:
            date: Date at which the position is being closed
            position: Position to remove
            exit_price: Price at which the position is being closed
            underlying_price: Price of the underlying at the time of exit
            current_volumes: List of current volume data for each option in position.legs
        """

    def on_add_position_success(self, position: "Position"):
        """
        Callback function for when a position is added successfully via the TradingEngine.

        Args:
            position: Position to add
        """

    def get_current_bar_timestamp(self, date: datetime, symbol: str) -> Optional[datetime]:
        """Most recent bar for ``symbol`` at or before ``date``."""
        try:
            frame = self.universe.frame(symbol)
        except KeyError:
            return None
        if frame.empty:
            return None

        index = market_index(frame)
        candidates = index[index <= date]
        return market_time(candidates[-1] if len(candidates) > 0 else index[-1])

    @staticmethod
    def universe_post_load(retriever: DataRetriever) -> PostLoadCallback:
        """Build the universe ``post_load`` hook for this strategy."""
        del retriever
        return identity_post_load

    @property
    def data(self) -> Optional[pd.DataFrame]:
        """Primary symbol frame (first universe symbol), when a universe is loaded."""
        if self._data_override is not None:
            return self._data_override
        if not self.universe.symbols:
            return None
        try:
            return self.universe.frame(self.universe.symbols[0])
        except KeyError:
            return None

    @data.setter
    def data(self, value: Optional[pd.DataFrame]) -> None:
        """Allow tests and mocks to inject a primary frame without a full universe load."""
        self._data_override = value

    def market_frame(self, symbol: str) -> pd.DataFrame:
        """Return the loaded frame for ``symbol``."""
        try:
            return self.universe.frame(symbol)
        except KeyError as exc:
            raise ValueError(f"No loaded frame for {symbol}") from exc

    def set_universe(self, universe: Universe, treasury_data: Optional[TreasuryRates] = None) -> None:
        """Attach loaded market data and optional treasury rates."""
        self.universe = universe
        self._data_override = None
        self.treasury_data = treasury_data

    def finalize_universe(self) -> None:
        """Optional hook after universe frames are loaded; override in subclasses."""

    def validate_universe(self, universe: Universe) -> bool:
        """Validate every symbol frame in ``universe``."""
        for symbol in universe.symbols:
            if not self.validate_data(universe.frame(symbol)):
                get_logger().warning(f"validate_data failed for {symbol}")
                return False
        return True

    def set_plot_config(self, plot_config: Optional["PlotConfig"]) -> None:
        """Set plotting configuration for strategy-driven charts."""
        self.plot_config = plot_config

    def set_profit_target(self, profit_target: float):
        """Set the profit target for the strategy."""
        self.profit_target = profit_target

    def set_stop_loss(self, stop_loss: float):
        """Set the stop loss for the strategy."""
        self.stop_loss = stop_loss

    def _profit_target_hit(self, position: "Position", exit_price: Optional[float]) -> bool:
        """
        Check if the profit target has been hit for a position.

        Args:
            position: Position to check
            exit_price: Current exit price

        Returns:
            True if profit target hit, False otherwise
        """
        if self.profit_target is None:
            return False
        return position.profit_target_hit(self.profit_target, exit_price)

    def _stop_loss_hit(self, position: "Position", exit_price: Optional[float]) -> bool:
        """
        Check if the stop loss has been hit for a position.

        Args:
            position: Position to check
            exit_price: Current exit price

        Returns:
            True if stop loss hit, False otherwise
        """
        if self.stop_loss is None:
            return False
        return position.stop_loss_hit(self.stop_loss, exit_price)

    def _update_positions(self, date: datetime, positions: tuple["Position", ...]) -> bool:
        """Replace each open leg's quote and timestamp with the current option bar.

        A leg with no bar keeps its previous quote. Returns False when any leg
        could not be refreshed.
        """
        updated = True
        for position in positions:
            refreshed_legs: list = []
            for option in position.legs:
                try:
                    bar = self.get_current_option_bar(
                        position.symbol,
                        option.ticker,
                        date,
                        timespan=self.option_bar_timespan,
                    )
                except Exception as exc:
                    get_logger().warning(f"Error refreshing {getattr(option, 'ticker', option)} on {date}: {exc}")
                    refreshed_legs.append(option)
                    updated = False
                    continue
                if bar is None:
                    get_logger().warning(f"No option bar for {getattr(option, 'ticker', option)} on {date.date()}")
                    refreshed_legs.append(option)
                    updated = False
                    continue
                refreshed_legs.append(option.set_bar_data(bar))
            position.legs = refreshed_legs
        return updated
