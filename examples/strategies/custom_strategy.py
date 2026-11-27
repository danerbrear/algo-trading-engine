from datetime import datetime
from decimal import Decimal

from algo_trading_engine import OptionsRetrieverHelper, Strategy
from algo_trading_engine.logging import get_logger
from algo_trading_engine.vo import Option, StrikePrice, create_position
from algo_trading_engine.vo.universe import UniverseSet
from algo_trading_engine.dto import ExpirationRangeDTO, StrikeRangeDTO
from algo_trading_engine.indicators import ATRIndicator
from algo_trading_engine._internal.util.date_formats import DateFormats
from algo_trading_engine.enums import BarTimeInterval, StrategyType


class MyCustomStrategy(Strategy):
    """
    Example custom strategy using hourly bars and a simple ATR indicator.

    This is a simple example - you would implement your own logic here.
    The Strategy base class provides the interface that BacktestEngine expects.
    """

    universe_set: UniverseSet = ["SPY"]

    def __init__(self, profit_target=0.5, stop_loss=0.6):
        """Initialize the custom strategy."""
        super().__init__(profit_target, stop_loss)

        # Example indicator
        atr_indicator = ATRIndicator(period=20, period_unit=BarTimeInterval.DAY)
        self.add_indicator(atr_indicator)

        # Add any custom initialization here

    def on_new_date(self, date, positions, add_position, remove_position):
        """
        Called for each trading day.

        Implement your strategy logic here:
        - Analyze current market conditions using self.universe.frame(symbol)
        - Check existing positions
        - Add new positions via add_position()
        - Close positions via remove_position()
        """
        super().on_new_date(date, positions, add_position, remove_position)
        get_logger().info(f"on_new_date called for date: {date}")
        for symbol in self.universe.symbols:
            symbol_positions = tuple(p for p in positions if p.symbol == symbol)
            if len(symbol_positions) == 0:
                self._try_open_position(date, add_position, symbol)

    def on_end(self, positions, remove_position, date):
        """
        Called at the end of backtest to close remaining positions.
        """
        for position in positions:
            exit_price = self.compute_exit_price(position, date)
            underlying = self.get_current_underlying_price(date, position.symbol)
            remove_position(date, position, exit_price if exit_price is not None else 0.0, underlying_price=underlying)

    def _try_open_position(self, date: datetime, add_position, symbol: str):
        """
        Try to open a position.
        """
        current_price = self.get_current_underlying_price(date, symbol)
        if current_price is None:
            return

        expiration_range = ExpirationRangeDTO(min_days=5, max_days=10)
        strike_range = StrikeRangeDTO(
            min_strike=StrikePrice(Decimal(str(current_price - 1))),
            max_strike=StrikePrice(Decimal(str(current_price + 1))),
        )

        chain = self.get_options_chain(
            symbol,
            date,
            current_price,
            expiration_range=expiration_range,
            strike_range=strike_range,
            timespan=BarTimeInterval.HOUR,
        )
        get_logger().info(f"Contracts: {len(chain.contracts)}")

        call, _ = OptionsRetrieverHelper.find_atm_contracts(chain.contracts, current_price)
        if call is None:
            return
        # Price the chosen leg via the live-aware helper: near-real-time snapshot when paper
        # trading, historical /aggs (at the chain's timespan) when backtesting.
        call_bar = self.get_current_option_bar(symbol, call.ticker, date, timespan=BarTimeInterval.HOUR)
        if call_bar is None:
            get_logger().warning("No bar data for ATM call, skipping position")
            return

        call_option = Option.from_contract_and_bar(call, call_bar)
        exp_date = datetime.strptime(str(call.expiration_date), DateFormats.EXPIRATION)
        position = create_position(
            symbol=call.underlying_ticker,
            expiration_date=exp_date,
            strategy_type=StrategyType.LONG_CALL,
            strike_price=float(call.strike_price.value),
            entry_date=date,
            entry_price=call_option.last_price,
            legs=[call_option],
        )
        add_position(position)
