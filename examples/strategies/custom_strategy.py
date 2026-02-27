from datetime import datetime
from decimal import Decimal

from algo_trading_engine import OptionsRetrieverHelper, Strategy
from algo_trading_engine.common.logger import get_logger
from algo_trading_engine.common.models import Option, StrategyType
from algo_trading_engine.dto import ExpirationRangeDTO, OptionsChainDTO, StrikeRangeDTO
from algo_trading_engine.indicators import ATRIndicator
from algo_trading_engine.enums import BarTimeInterval
from algo_trading_engine.vo import StrikePrice, create_position

class MyCustomStrategy(Strategy):
    """
    Example custom strategy using hourly bars and a simple ATR indicator.
    
    This is a simple example - you would implement your own logic here.
    The Strategy base class provides the interface that BacktestEngine expects.
    """
    
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
        - Analyze current market conditions using self.data
        - Check existing positions
        - Add new positions via add_position()
        - Close positions via remove_position()
        """
        # Add new positions
        if len(positions) == 0:
            self._try_open_position(date, add_position)
    
    def on_end(self, positions, remove_position, date):
        """
        Called at the end of backtest to close remaining positions.
        """
        for position in positions:
            exit_price = self.compute_exit_price(position, date) if hasattr(self, 'compute_exit_price') else None
            symbol = getattr(position, 'symbol', getattr(self, 'symbol', 'SPY'))
            underlying = self.get_current_underlying_price(date, symbol) if hasattr(self, 'get_current_underlying_price') else None
            remove_position(date, position, exit_price if exit_price is not None else 0.0, underlying_price=underlying)
    
    def validate_data(self, data):
        """
        Validate that the data has the required columns/features.
        
        Returns:
            True if valid, False otherwise
        """
        # Check for required columns
        required_columns = ['Close', 'Open', 'High', 'Low', 'Volume']
        return all(col in data.columns for col in required_columns)
    
    def _try_open_position(self, date: datetime, add_position):
        """
        Try to open a position.
        """
        current_price = self.get_current_underlying_price(date, self.symbol)
        if current_price is None:
            return

        expiration_range = ExpirationRangeDTO(min_days=5, max_days=10)
        strike_range = StrikeRangeDTO(min_strike=StrikePrice(Decimal(str(current_price - 1))), max_strike=StrikePrice(Decimal(str(current_price + 1))))

        chain = self.get_options_chain(
            date,
            current_price,
            expiration_range=expiration_range,
            strike_range=strike_range,
            timespan=BarTimeInterval.HOUR,
        )
        get_logger().info(f"Contracts: {len(chain.contracts)}")

        call, put = OptionsRetrieverHelper.find_atm_contracts(chain.contracts, current_price)
        if call is None:
            return
        call_bar = chain.get_bar_for_contract(call)
        if call_bar is None:
            get_logger().warning("No bar data for ATM call, skipping position")
            return

        call_option = Option.from_contract_and_bar(call, call_bar)
        entry_dt = date.to_pydatetime() if hasattr(date, "to_pydatetime") else date
        exp_date = datetime.strptime(str(call.expiration_date), "%Y-%m-%d")
        position = create_position(
            symbol=call.underlying_ticker,
            expiration_date=exp_date,
            strategy_type=StrategyType.LONG_CALL,
            strike_price=float(call.strike_price.value),
            entry_date=entry_dt,
            entry_price=call_option.last_price,
            spread_options=[call_option],
        )
        add_position(position)