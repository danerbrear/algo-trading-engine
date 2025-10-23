"""
Upward Trend Reversal Strategy

This strategy capitalizes on upward trend reversals by trading SPY put debit spreads.
It identifies 3-10 day upward trends and enters put debit spread positions when these 
trends show signs of reversal, specifically targeting single-day trend-ending drawdowns.
"""

from typing import Callable, Optional, Dict, List, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
import pandas as pd

from src.backtest.models import Strategy, Position, StrategyType
from src.common.models import Option, OptionType, OptionChain
from src.common.options_handler import OptionsHandler
from src.common.options_dtos import StrikeRangeDTO, ExpirationRangeDTO, StrikePrice, ExpirationDate
from src.common.options_helpers import OptionsRetrieverHelper
from src.common.trend_detector import TrendDetector, TrendInfo
from decimal import Decimal


@dataclass
class SpreadInfo:
    """Information about a put debit spread."""
    atm_put: Option
    otm_put: Option
    net_debit: float
    width: float
    max_risk: float
    max_reward: float
    risk_reward_ratio: float
    dte: int


class UpwardTrendReversalStrategy(Strategy):
    """
    Strategy that trades put debit spreads on upward trend reversals.
    
    Entry Criteria:
    - 3-4 day upward trend detected
    - First day of negative returns after trend
    - Market is NOT in momentum uptrend regime
    - Valid put debit spread available (width <= 6, 5-10 DTE)
    
    Exit Criteria:
    - Stop loss threshold met
    - Profit target achieved
    - Holding period >= 2 days
    - 0 DTE (expiration)
    """
    
    def __init__(
        self,
        options_handler: OptionsHandler,
        min_trend_duration: int = 3,
        max_trend_duration: int = 4,
        max_spread_width: float = 6.0,
        min_dte: int = 5,
        max_dte: int = 10,
        max_risk_per_trade: float = 0.20,
        max_holding_days: int = 2,
        profit_target: float = None,
        stop_loss: float = None,
        start_date_offset: int = 60
    ):
        super().__init__(profit_target=profit_target, stop_loss=stop_loss, start_date_offset=start_date_offset)
        self.options_handler = options_handler
        self.min_trend_duration = min_trend_duration
        self.max_trend_duration = max_trend_duration
        self.max_spread_width = max_spread_width
        self.min_dte = min_dte
        self.max_dte = max_dte
        self.max_risk_per_trade = max_risk_per_trade
        self.max_holding_days = max_holding_days
        
        # Track detected trends for analysis
        self.detected_trends: List[TrendInfo] = []
    
    def validate_data(self, data: pd.DataFrame) -> bool:
        """
        Validate that data has required columns.
        
        Required columns:
        - Close: Daily close prices
        - Market_State: HMM market state classification
        """
        required_columns = ['Close']
        
        for col in required_columns:
            if col not in data.columns:
                print(f"❌ Missing required column: {col}")
                return False
        
        # Market_State is optional but recommended
        if 'Market_State' not in data.columns:
            print("⚠️  Warning: Market_State column not found. Market regime filtering will be disabled.")
        
        return True
    
    def on_new_date(
        self,
        date: datetime,
        positions: tuple[Position, ...],
        add_position: Callable[[Position], None],
        remove_position: Callable[[datetime, Position, float, float, list[int]], None]
    ):
        """
        Execute strategy logic for a new date.
        
        1. Check for position closures (stop loss, profit target, holding period, expiration)
        2. Check for entry signals (trend reversal + market regime filter)
        3. Open new position if no positions are currently open
        """
        # Close existing positions if exit criteria met
        self._try_close_positions(date, positions, remove_position)
        
        # Only open new positions if we have no open positions
        if len(positions) == 0:
            self._try_open_position(date, add_position)
    
    def on_end(
        self,
        positions: tuple[Position, ...],
        remove_position: Callable[[datetime, Position, float, float, list[int]], None],
        date: datetime
    ):
        """Close all remaining positions at the end of the backtest."""
        for position in positions:
            try:
                # Get current underlying price
                current_price = self.data.loc[date, 'Close']
                
                # Close position at expiration with assignment
                remove_position(date, position, None, current_price, None)
            except Exception as e:
                print(f"⚠️  Error closing position at end: {e}")
    
    def _detect_upward_trends(self, data: pd.DataFrame) -> List[TrendInfo]:
        """
        Detect upward trends in the data.
        
        An upward trend is defined as:
        - 3-10 consecutive days of positive daily returns
        - Followed by at least one day of negative returns
        - Net positive price increase over the trend period
        
        Returns:
            List of TrendInfo objects for detected trends
        """
        return TrendDetector.detect_forward_trends(
            data=data,
            min_duration=self.min_trend_duration,
            max_duration=self.max_trend_duration,
            reversal_threshold=0.02,
            require_reversal=True
        )
    
    def _is_momentum_uptrend_regime(self, date: datetime) -> bool:
        """
        Check if the market is in momentum uptrend regime.
        
        Uses HMM market state classification (state 1 = MOMENTUM_UPTREND).
        
        Returns:
            True if in momentum uptrend regime, False otherwise
        """
        if self.data is None or 'Market_State' not in self.data.columns:
            return False
        
        try:
            market_state = self.data.loc[date, 'Market_State']
            # State 1 = MOMENTUM_UPTREND (based on feature document)
            return market_state == 1
        except KeyError:
            return False
    
    def _find_put_debit_spread(
        self,
        date: datetime,
        current_price: float
    ) -> Optional[Dict]:
        """
        Find a suitable put debit spread for the given date and price.
        
        Criteria:
        - Width <= max_spread_width
        - 5-10 DTE
        - Valid risk/reward ratio
        - Sufficient volume on both legs
        
        Returns:
            Dictionary with spread information, or None if no suitable spread found
        """
        try:
            # Define strike and expiration ranges
            strike_range = StrikeRangeDTO(
                min_strike=StrikePrice(Decimal(str(current_price - 20))),
                max_strike=StrikePrice(Decimal(str(current_price + 20)))
            )
            
            expiration_range = ExpirationRangeDTO(
                min_days=self.min_dte,
                max_days=self.max_dte,
                current_date=date.date()
            )
            
            # Get contracts from options handler
            contracts = self.options_handler.get_contract_list_for_date(
                date=date,
                strike_range=strike_range,
                expiration_range=expiration_range
            )
            
            if not contracts:
                return None
            
            # Filter for puts only
            put_contracts = OptionsRetrieverHelper.find_contracts_by_type(
                contracts, OptionType.PUT
            )
            
            if len(put_contracts) < 2:
                return None
            
            # Find ATM and OTM puts for spread
            put_contracts_sorted = OptionsRetrieverHelper.sort_contracts_by_strike(
                put_contracts, ascending=False
            )
            
            # Try to find a suitable spread
            for i, atm_contract in enumerate(put_contracts_sorted):
                # ATM should be close to current price
                if abs(float(atm_contract.strike_price.value) - current_price) > 10:
                    continue
                
                # Look for OTM put (lower strike)
                for otm_contract in put_contracts_sorted[i+1:]:
                    width = float(atm_contract.strike_price.value - otm_contract.strike_price.value)
                    
                    # Check width constraint
                    if width > self.max_spread_width:
                        break  # Strikes are sorted, so no point checking further
                    
                    if width < 1:
                        continue
                    
                    # Check same expiration
                    if atm_contract.expiration_date.date != otm_contract.expiration_date.date:
                        continue
                    
                    # Get bar data for pricing
                    atm_bar = self.options_handler.get_option_bar(atm_contract, date)
                    otm_bar = self.options_handler.get_option_bar(otm_contract, date)
                    
                    if not atm_bar or not otm_bar:
                        continue
                    
                    # Check volume
                    if atm_bar.volume < 10 or otm_bar.volume < 10:
                        continue
                    
                    # Calculate net debit (we pay to enter)
                    atm_price = float(atm_bar.close_price)
                    otm_price = float(otm_bar.close_price)
                    net_debit = atm_price - otm_price
                    
                    if net_debit <= 0:
                        continue
                    
                    # Create Option objects
                    atm_option = Option(
                        ticker=atm_contract.ticker,
                        symbol=atm_contract.underlying_ticker,
                        strike=float(atm_contract.strike_price.value),
                        expiration=str(atm_contract.expiration_date),
                        option_type=OptionType.PUT,
                        last_price=atm_price,
                        volume=atm_bar.volume
                    )
                    
                    otm_option = Option(
                        ticker=otm_contract.ticker,
                        symbol=otm_contract.underlying_ticker,
                        strike=float(otm_contract.strike_price.value),
                        expiration=str(otm_contract.expiration_date),
                        option_type=OptionType.PUT,
                        last_price=otm_price,
                        volume=otm_bar.volume
                    )
                    
                    # Calculate risk/reward
                    max_risk = net_debit * 100  # Per contract
                    max_reward = (width - net_debit) * 100
                    risk_reward = max_reward / max_risk if max_risk > 0 else 0
                    
                    dte = atm_contract.days_to_expiration(date.date())
                    
                    return {
                        'atm_put': atm_option,
                        'otm_put': otm_option,
                        'net_debit': net_debit,
                        'width': width,
                        'max_risk': max_risk,
                        'max_reward': max_reward,
                        'risk_reward': risk_reward,
                        'dte': dte,
                        'expiration': str(atm_contract.expiration_date)
                    }
            
            return None
            
        except Exception as e:
            print(f"⚠️  Error finding put debit spread: {e}")
            return None
    
    def _calculate_position_size(
        self,
        position: Position,
        max_risk_pct: float,
        capital: float = 10000
    ) -> int:
        """
        Calculate position size based on max risk percentage.
        
        Args:
            position: Position with spread_options set
            max_risk_pct: Maximum risk as percentage of capital
            capital: Available capital
            
        Returns:
            Number of contracts to trade (minimum 1)
        """
        max_risk_dollars = capital * max_risk_pct
        per_contract_risk = position.get_max_risk()
        
        if per_contract_risk <= 0:
            return 1
        
        quantity = int(max_risk_dollars / per_contract_risk)
        return max(1, quantity)
    
    def _try_open_position(
        self,
        date: datetime,
        add_position: Callable[[Position], None]
    ):
        """
        Try to open a new position if entry criteria are met.
        
        Entry criteria:
        1. Recent upward trend of 3-4 days detected
        2. Trend ended with negative day (reversal)
        3. Market is NOT in momentum uptrend regime
        4. Valid put debit spread available
        """
        if self.data is None:
            return
        
        try:
            # Check if we're in momentum uptrend regime (filter out)
            if self._is_momentum_uptrend_regime(date):
                return
            
            # Get current price
            current_price = self.data.loc[date, 'Close']
            
            # Efficiently check if TODAY is a reversal day
            # This is much faster than scanning all trends
            data_slice = self.data.loc[:date]
            date_index = len(data_slice) - 1  # Current date is last index
            
            is_reversal, trend_info = TrendDetector.check_reversal_at_index(
                data=data_slice,
                date_index=date_index,
                min_trend_duration=self.min_trend_duration,
                max_trend_duration=self.max_trend_duration,
                reversal_threshold=0.02
            )
            
            if not is_reversal:
                return
            
            # Find a suitable put debit spread
            spread_info = self._find_put_debit_spread(date, current_price)
            
            if not spread_info:
                return
            
            # Create position
            position = Position(
                symbol=self.options_handler.symbol,
                expiration_date=datetime.strptime(spread_info['expiration'], '%Y-%m-%d'),
                strategy_type=StrategyType.PUT_DEBIT_SPREAD,
                strike_price=spread_info['atm_put'].strike,
                entry_date=date,
                entry_price=spread_info['net_debit'],
                spread_options=[spread_info['atm_put'], spread_info['otm_put']]
            )
            
            # Add position
            add_position(position)
            
        except Exception as e:
            print(f"⚠️  Error trying to open position: {e}")
    
    def _try_close_positions(
        self,
        date: datetime,
        positions: tuple[Position, ...],
        remove_position: Callable[[datetime, Position, float, float, list[int]], None]
    ):
        """
        Try to close positions based on exit criteria.
        
        Exit criteria:
        - Stop loss hit
        - Profit target hit
        - Holding period >= max_holding_days
        - 0 DTE (expiration)
        """
        for position in positions:
            try:
                # Calculate exit price
                exit_price = self._compute_exit_price(date, position)
                
                if exit_price is None:
                    continue
                
                # Check exit criteria
                should_close = False
                rationale = ""
                
                # Check expiration (0 DTE)
                dte = position.get_days_to_expiration(date)
                if dte < 1:
                    should_close = True
                    rationale = "expiration"
                    # Close at assignment
                    current_price = self.data.loc[date, 'Close']
                    remove_position(date, position, None, current_price, None)
                    continue
                
                # Check holding period
                days_held = position.get_days_held(date)
                if days_held >= self.max_holding_days:
                    should_close = True
                    rationale = "holding_period"
                
                # Check stop loss
                if self.stop_loss and self._stop_loss_hit(position, exit_price):
                    should_close = True
                    rationale = "stop_loss"
                
                # Check profit target
                if self.profit_target and self._profit_target_hit(position, exit_price):
                    should_close = True
                    rationale = "profit_target"
                
                if should_close:
                    remove_position(date, position, exit_price, None, None)
                    
            except Exception as e:
                print(f"⚠️  Error checking position closure: {e}")
    
    def _compute_exit_price(self, date: datetime, position: Position) -> Optional[float]:
        """
        Compute the current exit price for a position.
        
        Returns:
            Current exit price (net credit received when closing), or None if unavailable
        """
        try:
            if not position.spread_options or len(position.spread_options) != 2:
                return None
            
            atm_option, otm_option = position.spread_options
            
            # Get current bar data for both legs
            atm_contract_matches = self.options_handler.get_contract_list_for_date(
                date=date,
                strike_range=StrikeRangeDTO(
                    min_strike=StrikePrice(Decimal(str(atm_option.strike - 0.5))),
                    max_strike=StrikePrice(Decimal(str(atm_option.strike + 0.5)))
                ),
                expiration_range=ExpirationRangeDTO(
                    target_date=ExpirationDate(datetime.strptime(atm_option.expiration, '%Y-%m-%d').date()),
                    current_date=date.date()
                )
            )
            
            if not atm_contract_matches:
                return None
            
            atm_contract = atm_contract_matches[0]
            atm_bar = self.options_handler.get_option_bar(atm_contract, date)
            
            otm_contract_matches = self.options_handler.get_contract_list_for_date(
                date=date,
                strike_range=StrikeRangeDTO(
                    min_strike=StrikePrice(Decimal(str(otm_option.strike - 0.5))),
                    max_strike=StrikePrice(Decimal(str(otm_option.strike + 0.5)))
                ),
                expiration_range=ExpirationRangeDTO(
                    target_date=ExpirationDate(datetime.strptime(otm_option.expiration, '%Y-%m-%d').date()),
                    current_date=date.date()
                )
            )
            
            if not otm_contract_matches:
                return None
            
            otm_contract = otm_contract_matches[0]
            otm_bar = self.options_handler.get_option_bar(otm_contract, date)
            
            if not atm_bar or not otm_bar:
                return None
            
            # For put debit spread: we sell to close
            # Exit price = credit received = ATM price - OTM price
            exit_price = float(atm_bar.close_price) - float(otm_bar.close_price)
            
            return exit_price
            
        except Exception as e:
            print(f"⚠️  Error computing exit price: {e}")
            return None

