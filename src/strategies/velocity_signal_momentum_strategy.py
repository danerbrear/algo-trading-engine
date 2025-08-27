from datetime import datetime
from typing import Callable, Optional, Dict
import pandas as pd

from src.backtest.models import Strategy, Position, StrategyType, OptionChain, TreasuryRates
from src.common.progress_tracker import progress_print
from src.model.options_handler import OptionsHandler

class VelocitySignalMomentumStrategy(Strategy):
    """
    A momentum strategy to trade credit spreads in order to capitalize on 
    the upward or downward trends. 
    """

    def __init__(self, options_handler: OptionsHandler, start_date_offset: int = 60):
        super().__init__(start_date_offset=start_date_offset)
        if options_handler is None:
            raise ValueError("options_handler is required for VelocitySignalMomentumStrategy")
        self.options_handler = options_handler
    
    def set_data(self, data: pd.DataFrame, options_data: Dict[str, OptionChain], treasury_data: Optional[TreasuryRates] = None):
        super().set_data(data, options_data, treasury_data)
        
        # Pre-calculate moving averages and velocity for performance
        if self.data is not None and not self.data.empty:
            # Calculate SMA 15 and SMA 30
            self.data['SMA_15'] = self.data['Close'].rolling(window=15).mean()
            self.data['SMA_30'] = self.data['Close'].rolling(window=30).mean()
            
            # Calculate MA velocity (SMA 15 / SMA 30)
            self.data['MA_Velocity_15_30'] = self.data['SMA_15'] / self.data['SMA_30']
            
            # Calculate velocity changes for signal detection
            self.data['Velocity_Changes'] = self.data['MA_Velocity_15_30'].diff()

    def on_new_date(self, date: datetime, positions: tuple['Position', ...], add_position: Callable[['Position'], None], remove_position: Callable[['Position'], None]):
        super().on_new_date(date, positions)

        if len(positions) == 0:
            # Determine if we should open a new position

            expiration_date = self._determine_expiration_date(date)

            if self._has_buy_signal(date):
                print(f"Buy signal detected for {date}")
        else:
            # Check if we should close any positions
            pass

    def on_end(self, positions: tuple['Position', ...], remove_position: Callable[['Position'], None], date: datetime):
        pass

    def _has_buy_signal(self, date: datetime) -> bool:
        """
        Check for buy signal using MA velocity (SMA 15/30) approach.
        
        If the data matches the following criteria for a buy signal, return True:
            - MA velocity (SMA 15/30) must increase (signal detected)
            - Price must increase over the trend period (3-60 days)
            - No significant reversals (>2% drop) during the trend
            - Trend must last at least 3 days
            - Trend must not exceed 60 days
        """
        if self.data is None or self.data.empty:
            return False
        
        # Get the current date index in the data
        try:
            current_idx = self.data.index.get_loc(date)
        except KeyError:
            return False
        
        # Check if we have enough data to analyze (need at least 30 days for SMA 30)
        if current_idx < 30:
            return False
        
        # Check if pre-calculated velocity data is available
        if 'Velocity_Changes' not in self.data.columns:
            return False
        
        # Check if current velocity increased (positive velocity change)
        if current_idx < 1 or self.data['Velocity_Changes'].iloc[current_idx] <= 0:
            return False
        
        # This is a velocity signal - now check if it leads to a successful trend
        signal_index = current_idx
        signal_price = self.data['Close'].iloc[signal_index]
        
        # Check if this leads to an upward trend of at least 3 days, max 60 days
        success, duration, trend_return = self._check_trend_success(
            self.data, signal_index, 'up', min_duration=3, max_duration=60
        )

        progress_print(f"Trend success: {success}, duration: {duration}, trend_return: {trend_return}")
        
        return success

    def _check_trend_success(self, data: pd.DataFrame, signal_index: int, 
                           trend_type: str, min_duration: int = 3, 
                           max_duration: int = 60) -> tuple[bool, int, float]:
        """
        Check if a trend signal leads to a successful trend.
        This method mimics the analysis module's approach.
        
        Args:
            data: DataFrame with price data
            signal_index: Index of the signal
            trend_type: 'up' or 'down'
            min_duration: Minimum trend duration in days
            max_duration: Maximum trend duration in days
            
        Returns:
            Tuple of (success, duration, return)
        """
        start_price = data['Close'].iloc[signal_index]
        
        # Look for trend continuation
        for duration in range(min_duration, min(max_duration + 1, len(data) - signal_index)):
            end_index = signal_index + duration
            if end_index >= len(data):
                break
                
            end_price = data['Close'].iloc[end_index]
            total_return = (end_price - start_price) / start_price
            
            if trend_type == 'up':
                if total_return > 0:
                    # Check if this is a sustained upward trend
                    # Look for any significant reversal within the trend period
                    trend_sustained = True
                    for j in range(signal_index + 1, end_index):
                        current_price = data['Close'].iloc[j]
                        current_return = (current_price - start_price) / start_price
                        if current_return < -0.02:  # 2% reversal threshold
                            trend_sustained = False
                            break
                    
                    if trend_sustained:
                        return True, duration, total_return
            else:  # down trend
                if total_return < 0:
                    # Check if this is a sustained downward trend
                    # Look for any significant reversal within the trend period
                    trend_sustained = True
                    for j in range(signal_index + 1, end_index):
                        current_price = data['Close'].iloc[j]
                        current_return = (current_price - start_price) / start_price
                        if current_return > 0.02:  # 2% reversal threshold
                            trend_sustained = False
                            break
                    
                    if trend_sustained:
                        return True, duration, total_return
        
        return False, 0, 0.0

    def _determine_expiration_date(self, date: datetime) -> datetime:
        """
        Find an expiration date by looking for the highest risk weighted return (Sharpe ratio)
        for an ATM/+10 put credit spread for each daily option chain. Use a 5-40 day range.
        
        Args:
            date: Current date to evaluate from
            
        Returns:
            datetime: Optimal expiration date with highest Sharpe ratio
        """
        if not self.options_data:
            raise ValueError("Options data is required for expiration date determination but is not available")
        
        if self.data is None or self.data.empty:
            raise ValueError("Market data is required for expiration date determination but is not available")
        
        if self.treasury_data is None:
            raise ValueError("Treasury data is required for Sharpe ratio calculation but is not available")
        
        date_key = date.strftime('%Y-%m-%d')
        if date_key not in self.options_data:
            progress_print(f"⚠️  No options data for {date_key}")
            return date + pd.Timedelta(days=30)
        
        current_price = self.data.loc[date]['Close']
        best_expiration = None
        best_sharpe_ratio = float('-inf')
        
        # Get all available option chains for this date
        option_chain = self.options_data[date_key]
        
        # Get expiration dates using options_handler
        expiration_dates = set()
        try:
            # Fetch contracts using options_handler to get all available expiration dates
            contracts = self.options_handler._fetch_filtered_option_contracts(date, current_price)
            if contracts:
                for contract in contracts:
                    if hasattr(contract, 'expiration_date') and contract.expiration_date:
                        expiration_dates.add(contract.expiration_date)
                progress_print(f"📊 Fetched {len(expiration_dates)} expiration dates using options_handler")
            else:
                progress_print(f"⚠️ No contracts found for {date.date()}")
                return date + pd.Timedelta(days=30)
        except Exception as e:
            progress_print(f"⚠️ Error fetching expiration dates from options_handler: {e}")
            return date + pd.Timedelta(days=30)
        
        progress_print(f"📊 Evaluating {len(expiration_dates)} expiration dates for optimal Sharpe ratio")
        
        for expiration_str in sorted(expiration_dates):
            try:
                expiration_date = datetime.strptime(expiration_str, '%Y-%m-%d')
                days_to_expiration = (expiration_date - date).days
                
                # Check if expiration is within our target range (5-40 days)
                if days_to_expiration < 5 or days_to_expiration > 40:
                    continue
                
                # Create ATM/+10 put credit spread for this expiration
                position = self._create_test_put_credit_spread(date, current_price, expiration_str)
                
                if position is None:
                    continue
                
                # Calculate Sharpe ratio for this position
                sharpe_ratio = self._calculate_sharpe_ratio(position, date)
                
                progress_print(f"   {expiration_str} ({days_to_expiration} days): Sharpe = {sharpe_ratio:.3f}")
                
                # Update best if this Sharpe ratio is higher
                if sharpe_ratio > best_sharpe_ratio:
                    best_sharpe_ratio = sharpe_ratio
                    best_expiration = expiration_date
                    
            except Exception as e:
                progress_print(f"⚠️  Error evaluating expiration {expiration_str}: {e}")
                continue
        
        if best_expiration is None:
            progress_print("⚠️  No suitable expiration dates found, using default 30 days")
            return date + pd.Timedelta(days=30)
        
        progress_print(f"✅ Selected expiration date: {best_expiration.strftime('%Y-%m-%d')} (Sharpe: {best_sharpe_ratio:.3f})")
        return best_expiration
    
    def _create_test_put_credit_spread(self, date: datetime, current_price: float, expiration: str) -> Optional[Position]:
        """
        Create a test ATM/+10 put credit spread for Sharpe ratio calculation.
        
        Args:
            date: Current date
            current_price: Current underlying price
            expiration: Expiration date string
            
        Returns:
            Position: Test position for evaluation, or None if creation fails
        """
        try:
            # Find ATM put option
            atm_strike = round(current_price)
            atm_put = None
            
            # Look for ATM put in the options data
            date_key = date.strftime('%Y-%m-%d')
            if date_key in self.options_data:
                option_chain = self.options_data[date_key]
                
                # Find ATM put
                for put in option_chain.puts:
                    if put.expiration == expiration and abs(put.strike - atm_strike) <= 5:
                        atm_put = put
                        break
                
                # Find OTM put (+10 strike)
                otm_strike = atm_strike - 10
                otm_put = None
                
                for put in option_chain.puts:
                    if put.expiration == expiration and put.strike <= otm_strike:
                        otm_put = put
                        break
                
                if atm_put and otm_put:
                    # Calculate net credit (sell ATM, buy OTM)
                    net_credit = atm_put.last_price - otm_put.last_price
                    
                    if net_credit > 0:  # Only consider if we receive a credit
                        # Create test position
                        position = Position(
                            symbol=self.data.index.name if self.data.index.name else 'SPY',
                            expiration_date=datetime.strptime(expiration, '%Y-%m-%d'),
                            strategy_type=StrategyType.PUT_CREDIT_SPREAD,
                            strike_price=atm_strike,
                            entry_date=date,
                            entry_price=net_credit,
                            spread_options=[atm_put, otm_put]
                        )
                        # Set quantity for test position (1 contract)
                        position.set_quantity(1)
                        return position
            
            return None
            
        except Exception as e:
            progress_print(f"⚠️  Error creating test put credit spread: {e}")
            return None

    def _calculate_sharpe_ratio(self, position: Position, current_date: datetime) -> float:
        """
        Calculate the Sharpe ratio for a position using max profit potential.
        
        Args:
            position: The position to analyze
            current_date: Current date for calculations
            
        Returns:
            float: Sharpe ratio for the position
        """
        # Get risk-free rate for the current date
        risk_free_rate = self._get_risk_free_rate(current_date)
        
        # Calculate position return using max profit (credit received)
        if position.spread_options:
            # Use the credit received as the max profit potential
            max_profit = position.entry_price  # This is the credit received
            
            # Calculate percentage return based on max risk
            max_risk = position.get_max_risk()
            if max_risk > 0:
                percentage_return = max_profit / max_risk
                
                # For a single position, we need to estimate volatility
                # Use a simple approach: assume volatility based on underlying asset
                if self.data is not None and not self.data.empty:
                    # Calculate historical volatility of the underlying
                    returns = self.data['Close'].pct_change().dropna()
                    if len(returns) > 0:
                        volatility = returns.std()
                        
                        # Calculate Sharpe ratio: (Return - Risk_Free_Rate) / Volatility
                        sharpe_ratio = (percentage_return - risk_free_rate) / volatility
                        return sharpe_ratio
        
        raise ValueError("Unable to calculate Sharpe ratio for position")
    
    def _get_risk_free_rate(self, date: datetime) -> float:
        """
        Get the risk-free rate for a specific date using the treasury data.
        
        Args:
            date: The date to get the risk-free rate for
            
        Returns:
            float: Risk-free rate (default 0.0 if not available)
        """
        if self.treasury_data is None:
            progress_print("⚠️  No treasury data available, using default 2.0%")
            return 2.0
            
        return float(self.treasury_data.get_risk_free_rate(date))
