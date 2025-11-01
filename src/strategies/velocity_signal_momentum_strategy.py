from datetime import datetime
from typing import Callable, Optional, Dict
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from src.backtest.models import Strategy, Position, StrategyType, OptionChain, TreasuryRates
from src.common.options_dtos import ExpirationRangeDTO, OptionsChainDTO
from src.common.progress_tracker import progress_print
from src.model.options_handler import OptionsHandler
from src.common.options_handler import OptionsHandler as NewOptionsHandler
from src.common.options_helpers import OptionsRetrieverHelper
from src.common.models import OptionType
from src.common.options_dtos import StrikeRangeDTO, StrikePrice
from decimal import Decimal

class VelocitySignalMomentumStrategy(Strategy):
    """
    A momentum strategy to trade credit spreads in order to capitalize on 
    the upward or downward trends. 
    """

    # Configurable holding period in trading days
    holding_period = 4

    def __init__(self, start_date_offset: int = 60, stop_loss: float = None):
        super().__init__(start_date_offset=start_date_offset, stop_loss=stop_loss)

        self.new_options_handler = NewOptionsHandler(symbol='SPY')
        
        # Track position entries for plotting
        self._position_entries = []
    
    def set_data(self, data: pd.DataFrame, treasury_data: Optional[TreasuryRates] = None):
        super().set_data(data, treasury_data)
        
        # Reset position entries tracking for new backtest run
        self._position_entries = []
        
        # Pre-calculate moving averages and velocity for performance
        if self.data is not None and not self.data.empty:
            # Calculate SMA 15 and SMA 30
            self.data['SMA_15'] = self.data['Close'].rolling(window=15).mean()
            self.data['SMA_30'] = self.data['Close'].rolling(window=30).mean()
            
            # Calculate MA velocity (SMA 15 / SMA 30)
            self.data['MA_Velocity_15_30'] = self.data['SMA_15'] / self.data['SMA_30']
            
            # Calculate velocity changes for signal detection
            self.data['Velocity_Changes'] = self.data['MA_Velocity_15_30'].diff()

    def _recalculate_moving_averages(self):
        """Recalculate moving averages and velocity changes after data updates."""
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
            self._try_open_position(date, add_position)
            return
        self._try_close_positions(date, positions, remove_position)

    def on_end(self, positions: tuple['Position', ...], remove_position: Callable[['Position'], None], date: datetime):
        """
        Create a plot showing SPY price over time with position entry indicators.
        """
        if self.data is None or self.data.empty:
            progress_print("⚠️  No data available for plotting")
            return
        
        try:
            # Create the plot
            fig, ax = plt.subplots(figsize=(15, 8))
            
            # Plot SPY price
            ax.plot(self.data.index, self.data['Close'], 
                   label='SPY Close Price', color='blue', alpha=0.7, linewidth=1)
            
            # Get position entry dates from the backtest engine
            # We need to access the backtest engine's closed_positions to get entry dates
            # Since we don't have direct access, we'll track entries in the strategy itself
            if hasattr(self, '_position_entries'):
                entry_dates = self._position_entries
            else:
                # Fallback: try to get from the strategy's internal tracking
                entry_dates = []
                if hasattr(self, '_entry_dates'):
                    entry_dates = self._entry_dates
            
            # Plot position entry indicators
            if entry_dates:
                for entry_date in entry_dates:
                    if entry_date in self.data.index:
                        entry_price = self.data.loc[entry_date, 'Close']
                        ax.scatter(entry_date, entry_price, 
                                 color='red', s=100, marker='^', 
                                 label='Position Entry' if entry_date == entry_dates[0] else "", 
                                 zorder=5, alpha=0.8)
            
            # Add moving averages if they exist
            if 'SMA_15' in self.data.columns:
                ax.plot(self.data.index, self.data['SMA_15'], 
                       label='SMA 15', color='orange', alpha=0.6, linewidth=1)
            
            if 'SMA_30' in self.data.columns:
                ax.plot(self.data.index, self.data['SMA_30'], 
                       label='SMA 30', color='green', alpha=0.6, linewidth=1)
            
            # Format the plot
            num_positions = len(entry_dates) if entry_dates else 0
            title = f'SPY Price with Position Entries - Velocity Signal Momentum Strategy\nTotal Positions: {num_positions}'
            ax.set_title(title, fontsize=14, fontweight='bold')
            ax.set_xlabel('Date', fontsize=12)
            ax.set_ylabel('SPY Price ($)', fontsize=12)
            ax.legend(loc='upper left')
            ax.grid(True, alpha=0.3)
            
            # Add text box with strategy info
            strategy_info = f'Strategy: Velocity Signal Momentum\nHolding Period: {self.holding_period} days\nMA Periods: 15/30'
            ax.text(0.02, 0.98, strategy_info, transform=ax.transAxes, fontsize=10,
                   verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
            
            # Format x-axis dates
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
            plt.xticks(rotation=45)
            
            # Adjust layout
            plt.tight_layout()
            
            # Show the plot
            plt.show()
            
            # Save the plot to a file
            import os
            from datetime import datetime
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            plot_filename = f"velocity_strategy_positions_{timestamp}.png"
            plot_path = os.path.join("predictions", plot_filename)
            
            # Create predictions directory if it doesn't exist
            os.makedirs("predictions", exist_ok=True)
            
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            progress_print(f"📊 Position entry plot saved to: {plot_path}")
            progress_print("📊 Position entry plot generated successfully")
            
        except Exception as e:
            progress_print(f"⚠️  Error creating plot: {e}")
            import traceback
            traceback.print_exc()

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
        current_date = datetime.now().date()
        is_current_date = date.date() == current_date
        
        # If it's the current date, always fetch live price (even if date exists in cache)
        # to ensure we're using the most recent data during market hours
        if is_current_date:
            live_price = self._get_current_underlying_price(date)
            if live_price is not None:
                # Check if current date already exists in data (from stale cache)
                if date in self.data.index:
                    # Update existing row with fresh live price
                    self.data.loc[date, 'Close'] = live_price
                    self.data.loc[date, 'Open'] = live_price
                    self.data.loc[date, 'High'] = live_price
                    self.data.loc[date, 'Low'] = live_price
                    self.data.loc[date, 'Volume'] = 0
                    progress_print(f"🔄 Updated current date {date.date()} with live price ${live_price:.2f}")
                else:
                    # Create a new row with the live price
                    new_row = pd.DataFrame({
                        'Close': [live_price],
                        'Open': [live_price],
                        'High': [live_price],
                        'Low': [live_price],
                        'Volume': [0]
                    }, index=[date])
                    # Append to existing data
                    self.data = pd.concat([self.data, new_row])
                    progress_print(f"📅 Fetched live price ${live_price:.2f} for current date {date.date()} and appended to data")
                
                # Recalculate moving averages and velocity changes for the updated data
                self._recalculate_moving_averages()
                current_idx = self.data.index.get_loc(date)
            else:
                # Fallback: if live price fetch fails, try to use cached data if available
                if date in self.data.index:
                    current_idx = self.data.index.get_loc(date)
                    progress_print(f"⚠️ Could not fetch live price for {date.date()}, using cached data")
                else:
                    # No cached data and no live price - use last available
                    current_idx = len(self.data) - 1
                    progress_print(f"⚠️ Could not fetch live price for {date.date()}, using last available data point")
        else:
            # For historical dates, use cached data
            try:
                current_idx = self.data.index.get_loc(date)
            except KeyError:
                progress_print("⚠️  Date not found in data")
                return False
        
        # Check if we have enough data to analyze (need at least 30 days for SMA 30)
        if current_idx < 30:
            progress_print("⚠️  Not enough data to analyze")
            return False
        
        # Check if pre-calculated velocity data is available
        if 'Velocity_Changes' not in self.data.columns:
            progress_print("⚠️  No velocity changes data available")
            return False
        
        # Check if current velocity increased (positive velocity change)
        if current_idx < 1 or self.data['Velocity_Changes'].iloc[current_idx] <= 0:
            progress_print(f"Velocity changes: {self.data['Velocity_Changes'].iloc[current_idx]}")
            return False
        
        # This is a velocity signal - now check if it leads to a successful trend
        signal_index = current_idx
        
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
        Check if a trend signal is part of a successful upward trend by looking backward.
        This method always uses backward analysis for consistency between backtesting and live trading.
        
        Note: This strategy only considers upward trends for momentum trading.
        
        Args:
            data: DataFrame with price data
            signal_index: Index of the signal
            trend_type: Only 'up' is supported (downward trends are ignored)
            min_duration: Minimum trend duration in days
            max_duration: Maximum trend duration in days
            
        Returns:
            Tuple of (success, duration, return)
        """
        # Always use backward trend analysis for consistency
        return self._check_backward_trend_success(data, signal_index, trend_type, min_duration, max_duration)
    
    def _check_backward_trend_success(self, data: pd.DataFrame, signal_index: int, 
                                    trend_type: str, min_duration: int = 3, 
                                    max_duration: int = 60) -> tuple[bool, int, float]:
        """
        Check for successful upward trend by looking backward from the current date.
        This is used when we're on the current date and can't look forward.
        
        Note: This strategy only considers upward trends for momentum trading.
        
        Args:
            data: DataFrame with price data
            signal_index: Index of the signal (current date)
            trend_type: Only 'up' is supported (downward trends are ignored)
            min_duration: Minimum trend duration in days
            max_duration: Maximum trend duration in days
            
        Returns:
            Tuple of (success, duration, return)
        """
        current_price = data['Close'].iloc[signal_index]
        
        # Look backward for trend continuation
        for duration in range(min_duration, min(max_duration + 1, signal_index + 1)):
            start_index = signal_index - duration
            if start_index < 0:
                break
                
            start_price = data['Close'].iloc[start_index]
            total_return = (current_price - start_price) / start_price
            
            if trend_type == 'up':
                progress_print(f"🔍 Backward up trend detected for {duration} days, total return: {total_return}")
                if total_return > 0:
                    # Check if this is a sustained upward trend
                    # Look for any significant reversal within the trend period
                    trend_sustained = True
                    for j in range(start_index + 1, signal_index):
                        current_price_in_trend = data['Close'].iloc[j]
                        current_return = (current_price_in_trend - start_price) / start_price
                        if current_return < -0.02:  # 2% reversal threshold
                            trend_sustained = False
                            break
                    
                    if trend_sustained:
                        return True, duration, total_return
            # Note: This strategy only considers upward trends for momentum trading
            # Downward trends are not used as they don't align with the strategy's purpose
        
        return False, 0, 0.0
    
    def _create_put_credit_spread(self, date: datetime, current_price: float, expiration: str) -> Optional[Position]:
        """
        Create a test ATM/+10 put credit spread.
        
        Args:
            date: Current date
            current_price: Current underlying price
            expiration: Target expiration date string (YYYY-MM-DD)
            
        Returns:
            Position: Test position for evaluation, or None if creation fails
        """
        try:
            # Get list of contracts for the date
            expiration_range = ExpirationRangeDTO(min_days=5, max_days=10)
            
            # Add strike range filter to prevent super far ITM contracts
            strike_range = StrikeRangeDTO(
                min_strike=StrikePrice(Decimal(str(current_price - 7))),  # current_price - 7 (width of 6 + buffer)
                max_strike=StrikePrice(Decimal(str(current_price + 1)))    # current_price + 1
            )
            
            contracts = self.new_options_handler.get_contract_list_for_date(date, strike_range=strike_range, expiration_range=expiration_range)

            if not contracts:
                progress_print("⚠️  No contracts found for the date")
                return None

            # CRITICAL: Filter for contracts with the specific expiration FIRST
            # This ensures both legs have the same expiration (vertical spread, not diagonal)
            contracts_for_expiration = [
                c for c in contracts 
                if str(c.expiration_date) == expiration
            ]
            
            if not contracts_for_expiration:
                progress_print(f"⚠️  No contracts found for target expiration {expiration}")
                return None
            
            progress_print(f"✅ Found {len(contracts_for_expiration)} contracts for expiration {expiration}")

            # Find ATM put option from contracts with the target expiration
            atm_strike = round(current_price)
            atm_call, atm_put = OptionsRetrieverHelper.find_atm_contracts(contracts_for_expiration, current_price)
            
            if not atm_put:
                progress_print(f"⚠️  No ATM put found for expiration {expiration}")
                return None
                
            progress_print(f"Found ATM put: {atm_put.ticker} @ ${atm_put.strike_price.value} exp {atm_put.expiration_date}")
            
            # Find OTM put (-6 strike for 6-point width) from the same expiration
            otm_strike = atm_strike - 6
            
            # Filter for puts only (already filtered by expiration)
            puts_for_expiration = [
                c for c in contracts_for_expiration 
                if c.contract_type == OptionType.PUT
            ]
            
            if not puts_for_expiration:
                progress_print(f"⚠️  No put contracts found for expiration {expiration}")
                return None
                
            # Find the put with the closest strike to the target OTM strike
            otm_put = min(
                puts_for_expiration, 
                key=lambda put: abs(float(put.strike_price.value) - otm_strike)
            )
            
            min_difference = abs(float(otm_put.strike_price.value) - otm_strike)
            progress_print(f"Found OTM put: {otm_put.ticker} @ ${otm_put.strike_price.value} exp {otm_put.expiration_date} (strike difference: {min_difference})")
            
            # Verify both legs have the same expiration (vertical spread check)
            if str(atm_put.expiration_date) != str(otm_put.expiration_date):
                progress_print(f"❌ ERROR: Expiration mismatch! ATM: {atm_put.expiration_date}, OTM: {otm_put.expiration_date}")
                progress_print("❌ This would create a diagonal spread, not a vertical spread. Rejecting position.")
                return None
            
            progress_print(f"✅ Verified: Both legs have same expiration {expiration} (vertical spread)")
            
            # Get bar data to calculate net credit
            atm_bar = self.new_options_handler.get_option_bar(atm_put, date)
            otm_bar = self.new_options_handler.get_option_bar(otm_put, date)
            
            if not atm_bar or not otm_bar:
                progress_print("⚠️  No bar data available for credit calculation")
                return None
            
            # Calculate net credit (sell ATM, buy OTM)
            net_credit = float(atm_bar.close_price) - float(otm_bar.close_price)
            
            if net_credit > 0:  # Only consider if we receive a credit
                # Convert OptionContractDTO to Option using the new conversion method
                from src.common.models import Option
                atm_option = Option.from_contract_and_bar(atm_put, atm_bar)
                otm_option = Option.from_contract_and_bar(otm_put, otm_bar)
                
                # Create test position
                position = Position(
                    symbol=self.data.index.name if self.data.index.name else 'SPY',
                    expiration_date=datetime.strptime(expiration, '%Y-%m-%d'),
                    strategy_type=StrategyType.PUT_CREDIT_SPREAD,
                    strike_price=atm_strike,
                    entry_date=date,
                    entry_price=net_credit,
                    spread_options=[atm_option, otm_option]
                )
                # Set quantity for test position (1 contract)
                position.set_quantity(1)
                return position
            else:
                progress_print(f"⚠️  Negative credit: {net_credit:.2f}")
                return None
            
        except Exception as e:
            progress_print(f"⚠️  Error creating test put credit spread: {e}")
            return None
    
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

    # ==== Helper methods (opening) ====
    def _try_open_position(self, date: datetime, add_position: Callable[['Position'], None]):
        # Standard expiration target ~1 week
        if not self._has_buy_signal(date):
            return
        progress_print(f"📈 Buy signal detected for {date.strftime('%Y-%m-%d')}")

        current_price = self._get_current_underlying_price(date)
        if current_price is None:
            print("⚠️  Failed to get current price.")
            return

        # Select expiration (target ~1 week)
        expiration_str = self._select_week_expiration(date)
        if not expiration_str:
            progress_print("⚠️  Failed to select expiration")
            return
            
        position = self._create_put_credit_spread(date, current_price, expiration_str)
        if position is None:
            progress_print("⚠️  Failed to create put credit spread for selected expiration")
            return

        print("Current underlying price", current_price)
        
        # Track position entry for plotting
        self._position_entries.append(date)
        
        add_position(position)

    def _get_current_underlying_price(self, date: datetime) -> Optional[float]:
        """
        Fetch and return the live price if the date is the current date, otherwise return last_price for the date
        """
        current_date = datetime.now().date()
        if date.date() == current_date:
            # Use live price from DataRetriever if available
            if hasattr(self, 'data_retriever') and self.data_retriever:
                live_price = self.data_retriever.get_live_price()
                if live_price is not None:
                    return live_price
            else:
                raise ValueError("Data retriever is not initialized.")
        else:
            return float(self.data.loc[date]['Close'])

    def _get_option_chain(self, date: datetime, current_price: float) -> Optional[OptionsChainDTO]:
        date_key = date.strftime('%Y-%m-%d')

        expiration_range = ExpirationRangeDTO(min_days=5, max_days=10)
        
        strike_range = StrikeRangeDTO(
            min_strike=StrikePrice(Decimal(str(current_price - 7))),  # current_price - 7 (width of 6 + buffer)
            max_strike=StrikePrice(Decimal(str(current_price + 1)))    # current_price + 1
        )

        live_chain = self.new_options_handler.get_options_chain(date, current_price, strike_range=strike_range, expiration_range=expiration_range)
        if live_chain and (live_chain.get_calls() or live_chain.get_puts()):
            progress_print(f"✅ Successfully fetched live option chain with {len(live_chain.get_calls())} calls and {len(live_chain.get_puts())} puts")
            self.options_data[date_key] = live_chain
            return live_chain
        else:
            progress_print(f"⚠️  Live option chain fetch returned empty data")
            return None

    def _select_week_expiration(self, date: datetime) -> Optional[str]:
        """
        Select the best expiration date for the strategy using new_options_handler.
        Prefer expirations 5-10 days out, else nearest > 0 days, target 7 days.
        """
        progress_print(f"🔍 _select_week_expiration called for {date.strftime('%Y-%m-%d')}")
        target_days = 7
        
        def days_out(exp_str: str) -> int:
            try:
                exp_dt = datetime.strptime(exp_str, '%Y-%m-%d')
                return (exp_dt - date).days
            except Exception:
                return -9999
        
        # Try to get expirations from new_options_handler
        try:
            progress_print("🔍 Fetching expirations from new_options_handler for 5-10 day window...")
            
            # Use new_options_handler to get available expirations
            from src.common.options_dtos import ExpirationRangeDTO
            expiration_range = ExpirationRangeDTO(min_days=5, max_days=10)
            
            # Get contracts for the date (already filtered by expiration range)
            contracts = self.new_options_handler.get_contract_list_for_date(date, expiration_range=expiration_range)
            
            if not contracts:
                progress_print("⚠️  No contracts found for the date")
                return None
            
            # Extract unique expiration dates from contracts
            expirations = set(str(contract.expiration_date) for contract in contracts)
            progress_print(f"🔍 Found {len(expirations)} expirations from contracts")
            
            if not expirations:
                progress_print("⚠️  No expirations found in option chain")
                return None
                
            # Calculate days out for each expiration and select closest to target (7 days)
            valid_expirations = [(e, days_out(e)) for e in expirations]
            valid_expirations = [(e, d) for e, d in valid_expirations if d > 0]
            
            if not valid_expirations:
                progress_print("⚠️  No future expirations available")
                return None
                
            # Select the expiration closest to target (7 days)
            best_expiration = min(valid_expirations, key=lambda x: abs(x[1] - target_days))[0]
            days_to_exp = days_out(best_expiration)
            progress_print(f"✅ Selected expiration: {best_expiration} ({days_to_exp} days out)")
            
            return best_expiration
            
        except Exception as e:
            progress_print(f"❌ Error fetching expirations from new_options_handler: {str(e)}")
            return None
    

    # ==== Helper methods (closing) ====
    def _try_close_positions(self, date: datetime, positions: tuple['Position', ...], remove_position: Callable[['Position'], None]):
        current_underlying_price = self._get_current_underlying_price(date)
        progress_print(f"Current underlying price: {current_underlying_price}") 
        progress_print(f"🤖 Strategy evaluating {len(positions)} open position(s) for potential closure...")
               
        for position in positions:
            # Debug logging for position status
            days_held = position.get_days_held(date) if hasattr(position, 'get_days_held') else 0
            days_to_exp = position.get_days_to_expiration(date) if hasattr(position, 'get_days_to_expiration') else 0
            progress_print(f"🔍 Position {position.__str__()} - Days held: {days_held}, Days to exp: {days_to_exp}")

            # Assignment/expiration close
            if self._should_close_due_to_assignment(position, date):
                print(f"⏰ Position {position.__str__()} expired or near expiration (days to exp: {days_to_exp})")
                if current_underlying_price is not None:
                    current_volumes = self.get_current_volumes_for_position(position, date)
                    remove_position(date, position, 0.0, underlying_price=current_underlying_price, current_volumes=current_volumes)
                else:
                    progress_print("⚠️  Underlying price unavailable for assignment close; skipping.")
                continue

            # Compute exit price for stop/holding decisions
            exit_price, has_error = self._compute_exit_price(date, position)
            if not has_error and exit_price is not None:
                exit_price = self._sanitize_exit_price(exit_price)
                progress_print(f"💰 Calculated exit price for {position.__str__()}: {exit_price}")

            # Stop loss
            if self._should_close_due_to_stop(position, exit_price):
                print(f"🛑 Stop loss hit for {position.__str__()} at exit {exit_price}")
                current_volumes = self.get_current_volumes_for_position(position, date)
                remove_position(date, position, exit_price if exit_price is not None else 0.0, current_volumes=current_volumes)
                continue

            # Holding period
            if self._should_close_due_to_holding(position, date, self.holding_period):
                if exit_price is not None and not has_error:
                    print(f"📆 Holding period met for {position.__str__()} at exit {exit_price} (held {days_held} days, target: {self.holding_period})")
                    current_volumes = self.get_current_volumes_for_position(position, date)
                    remove_position(date, position, exit_price, current_volumes=current_volumes)
                else:
                    progress_print(f"⚠️  No exit price available for {position.__str__()} on {date}. Skipping holding-period close.")
            else:
                # Position not closed - show why
                progress_print(f"📋 Position {position.__str__()} remains open - Days held: {days_held}/{self.holding_period}, Days to exp: {days_to_exp}")
                progress_print(f"🤖 Strategy decision: Position does not meet closing criteria (holding period: {self.holding_period} days, stop loss, or expiration)")
        
        # Summary of strategy decisions
        progress_print(f"✅ Strategy evaluation complete - no positions closed on {date.strftime('%Y-%m-%d')}")

    def _compute_exit_price(self, date: datetime, position: Position) -> tuple[Optional[float], bool]:
        """Compute exit price using new_options_handler.get_option_bar and calculate_exit_price_from_bars"""
        try:
            if not position.spread_options or len(position.spread_options) != 2:
                progress_print("⚠️  Position doesn't have valid spread options")
                return None, True
                
            atm_option, otm_option = position.spread_options
            progress_print(f"🔍 Attempting to get bar data for {date.strftime('%Y-%m-%d')} - ATM: {atm_option.ticker}, OTM: {otm_option.ticker}")
            
            # Get bar data for both options using new_options_handler
            atm_bar = self.new_options_handler.get_option_bar(atm_option, date)
            otm_bar = self.new_options_handler.get_option_bar(otm_option, date)
            
            progress_print(f"🔍 Bar data results - ATM bar: {atm_bar is not None}, OTM bar: {otm_bar is not None}")
            
            if not atm_bar or not otm_bar:
                progress_print(f"⚠️  No bar data available for options on {date.strftime('%Y-%m-%d')} - ATM: {atm_bar is None}, OTM: {otm_bar is None}")
                return None, True
            
            # Use the new calculate_exit_price_from_bars method
            exit_price = position.calculate_exit_price_from_bars(atm_bar, otm_bar)
            progress_print(f"💰 Calculated exit price: {exit_price}")
            return exit_price, False
            
        except Exception as e:
            progress_print(f"⚠️  Error calculating exit price: {e}")
            import traceback
            traceback.print_exc()
            return None, True

    def _sanitize_exit_price(self, value: Optional[float]) -> Optional[float]:
        if value is None:
            return None
        return round(max(value, 0), 2)

    def _should_close_due_to_assignment(self, position: Position, date: datetime) -> bool:
        try:
            return position.get_days_to_expiration(date) < 1
        except Exception:
            return False

    def _should_close_due_to_stop(self, position: Position, exit_price: Optional[float]) -> bool:
        return (exit_price is not None) and self._stop_loss_hit(position, exit_price)

    def _should_close_due_to_holding(self, position: Position, date: datetime, holding_period: int) -> bool:
        try:
            return position.get_days_held(date) >= holding_period
        except Exception:
            return False


    def get_current_volumes_for_position(self, position: Position, date: datetime) -> list[int]:
        """
        Fetch current date volume data for all options in a position using new_options_handler.
        """
        current_volumes = []
        for option in position.spread_options:
            try:
                # Use new_options_handler.get_bar to get current volume data
                bar_data = self.new_options_handler.get_option_bar(option, date)
                
                if bar_data and bar_data.volume is not None:
                    current_volumes.append(bar_data.volume)
                    progress_print(f"📡 Fetched volume data for {option.ticker} on {date.date()}: {bar_data.volume}")
                else:
                    current_volumes.append(None)
                    progress_print(f"⚠️  No volume data available for {option.ticker} on {date.date()}")
                    
            except Exception as e:
                progress_print(f"⚠️  Error fetching volume data for {option.ticker}: {e}")
                current_volumes.append(None)
        return current_volumes

    def validate_data(self, data: pd.DataFrame) -> bool:
        """
        Validate the data for the Velocity Signal Momentum Strategy.
        
        This strategy requires:
        - Basic OHLCV data
        - Moving averages (SMA_15, SMA_30) for velocity calculation
        - Velocity changes data
        
        Args:
            data: DataFrame with market data and features
            
        Returns:
            bool: True if data is valid for this strategy, False otherwise
        """        
        progress_print(f"\n🔍 Validating data for Velocity Signal Momentum Strategy...")
        progress_print(f"   Data shape: {data.shape}")
        
        # Check if the data has the required columns for velocity momentum strategy
        required_columns = [
            'Open', 'High', 'Low', 'Close', 'Volume',  # Basic OHLCV data
        ]
        
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            progress_print(f"⚠️  Warning: Missing required columns: {missing_columns}")
            progress_print(f"   Available columns: {list(data.columns)}")
            return False
        else:
            progress_print(f"✅ All required columns present")
        
        # Check if data has datetime index
        if not isinstance(data.index, pd.DatetimeIndex):
            progress_print("❌ Error: Data must have a datetime index for backtesting")
            return False
        
        # Check if we have enough data for moving averages (need at least 30 days for SMA 30)
        if len(data) < 30:
            progress_print(f"⚠️  Warning: Not enough data for velocity analysis. Need at least 30 days, got {len(data)}")
            return False
        
        # Check for gaps in the data (missing trading days)
        if len(data) > 1:
            date_range = pd.bdate_range(start=data.index.min(), end=data.index.max())
            expected_business_days = len(date_range)
            actual_trading_days = len(data)
            if actual_trading_days < expected_business_days * 0.9:  # Allow for some holidays
                progress_print(f"⚠️  Warning: Data may have gaps. Expected ~{expected_business_days} business days, got {actual_trading_days}")
        
        progress_print(f"✅ Data validation complete for Velocity Signal Momentum Strategy")
        progress_print(f"   Final data shape: {data.shape}")
        progress_print(f"   Date range: {data.index.min()} to {data.index.max()}")
        progress_print(f"   Trading days: {len(data)}")
        
        return True
