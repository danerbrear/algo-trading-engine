from datetime import datetime
from typing import Callable, Optional, Dict
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from src.backtest.models import Strategy, Position, StrategyType, OptionChain, TreasuryRates
from src.common.models import OptionType
from src.common.progress_tracker import progress_print
from src.model.options_handler import OptionsHandler

class VelocitySignalMomentumStrategy(Strategy):
    """
    A momentum strategy to trade credit spreads in order to capitalize on 
    the upward or downward trends. 
    """

    # Configurable holding period in trading days
    holding_period = 5

    def __init__(self, options_handler: OptionsHandler, start_date_offset: int = 60):
        super().__init__(start_date_offset=start_date_offset)
        if options_handler is None:
            raise ValueError("options_handler is required for VelocitySignalMomentumStrategy")
        self.options_handler = options_handler
        # Track position entries for plotting
        self._position_entries = []
    
    def set_data(self, data: pd.DataFrame, options_data: Dict[str, OptionChain], treasury_data: Optional[TreasuryRates] = None):
        super().set_data(data, options_data, treasury_data)
        
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
    
    def _create_put_credit_spread(self, date: datetime, current_price: float, expiration: str) -> Optional[Position]:
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
                
                # Find OTM put (-10 strike)
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
        chain = self._get_option_chain(date)
        if chain is None:
            progress_print("⚠️  Failed to get option chain for: {}", date.strftime('%Y-%m-%d'))
            return
        expiration_str = self._select_week_expiration(date, chain)
        progress_print(f"Selected expiration: {expiration_str}")
        if not expiration_str:
            progress_print("⚠️  Failed to get valid expiration date")
            return
        position = self._create_put_credit_spread(date, current_price, expiration_str)
        if position is None:
            progress_print("⚠️  Failed to create put credit spread for selected expiration")
            return
        
        # Track position entry for plotting
        self._position_entries.append(date)
        
        add_position(position)

    def _get_current_underlying_price(self, date: datetime) -> Optional[float]:
        if self.data is None or self.data.empty or date not in self.data.index:
            return None
        try:
            return float(self.data.loc[date]['Close'])
        except Exception:
            return None

    def _get_option_chain(self, date: datetime) -> Optional[OptionChain]:
        date_key = date.strftime('%Y-%m-%d')
        if not self.options_data or date_key not in self.options_data:
            progress_print(f"⚠️  No options data available for {date_key}")
            return None
        return self.options_data[date_key]

    def _select_week_expiration(self, date: datetime, chain: OptionChain) -> Optional[str]:
        # Prefer expirations 5-10 days out, else nearest > 0 days, target 7
        target_days = 7
        expirations = set(p.expiration for p in chain.puts) if chain and chain.puts else set()
        if not expirations:
            progress_print("⚠️  No put expirations found in option chain")
            return None
        def days_out(exp_str: str) -> int:
            try:
                exp_dt = datetime.strptime(exp_str, '%Y-%m-%d')
                return (exp_dt - date).days
            except Exception:
                return -9999
        valid = [(e, days_out(e)) for e in expirations]
        valid = [(e, d) for e, d in valid if d > 0]
        if not valid:
            progress_print("⚠️  No future expirations available")
            return None
        window = [(e, d) for e, d in valid if 5 <= d <= 10]
        if window:
            candidates = window
        else:
            progress_print("No expirations within window")

            # TODO: Fetch list of contracts for specific window in case cached data doesn't contain valid expirations

        return min(candidates, key=lambda x: abs(x[1] - target_days))[0]
    

    # ==== Helper methods (closing) ====
    def _try_close_positions(self, date: datetime, positions: tuple['Position', ...], remove_position: Callable[['Position'], None]):
        current_underlying_price = self._get_current_underlying_price(date)
        for position in positions:
            # Assignment/expiration close
            if self._should_close_due_to_assignment(position, date):
                progress_print(f"⏰ Position {position.__str__()} expired or near expiration")
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

            # Stop loss
            if self._should_close_due_to_stop(position, exit_price):
                progress_print(f"🛑 Stop loss hit for {position.__str__()} at exit {exit_price}")
                current_volumes = self.get_current_volumes_for_position(position, date)
                remove_position(date, position, exit_price if exit_price is not None else 0.0, current_volumes=current_volumes)
                continue

            # Holding period
            if self._should_close_due_to_holding(position, date, self.holding_period):
                if exit_price is not None and not has_error:
                    progress_print(f"📆 Holding period met for {position.__str__()} at exit {exit_price}")
                    current_volumes = self.get_current_volumes_for_position(position, date)
                    remove_position(date, position, exit_price, current_volumes=current_volumes)
                else:
                    progress_print(f"⚠️  No exit price available for {position.__str__()} on {date}. Skipping holding-period close.")

    def _compute_exit_price(self, date: datetime, position: Position) -> tuple[Optional[float], bool]:
        date_key = date.strftime('%Y-%m-%d')
        option_chain = self.options_data.get(date_key) if self.options_data else None
        has_error = False
        exit_price = None
        if option_chain is not None:
            try:
                exit_price = position.calculate_exit_price(option_chain)
            except Exception as e:
                progress_print(f"⚠️  Error calculating exit price: {e}")
        if exit_price is None:
            # Attempt to augment chain with missing contracts
            if option_chain is None:
                option_chain = OptionChain(calls=[], puts=[])
            for option in position.spread_options:
                try:
                    contract = self.options_handler.get_specific_option_contract(option.strike, option.expiration, option.option_type.value, date)
                except Exception as e:
                    progress_print(f"⚠️  Error fetching contract for {option.strike} {option.expiration} {option.option_type.value}: {e}")
                    contract = None
                if contract is None:
                    progress_print(f"⚠️  No contract found for {option.strike} {option.expiration} {option.option_type.value}")
                    has_error = True
                    continue
                option_chain = option_chain.add_option(contract)
            if not has_error:
                try:
                    exit_price = position.calculate_exit_price(option_chain)
                except Exception as e:
                    progress_print(f"⚠️  Error recalculating exit price: {e}")
                    has_error = True
        return exit_price, has_error

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

    def recommend_open_position(self, date: datetime, current_price: float) -> Optional[dict]:
        """
        Recommend opening a position for the given date and current price.
        
        Args:
            date: Current date
            current_price: Current underlying price
            
        Returns:
            dict or None: Position recommendation with required keys, None if no position should be opened
        """

        # Print price information for the last 10 days
        print(f"🔍 Checking for buy signal for {date.strftime('%Y-%m-%d')}")
        print(f"🔍 Current price: {current_price}")
        print(f"🔍 Last 10 days prices: {self.data['Close'].tail(10)}")
        
        # Check for buy signal using velocity momentum logic
        if not self._has_buy_signal(date):
            return None
            
        # For velocity momentum, we prefer put credit spreads on buy signals
        strategy_type = StrategyType.PUT_CREDIT_SPREAD
        confidence = 0.7  # Fixed confidence for rule-based strategy
        
        # Get option chain for the date
        chain = self._get_option_chain(date)
        if chain is None:
            return None
            
        # Select expiration (target ~1 week)
        expiration_str = self._select_week_expiration(date, chain)
        if not expiration_str:
            return None
            
        # Create the position using existing logic
        position = self._create_put_credit_spread(date, current_price, expiration_str)
        if position is None:
            return None
            
        # Extract legs and other details
        if not position.spread_options or len(position.spread_options) != 2:
            return None
            
        atm_option, otm_option = position.spread_options
        width = abs(atm_option.strike - otm_option.strike)
        
        return {
            "strategy_type": strategy_type,
            "legs": (atm_option, otm_option),
            "credit": position.entry_price,
            "width": width,
            "probability_of_profit": confidence,
            "confidence": confidence,
            "expiration_date": expiration_str,
        }

    def get_current_volumes_for_position(self, position: Position, date: datetime) -> list[int]:
        """
        Fetch current date volume data for all options in a position.
        """
        current_volumes = []
        for option in position.spread_options:
            try:
                fresh_option = self.options_handler.get_specific_option_contract(
                    option.strike,
                    option.expiration,
                    option.option_type.value,
                    date
                )
                if fresh_option and fresh_option.volume is not None:
                    current_volumes.append(fresh_option.volume)
                else:
                    current_volumes.append(None)
            except Exception as e:
                progress_print(f"⚠️  Error fetching volume data for {option.symbol}: {e}")
                current_volumes.append(None)
        return current_volumes
