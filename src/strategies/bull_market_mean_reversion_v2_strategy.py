from datetime import datetime
from typing import Callable, Optional
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import yfinance as yf

from src.backtest.models import Strategy, Position, StrategyType
from src.common.options_dtos import ExpirationRangeDTO, StrikeRangeDTO, StrikePrice
from src.common.progress_tracker import progress_print
from src.common.options_handler import OptionsHandler
from src.common.options_helpers import OptionsRetrieverHelper
from src.common.models import OptionType
from src.common.data_retriever import DataRetriever
from decimal import Decimal


class BullMarketMeanReversionV2Strategy(Strategy):
    """
    Bull Market SPY Mean Reversion v2 Strategy
    
    During a bull market, there are days where the price of the underlying is higher 
    than its historical average and will revert towards its historical value.
    
    Entry Criteria:
    - Upward trend: SMA15 > SMA30 and width between them is increasing
    - Z-Score of underlying price > 1.5
    - Put debit spread (buy higher strike put, sell lower strike put)
    - Width ≤ $6
    - 7-10 DTE
    
    Exit Criteria:
    - Stop loss hit
    - Profit take hit
    - Z-Score < 0.5
    - Z-Score decreased by > 0.7 from entry
    - 2 DTE
    
    Position Management:
    - Only 1 open position at a time
    - Size based on maximum risk per trade of 8%
    """

    def __init__(
        self, 
        options_handler: OptionsHandler, 
        start_date_offset: int = 60,
        stop_loss: float = None,
        profit_target: float = None,
        z_score_entry_threshold: float = 1.5,
        z_score_exit_threshold: float = 0.5,
        z_score_decrease_threshold: float = 0.7,
        max_risk_per_trade: float = 0.08,
        max_spread_width: float = 6.0
    ):
        super().__init__(start_date_offset=start_date_offset, stop_loss=stop_loss, profit_target=profit_target)
        
        self.options_handler = options_handler
        self.z_score_entry_threshold = z_score_entry_threshold
        self.z_score_exit_threshold = z_score_exit_threshold
        self.z_score_decrease_threshold = z_score_decrease_threshold
        self.max_risk_per_trade = max_risk_per_trade
        self.max_spread_width = max_spread_width
        
        # Track position entries for plotting
        self._position_entries = []
        # Track position exits for plotting
        self._position_exits = []
        # Track Z-Score at entry for each position
        self._position_entry_z_scores = {}
        # VIX data for overlay
        self.vix_data = None
        self.vix_z_score = None
    
    def set_data(self, data: pd.DataFrame, treasury_data: Optional = None):
        super().set_data(data, treasury_data)
        
        # Reset position entries and exits tracking for new backtest run
        self._position_entries = []
        self._position_exits = []
        self._position_entry_z_scores = {}
        
        # Fetch and calculate VIX Z-Score
        self._load_vix_data()
        
        # Pre-calculate moving averages and Z-Score for performance
        if self.data is not None and not self.data.empty:
            # Calculate SMA 15 and SMA 30
            self.data['SMA_15'] = self.data['Close'].rolling(window=15).mean()
            self.data['SMA_30'] = self.data['Close'].rolling(window=30).mean()
            
            # Calculate width between SMA15 and SMA30
            self.data['SMA_Width'] = self.data['SMA_15'] - self.data['SMA_30']
            
            # Calculate width change (to detect increasing width)
            self.data['SMA_Width_Change'] = self.data['SMA_Width'].diff()
            
            # Calculate Z-Score: (price - mean) / std
            # Use rolling window for mean and std (e.g., 60 days)
            window = 60
            self.data['Price_Mean'] = self.data['Close'].rolling(window=window).mean()
            self.data['Price_Std'] = self.data['Close'].rolling(window=window).std()
            self.data['Z_Score'] = (self.data['Close'] - self.data['Price_Mean']) / self.data['Price_Std']
    
    def _load_vix_data(self):
        """Load VIX data and calculate Z-Score for overlay on plot."""
        if self.data is None or self.data.empty:
            self.vix_data = None
            self.vix_z_score = None
            return
        
        try:
            # Get date range from SPY data (extend slightly to ensure we have enough data)
            start_date = self.data.index[0] - pd.Timedelta(days=90)  # Extra days for rolling window
            end_date = self.data.index[-1] + pd.Timedelta(days=1)
            
            # Fetch VIX data using yfinance
            vix_ticker = yf.Ticker('^VIX')
            vix_data = vix_ticker.history(start=start_date, end=end_date)
            
            if vix_data.empty:
                progress_print("⚠️  No VIX data available")
                self.vix_data = None
                self.vix_z_score = None
                return
            
            # Remove timezone from index if present
            if vix_data.index.tz is not None:
                vix_data.index = vix_data.index.tz_localize(None)
            
            # Align VIX data with SPY data dates using reindex with forward fill
            # This ensures we have VIX data for each SPY trading day
            aligned_vix = pd.DataFrame(index=self.data.index)
            aligned_vix['Close'] = vix_data['Close'].reindex(self.data.index, method='ffill')
            
            # Drop rows where we couldn't align VIX data
            aligned_vix = aligned_vix.dropna(subset=['Close'])
            
            if aligned_vix.empty:
                progress_print("⚠️  Could not align VIX data with SPY dates")
                self.vix_data = None
                self.vix_z_score = None
                return
            
            # Calculate VIX Z-Score using same 60-day window
            window = 60
            aligned_vix['VIX_Mean'] = aligned_vix['Close'].rolling(window=window).mean()
            aligned_vix['VIX_Std'] = aligned_vix['Close'].rolling(window=window).std()
            aligned_vix['VIX_Z_Score'] = (aligned_vix['Close'] - aligned_vix['VIX_Mean']) / aligned_vix['VIX_Std']
            
            self.vix_data = aligned_vix
            self.vix_z_score = aligned_vix['VIX_Z_Score']
            
            progress_print(f"✅ VIX data loaded and Z-Score calculated ({len(aligned_vix)} aligned dates)")
            
        except Exception as e:
            progress_print(f"⚠️  Error loading VIX data: {e}")
            import traceback
            traceback.print_exc()
            self.vix_data = None
            self.vix_z_score = None
    
    def _recalculate_indicators(self):
        """Recalculate moving averages, width, and Z-Score after data updates."""
        if self.data is not None and not self.data.empty:
            # Calculate SMA 15 and SMA 30
            self.data['SMA_15'] = self.data['Close'].rolling(window=15).mean()
            self.data['SMA_30'] = self.data['Close'].rolling(window=30).mean()
            
            # Calculate width between SMA15 and SMA30
            self.data['SMA_Width'] = self.data['SMA_15'] - self.data['SMA_30']
            
            # Calculate width change (to detect increasing width)
            self.data['SMA_Width_Change'] = self.data['SMA_Width'].diff()
            
            # Calculate Z-Score
            window = 60
            self.data['Price_Mean'] = self.data['Close'].rolling(window=window).mean()
            self.data['Price_Std'] = self.data['Close'].rolling(window=window).std()
            self.data['Z_Score'] = (self.data['Close'] - self.data['Price_Mean']) / self.data['Price_Std']
    
    def on_new_date(self, date: datetime, positions: tuple['Position', ...], add_position: Callable[['Position'], None], remove_position: Callable[['Position'], None]):
        super().on_new_date(date, positions)
        
        # Only 1 open position at a time
        if len(positions) == 0:
            self._try_open_position(date, add_position)
        else:
            # Close existing positions first if exit criteria are met
            self._try_close_positions(date, positions, remove_position)
    
    def on_end(self, positions: tuple['Position', ...], remove_position: Callable[['Position'], None], date: datetime):
        """
        Create a plot showing SPY price over time with position entry indicators and Z-Score.
        """
        if self.data is None or self.data.empty:
            progress_print("⚠️  No data available for plotting")
            return
        
        try:
            # Create the plot with two subplots
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10), sharex=True)
            
            # Plot SPY price on first subplot
            ax1.plot(self.data.index, self.data['Close'], 
                   label='SPY Close Price', color='blue', alpha=0.7, linewidth=1)
            
            # Plot moving averages
            if 'SMA_15' in self.data.columns:
                ax1.plot(self.data.index, self.data['SMA_15'], 
                       label='SMA 15', color='orange', alpha=0.6, linewidth=1)
            
            if 'SMA_30' in self.data.columns:
                ax1.plot(self.data.index, self.data['SMA_30'], 
                       label='SMA 30', color='green', alpha=0.6, linewidth=1)
            
            # Plot position entry indicators
            if hasattr(self, '_position_entries'):
                entry_dates = self._position_entries
            else:
                entry_dates = []
            
            if entry_dates:
                for entry_date in entry_dates:
                    if entry_date in self.data.index:
                        entry_price = self.data.loc[entry_date, 'Close']
                        ax1.scatter(entry_date, entry_price, 
                                 color='red', s=100, marker='^', 
                                 label='Position Entry' if entry_date == entry_dates[0] else "", 
                                 zorder=5, alpha=0.8)
            
            # Plot position exit indicators
            if hasattr(self, '_position_exits'):
                exit_dates = self._position_exits
            else:
                exit_dates = []
            
            if exit_dates:
                for exit_date in exit_dates:
                    if exit_date in self.data.index:
                        exit_price = self.data.loc[exit_date, 'Close']
                        ax1.scatter(exit_date, exit_price, 
                                 color='blue', s=100, marker='o', 
                                 label='Position Exit' if exit_date == exit_dates[0] else "", 
                                 zorder=5, alpha=0.8)
            
            # Format first subplot
            num_positions = len(entry_dates) if entry_dates else 0
            title = f'SPY Price with Position Entries - Bull Market Mean Reversion v2\nTotal Positions: {num_positions}'
            ax1.set_title(title, fontsize=14, fontweight='bold')
            ax1.set_ylabel('SPY Price ($)', fontsize=12)
            ax1.legend(loc='upper left')
            ax1.grid(True, alpha=0.3)
            
            # Plot Z-Score on second subplot
            if 'Z_Score' in self.data.columns:
                ax2.plot(self.data.index, self.data['Z_Score'], 
                        label='SPY Z-Score', color='purple', alpha=0.7, linewidth=1)
                ax2.axhline(y=self.z_score_entry_threshold, color='green', linestyle='--', 
                           label=f'Entry Threshold ({self.z_score_entry_threshold})', alpha=0.5)
                ax2.axhline(y=self.z_score_exit_threshold, color='red', linestyle='--', 
                           label=f'Exit Threshold ({self.z_score_exit_threshold})', alpha=0.5)
                ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)
            
            # Plot VIX Z-Score overlay
            if self.vix_z_score is not None and not self.vix_z_score.empty:
                # Filter out NaN values for plotting
                valid_vix = self.vix_z_score.dropna()
                if not valid_vix.empty:
                    ax2.plot(valid_vix.index, valid_vix.values, 
                            label='VIX Z-Score', color='orange', alpha=0.9, linewidth=1, linestyle='-')
            
            ax2.set_xlabel('Date', fontsize=12)
            ax2.set_ylabel('Z-Score', fontsize=12)
            ax2.legend(loc='upper left')
            ax2.grid(True, alpha=0.3)
            
            # Add text box with strategy info
            strategy_info = f'Strategy: Bull Market Mean Reversion v2\nZ-Score Entry: >{self.z_score_entry_threshold}\nZ-Score Exit: <{self.z_score_exit_threshold}'
            ax1.text(0.02, 0.98, strategy_info, transform=ax1.transAxes, fontsize=10,
                   verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
            
            # Format x-axis dates
            ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
            plt.xticks(rotation=45)
            
            # Adjust layout
            plt.tight_layout()
            
            # Show the plot
            plt.show()
            
            progress_print("📊 Position entry plot generated successfully")
            
        except Exception as e:
            progress_print(f"⚠️  Error creating plot: {e}")
            import traceback
            traceback.print_exc()
    
    def _has_entry_signal(self, date: datetime) -> bool:
        """
        Check for entry signal:
        1. Upward trend: SMA15 > SMA30 and width between them is increasing
        2. Z-Score > entry threshold
        """
        if self.data is None or self.data.empty:
            return False
        
        # Get the current date index in the data
        current_date = datetime.now().date()
        is_current_date = date.date() == current_date
        
        # Handle current date with live price
        if is_current_date:
            live_price = self._get_current_underlying_price(date)
            if live_price is not None:
                if date in self.data.index:
                    self.data.loc[date, 'Close'] = live_price
                    self.data.loc[date, 'Open'] = live_price
                    self.data.loc[date, 'High'] = live_price
                    self.data.loc[date, 'Low'] = live_price
                    self.data.loc[date, 'Volume'] = 0
                    progress_print(f"🔄 Updated current date {date.date()} with live price ${live_price:.2f}")
                else:
                    new_row = pd.DataFrame({
                        'Close': [live_price],
                        'Open': [live_price],
                        'High': [live_price],
                        'Low': [live_price],
                        'Volume': [0]
                    }, index=[date])
                    self.data = pd.concat([self.data, new_row])
                    progress_print(f"📅 Fetched live price ${live_price:.2f} for current date {date.date()}")
                
                self._recalculate_indicators()
                current_idx = self.data.index.get_loc(date)
            else:
                if date in self.data.index:
                    current_idx = self.data.index.get_loc(date)
                    progress_print(f"⚠️ Could not fetch live price for {date.date()}, using cached data")
                else:
                    current_idx = len(self.data) - 1
                    progress_print(f"⚠️ Could not fetch live price for {date.date()}, using last available data point")
        else:
            try:
                current_idx = self.data.index.get_loc(date)
            except KeyError:
                progress_print("⚠️  Date not found in data")
                return False
        
        # Check if we have enough data to analyze (need at least 60 days for Z-Score)
        if current_idx < 60:
            progress_print("⚠️  Not enough data to analyze (need at least 60 days)")
            return False
        
        # Check if required columns exist
        required_cols = ['SMA_15', 'SMA_30', 'SMA_Width', 'SMA_Width_Change', 'Z_Score']
        for col in required_cols:
            if col not in self.data.columns:
                progress_print(f"⚠️  Missing required column: {col}")
                return False
        
        # Check for upward trend: SMA15 > SMA30
        sma15 = self.data['SMA_15'].iloc[current_idx]
        sma30 = self.data['SMA_30'].iloc[current_idx]
        
        if pd.isna(sma15) or pd.isna(sma30) or sma15 <= sma30:
            progress_print(f"⚠️  No upward trend: SMA15={sma15:.2f}, SMA30={sma30:.2f}")
            return False
        
        # Check if width is increasing (comparing to previous day)
        if current_idx < 1:
            progress_print("⚠️  Not enough data to check width change")
            return False
        
        width_change = self.data['SMA_Width_Change'].iloc[current_idx]
        if pd.isna(width_change) or width_change <= 0:
            progress_print(f"⚠️  Width not increasing: change={width_change:.4f}")
            return False
        
        # Check Z-Score > entry threshold
        z_score = self.data['Z_Score'].iloc[current_idx]
        if pd.isna(z_score) or z_score <= self.z_score_entry_threshold:
            progress_print(f"⚠️  Z-Score not above threshold: Z-Score={z_score:.2f}, threshold={self.z_score_entry_threshold}")
            return False
        
        progress_print(f"✅ Entry signal detected: SMA15={sma15:.2f} > SMA30={sma30:.2f}, width increasing, Z-Score={z_score:.2f}")
        return True
    
    def _get_current_underlying_price(self, date: datetime) -> Optional[float]:
        """
        Fetch and return the live price if the date is the current date, otherwise return last_price for the date
        """
        current_date = datetime.now().date()
        if date.date() == current_date:
            # Initialize DataRetriever when needed
            symbol = None
            
            if hasattr(self, 'options_handler') and hasattr(self.options_handler, 'symbol'):
                symbol = self.options_handler.symbol
            
            if symbol is None:
                raise ValueError("Symbol not found in options handler.")
            
            try:
                data_retriever = DataRetriever(symbol=symbol, use_free_tier=True, quiet_mode=True)
                live_price = data_retriever.get_live_price()
            except Exception as e:
                raise ValueError(f"Failed to fetch live price from DataRetriever: {e}")
            
            if live_price is not None:
                return live_price
            else:
                raise ValueError("Failed to fetch live price from DataRetriever.")
        else:
            return float(self.data.loc[date]['Close'])
    
    def _get_z_score(self, date: datetime) -> Optional[float]:
        """Get Z-Score for a given date."""
        if self.data is None or self.data.empty:
            return None
        
        try:
            current_idx = self.data.index.get_loc(date)
            if 'Z_Score' in self.data.columns:
                z_score = self.data['Z_Score'].iloc[current_idx]
                return float(z_score) if not pd.isna(z_score) else None
        except (KeyError, IndexError):
            pass
        
        return None
    
    def _create_put_debit_spread(self, date: datetime, current_price: float, expiration: str) -> Optional[Position]:
        """
        Create a put debit spread (buy higher strike put, sell lower strike put).
        Width must be ≤ max_spread_width (default $6).
        
        Args:
            date: Current date
            current_price: Current underlying price
            expiration: Target expiration date string (YYYY-MM-DD)
            
        Returns:
            Position: Put debit spread position, or None if creation fails
        """
        try:
            # Get list of contracts for the date
            expiration_range = ExpirationRangeDTO(min_days=7, max_days=10)
            
            # Strike range: need strikes around current price for put debit spread
            # For put debit spread: buy higher strike (closer to ATM), sell lower strike (OTM)
            strike_range = StrikeRangeDTO(
                min_strike=StrikePrice(Decimal(str(current_price - self.max_spread_width - 2))),
                max_strike=StrikePrice(Decimal(str(current_price + 2)))
            )
            
            contracts = self.options_handler.get_contract_list_for_date(
                date, 
                strike_range=strike_range, 
                expiration_range=expiration_range
            )
            
            if not contracts:
                progress_print("⚠️  No contracts found for the date")
                return None
            
            # Filter for contracts with the specific expiration
            contracts_for_expiration = [
                c for c in contracts 
                if str(c.expiration_date) == expiration
            ]
            
            if not contracts_for_expiration:
                progress_print(f"⚠️  No contracts found for target expiration {expiration}")
                return None
            
            progress_print(f"✅ Found {len(contracts_for_expiration)} contracts for expiration {expiration}")
            
            # Find ATM put option
            atm_put = OptionsRetrieverHelper.find_atm_contracts(contracts_for_expiration, current_price)[1]
            
            if not atm_put:
                progress_print(f"⚠️  No ATM put found for expiration {expiration}")
                return None
            
            progress_print(f"Found ATM put: {atm_put.ticker} @ ${atm_put.strike_price.value} exp {atm_put.expiration_date}")
            
            # For put debit spread: buy higher strike (ATM or slightly ITM), sell lower strike (OTM)
            # Higher strike = ATM or slightly above
            higher_strike = round(current_price)
            
            # Lower strike = higher_strike - width (≤ max_spread_width)
            # Try to find a strike that gives us width ≤ max_spread_width
            target_lower_strike = higher_strike - self.max_spread_width
            
            # Filter for puts only
            puts_for_expiration = [
                c for c in contracts_for_expiration 
                if c.contract_type == OptionType.PUT
            ]
            
            if not puts_for_expiration:
                progress_print(f"⚠️  No put contracts found for expiration {expiration}")
                return None
            
            # Find the put with strike closest to higher_strike (ATM)
            higher_put = min(
                puts_for_expiration,
                key=lambda put: abs(float(put.strike_price.value) - higher_strike)
            )
            
            # Find the put with strike closest to target_lower_strike
            lower_put = min(
                puts_for_expiration,
                key=lambda put: abs(float(put.strike_price.value) - target_lower_strike)
            )
            
            # Verify width ≤ max_spread_width
            actual_width = abs(float(higher_put.strike_price.value) - float(lower_put.strike_price.value))
            if actual_width > self.max_spread_width:
                progress_print(f"⚠️  Spread width {actual_width:.2f} exceeds max {self.max_spread_width}")
                return None
            
            # Verify both legs have the same expiration
            if str(higher_put.expiration_date) != str(lower_put.expiration_date):
                progress_print(f"❌ ERROR: Expiration mismatch! Higher: {higher_put.expiration_date}, Lower: {lower_put.expiration_date}")
                return None
            
            progress_print(f"✅ Verified: Both legs have same expiration {expiration} (vertical spread)")
            progress_print(f"Higher strike put: {higher_put.ticker} @ ${higher_put.strike_price.value}")
            progress_print(f"Lower strike put: {lower_put.ticker} @ ${lower_put.strike_price.value}")
            progress_print(f"Spread width: ${actual_width:.2f}")
            
            # Get bar data to calculate net debit
            higher_bar = self.options_handler.get_option_bar(higher_put, date)
            lower_bar = self.options_handler.get_option_bar(lower_put, date)
            
            if not higher_bar or not lower_bar:
                progress_print("⚠️  No bar data available for debit calculation")
                return None
            
            # Calculate net debit (buy higher strike, sell lower strike)
            # Net debit = price of higher strike put - price of lower strike put
            net_debit = float(higher_bar.close_price) - float(lower_bar.close_price)
            
            if net_debit <= 0:
                progress_print(f"⚠️  Invalid debit: {net_debit:.2f} (should be positive for debit spread)")
                return None
            
            # Convert OptionContractDTO to Option using the conversion method
            from src.common.models import Option
            higher_option = Option.from_contract_and_bar(higher_put, higher_bar)
            lower_option = Option.from_contract_and_bar(lower_put, lower_bar)
            
            # Create position (put debit spread)
            # Put debit spread: Buy higher strike put, Sell lower strike put
            # This is a BEARISH position - profits when price goes down (mean reversion from high prices)
            # Note: Position class uses credit spread formula: Return = entry_price - exit_price
            # For debit spread to work correctly, we store:
            # entry_price = -net_debit (negative) so that Return = entry_price - exit_price = -net_debit - exit_price
            # exit_price = -net_credit_to_close (negative) so that Return = -net_debit - (-net_credit) = net_credit - net_debit
            # This gives us the correct return: (H' - L') - (H - L)
            position = Position(
                symbol=self.data.index.name if self.data.index.name else 'SPY',
                expiration_date=datetime.strptime(expiration, '%Y-%m-%d'),
                strategy_type=StrategyType.PUT_DEBIT_SPREAD,  # Put debit spread for bearish mean reversion
                strike_price=higher_strike,
                entry_date=date,
                entry_price=-net_debit,  # Store as negative so Position calculation works for debit spread
                spread_options=[higher_option, lower_option]  # [higher strike (buy), lower strike (sell)]
            )
            
            # Set quantity for position (will be calculated by backtest engine based on risk)
            position.set_quantity(1)
            return position
            
        except Exception as e:
            progress_print(f"⚠️  Error creating put debit spread: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _select_expiration(self, date: datetime) -> Optional[str]:
        """
        Select the best expiration date for the strategy.
        Target: 7-10 days out.
        """
        progress_print(f"🔍 _select_expiration called for {date.strftime('%Y-%m-%d')}")
        target_days = 8  # Middle of 7-10 range
        
        def days_out(exp_str: str) -> int:
            try:
                exp_dt = datetime.strptime(exp_str, '%Y-%m-%d')
                return (exp_dt - date).days
            except Exception:
                return -9999
        
        try:
            progress_print("🔍 Fetching expirations from options_handler for 7-10 day window...")
            
            expiration_range = ExpirationRangeDTO(min_days=7, max_days=10)
            contracts = self.options_handler.get_contract_list_for_date(date, expiration_range=expiration_range)
            
            if not contracts:
                progress_print("⚠️  No contracts found for the date")
                return None
            
            # Extract unique expiration dates from contracts
            expirations = set(str(contract.expiration_date) for contract in contracts)
            progress_print(f"🔍 Found {len(expirations)} expirations from contracts")
            
            if not expirations:
                progress_print("⚠️  No expirations found in option chain")
                return None
            
            # Calculate days out for each expiration and select closest to target (8 days)
            valid_expirations = [(e, days_out(e)) for e in expirations]
            valid_expirations = [(e, d) for e, d in valid_expirations if d > 0]
            
            if not valid_expirations:
                progress_print("⚠️  No future expirations available")
                return None
            
            # Select the expiration closest to target (8 days)
            best_expiration = min(valid_expirations, key=lambda x: abs(x[1] - target_days))[0]
            days_to_exp = days_out(best_expiration)
            progress_print(f"✅ Selected expiration: {best_expiration} ({days_to_exp} days out)")
            
            return best_expiration
            
        except Exception as e:
            progress_print(f"❌ Error fetching expirations from options_handler: {str(e)}")
            return None
    
    def _try_open_position(self, date: datetime, add_position: Callable[['Position'], None]):
        """Try to open a new position if entry signal is present."""
        if not self._has_entry_signal(date):
            return
        
        progress_print(f"📈 Entry signal detected for {date.strftime('%Y-%m-%d')}")
        
        current_price = self._get_current_underlying_price(date)
        if current_price is None:
            progress_print("⚠️  Failed to get current price.")
            return
        
        # Select expiration (target 7-10 days)
        expiration_str = self._select_expiration(date)
        if not expiration_str:
            progress_print("⚠️  Failed to select expiration")
            return
        
        position = self._create_put_debit_spread(date, current_price, expiration_str)
        if position is None:
            progress_print("⚠️  Failed to create put debit spread for selected expiration")
            return
        
        progress_print(f"Current underlying price: {current_price:.2f}")
        
        # Track position entry for plotting
        self._position_entries.append(date)
        
        # Track Z-Score at entry
        entry_z_score = self._get_z_score(date)
        if entry_z_score is not None:
            # Use position ID as key (symbol + entry_date + strike)
            position_id = f"{position.symbol}_{date.strftime('%Y-%m-%d')}_{position.strike_price}"
            self._position_entry_z_scores[position_id] = entry_z_score
            progress_print(f"📊 Entry Z-Score: {entry_z_score:.2f}")
        
        add_position(position)
    
    def _try_close_positions(self, date: datetime, positions: tuple['Position', ...], remove_position: Callable[['Position'], None]):
        """Try to close positions if exit criteria are met."""
        current_underlying_price = self._get_current_underlying_price(date)
        print(f"Current underlying price: {current_underlying_price:.2f}")
        progress_print(f"🤖 Strategy evaluating {len(positions)} open position(s) for potential closure...")
        
        for position in positions:
            days_held = position.get_days_held(date) if hasattr(position, 'get_days_held') else 0
            days_to_exp = position.get_days_to_expiration(date) if hasattr(position, 'get_days_to_expiration') else 0
            progress_print(f"🔍 Position {position.__str__()} - Days held: {days_held}, Days to exp: {days_to_exp}")
            
            # Check exit criteria
            
            # 1. Close at 2 DTE
            if days_to_exp <= 2:
                progress_print(f"⏰ Position {position.__str__()} at 2 DTE or less (days to exp: {days_to_exp})")
                exit_price, has_error = self._compute_exit_price(date, position)
                if not has_error and exit_price is not None:
                    exit_price = self._sanitize_exit_price(exit_price)
                    current_volumes = self.get_current_volumes_for_position(position, date)
                    # Track exit for plotting
                    self._position_exits.append(date)
                    remove_position(date, position, exit_price, underlying_price=current_underlying_price, current_volumes=current_volumes)
                else:
                    progress_print(f"⚠️  Could not compute exit price for 2 DTE closure, using 0.0")
                    current_volumes = self.get_current_volumes_for_position(position, date)
                    # Track exit for plotting
                    self._position_exits.append(date)
                    remove_position(date, position, 0.0, underlying_price=current_underlying_price, current_volumes=current_volumes)
                continue
            
            # 2. Check Z-Score exit conditions
            current_z_score = self._get_z_score(date)
            if current_z_score is not None:
                # Get entry Z-Score
                position_id = f"{position.symbol}_{position.entry_date.strftime('%Y-%m-%d')}_{position.strike_price}"
                entry_z_score = self._position_entry_z_scores.get(position_id)
                
                if entry_z_score is not None:
                    # Check if Z-Score < exit threshold
                    if current_z_score < self.z_score_exit_threshold:
                        progress_print(f"📉 Z-Score exit: current={current_z_score:.2f} < threshold={self.z_score_exit_threshold}")
                        exit_price, has_error = self._compute_exit_price(date, position)
                        if not has_error and exit_price is not None:
                            exit_price = self._sanitize_exit_price(exit_price)
                            current_volumes = self.get_current_volumes_for_position(position, date)
                            # Track exit for plotting
                            self._position_exits.append(date)
                            remove_position(date, position, exit_price, current_volumes=current_volumes)
                            continue
                    
                    # Check if Z-Score decreased by > threshold from entry
                    z_score_decrease = entry_z_score - current_z_score
                    if z_score_decrease > self.z_score_decrease_threshold:
                        progress_print(f"📉 Z-Score decrease exit: decrease={z_score_decrease:.2f} > threshold={self.z_score_decrease_threshold}")
                        exit_price, has_error = self._compute_exit_price(date, position)
                        if not has_error and exit_price is not None:
                            exit_price = self._sanitize_exit_price(exit_price)
                            current_volumes = self.get_current_volumes_for_position(position, date)
                            # Track exit for plotting
                            self._position_exits.append(date)
                            remove_position(date, position, exit_price, current_volumes=current_volumes)
                            continue
            
            # 3. Compute exit price for stop loss and profit target
            exit_price, has_error = self._compute_exit_price(date, position)
            if not has_error and exit_price is not None:
                exit_price = self._sanitize_exit_price(exit_price)
                progress_print(f"💰 Calculated exit price for {position.__str__()}: {exit_price}")
                
                # Stop loss
                if self._should_close_due_to_stop(position, exit_price):
                    progress_print(f"🛑 Stop loss hit for {position.__str__()} at exit {exit_price}")
                    current_volumes = self.get_current_volumes_for_position(position, date)
                    # Track exit for plotting
                    self._position_exits.append(date)
                    remove_position(date, position, exit_price, current_volumes=current_volumes)
                    continue
                
                # Profit target
                if self._should_close_due_to_profit(position, exit_price):
                    progress_print(f"🎯 Profit target hit for {position.__str__()} at exit {exit_price}")
                    current_volumes = self.get_current_volumes_for_position(position, date)
                    # Track exit for plotting
                    self._position_exits.append(date)
                    remove_position(date, position, exit_price, current_volumes=current_volumes)
                    continue
            
            # Position not closed - show why
            progress_print(f"📋 Position {position.__str__()} remains open - Days held: {days_held}, Days to exp: {days_to_exp}")
        
        # Summary
        progress_print(f"✅ Strategy evaluation complete for {date.strftime('%Y-%m-%d')}")
    
    def _compute_exit_price(self, date: datetime, position: Position) -> tuple[Optional[float], bool]:
        """Compute exit price using options_handler.get_option_bar and calculate_exit_price_from_bars"""
        try:
            if not position.spread_options or len(position.spread_options) != 2:
                progress_print("⚠️  Position doesn't have valid spread options")
                return None, True
            
            higher_option, lower_option = position.spread_options
            progress_print(f"🔍 Attempting to get bar data for {date.strftime('%Y-%m-%d')} - Higher: {higher_option.ticker}, Lower: {lower_option.ticker}")
            
            # Get bar data for both options
            higher_bar = self.options_handler.get_option_bar(higher_option, date)
            lower_bar = self.options_handler.get_option_bar(lower_option, date)
            
            progress_print(f"🔍 Bar data results - Higher bar: {higher_bar is not None}, Lower bar: {lower_bar is not None}")
            
            if not higher_bar or not lower_bar:
                progress_print(f"⚠️  No bar data available for options on {date.strftime('%Y-%m-%d')}")
                return None, True
            
            # For put debit spread: 
            # Entry: Buy higher strike put at H, Sell lower strike put at L
            # Net debit paid = H - L (stored as entry_price = -(H - L), negative)
            # Exit: Sell higher strike put at H', Buy back lower strike put at L'
            # Net credit received = H' - L'
            # 
            # Position class uses credit spread formula: Return = entry_price - exit_price
            # With entry_price = -(H - L) and exit_price = -(H' - L'):
            # Return = -(H - L) - (-(H' - L')) = -(H - L) + (H' - L') = (H' - L') - (H - L) ✓
            net_credit_to_close = float(higher_bar.close_price) - float(lower_bar.close_price)
            # Store as negative so Position calculation works for debit spread
            exit_price = -net_credit_to_close
            progress_print(f"💰 Calculated exit price: {exit_price} (net credit to close: {net_credit_to_close})")
            return exit_price, False
            
        except Exception as e:
            progress_print(f"⚠️  Error calculating exit price: {e}")
            import traceback
            traceback.print_exc()
            return None, True
    
    def _sanitize_exit_price(self, value: Optional[float]) -> Optional[float]:
        """Sanitize exit price value. For debit spreads, exit_price can be negative."""
        if value is None:
            return None
        return round(value, 2)
    
    def _should_close_due_to_stop(self, position: Position, exit_price: Optional[float]) -> bool:
        """Check if position should close due to stop loss."""
        return (exit_price is not None) and self._stop_loss_hit(position, exit_price)
    
    def _should_close_due_to_profit(self, position: Position, exit_price: Optional[float]) -> bool:
        """Check if position should close due to profit target."""
        return (exit_price is not None) and self._profit_target_hit(position, exit_price)
    
    def get_current_volumes_for_position(self, position: Position, date: datetime) -> list[int]:
        """
        Fetch current date volume data for all options in a position using options_handler.
        """
        current_volumes = []
        for option in position.spread_options:
            try:
                bar_data = self.options_handler.get_option_bar(option, date)
                
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
        Validate the data for the Bull Market Mean Reversion v2 Strategy.
        
        This strategy requires:
        - Basic OHLCV data
        - Moving averages (SMA_15, SMA_30) for trend detection
        - Z-Score calculation (60-day rolling window)
        
        Args:
            data: DataFrame with market data and features
            
        Returns:
            bool: True if data is valid for this strategy, False otherwise
        """
        progress_print(f"\n🔍 Validating data for Bull Market Mean Reversion v2 Strategy...")
        progress_print(f"   Data shape: {data.shape}")
        
        # Check if the data has the required columns
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
        
        # Check if we have enough data for Z-Score calculation (need at least 60 days)
        if len(data) < 60:
            progress_print(f"⚠️  Warning: Not enough data for Z-Score analysis. Need at least 60 days, got {len(data)}")
            return False
        
        # Check for gaps in the data (missing trading days)
        if len(data) > 1:
            date_range = pd.bdate_range(start=data.index.min(), end=data.index.max())
            expected_business_days = len(date_range)
            actual_trading_days = len(data)
            if actual_trading_days < expected_business_days * 0.9:  # Allow for some holidays
                progress_print(f"⚠️  Warning: Data may have gaps. Expected ~{expected_business_days} business days, got {actual_trading_days}")
        
        progress_print(f"✅ Data validation complete for Bull Market Mean Reversion v2 Strategy")
        progress_print(f"   Final data shape: {data.shape}")
        progress_print(f"   Date range: {data.index.min()} to {data.index.max()}")
        progress_print(f"   Trading days: {len(data)}")
        
        return True

