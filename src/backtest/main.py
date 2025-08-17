import os
import pandas as pd
from datetime import datetime
from typing import List
import argparse

from src.model.options_handler import OptionsHandler
from .models import Benchmark, Strategy, Position, StrategyType
from src.common.data_retriever import DataRetriever
from src.common.functions import load_hmm_model, load_lstm_model
from .config import VolumeConfig, VolumeStats, OverallPerformanceStats, StrategyPerformanceStats
from src.common.progress_tracker import ProgressTracker, set_global_progress_tracker, progress_print
from .strategy_builder import StrategyFactory, create_strategy_from_args

class BacktestEngine:
    """
    BacktestEngine is a class that runs a backtest on a given dataset and model.
    """

    def __init__(self, data: pd.DataFrame, strategy: Strategy, initial_capital: float = 100000,
                 start_date: datetime = datetime.now(),
                 end_date: datetime = datetime.now(),
                 max_position_size: float = None,
                 volume_config: VolumeConfig = None,
                 enable_progress_tracking: bool = True,
                 quiet_mode: bool = True):
        self.data = data
        self.strategy = strategy
        self.capital = initial_capital
        self.initial_capital = initial_capital  # Store initial capital for reporting
        self.start_date = start_date
        self.end_date = end_date
        self.positions = []
        self.total_positions = 0
        self.benchmark = Benchmark(initial_capital)
        self.max_position_size = max_position_size
        self.daily_returns = []  # Track daily returns for Sharpe Ratio calculation
        self.previous_capital = initial_capital  # Track previous day's capital
        
        # Volume validation configuration and statistics
        self.volume_config = volume_config or VolumeConfig(min_volume=10)
        self.volume_stats = VolumeStats(options_checked=0, positions_rejected_volume=0, positions_rejected_closure_volume=0, skipped_closures=0)
        
        # Progress tracking
        self.enable_progress_tracking = enable_progress_tracking
        self.quiet_mode = quiet_mode
        self.progress_tracker = None
        
        # Position tracking for statistics
        self.closed_positions = []

    def run(self) -> bool:
        """
        Run the backtest.
        """
        
        # Validate the data first
        if not self._validate_data(self.data):
            print("❌ Backtest aborted due to invalid data")
            return False

        # Use only dates that exist in the data (not pd.bdate_range which includes holidays)
        # Filter data to the specified date range and use the actual dates
        date_range = self.data.index

        self.benchmark.set_start_price(self.data.iloc[self.strategy.start_date_offset]['Close'])
        
        # Initialize progress tracker if enabled
        if self.enable_progress_tracking:
            # Account for start_date_offset in progress tracking
            effective_start_date = date_range[self.strategy.start_date_offset] if self.strategy.start_date_offset < len(date_range) else date_range[0]
            effective_total_dates = len(date_range) - self.strategy.start_date_offset
            
            self.progress_tracker = ProgressTracker(
                start_date=effective_start_date,
                end_date=date_range[-1],
                total_dates=effective_total_dates,
                desc="Running Backtest",
                quiet_mode=self.quiet_mode
            )
            set_global_progress_tracker(self.progress_tracker)
            
        print(f"📅 Running backtest on {len(date_range)} trading days")
        print(f"   Date range: {date_range[0].date()} to {date_range[-1].date()}")

        # For each date in the range, simulate the strategy
        for i, date in enumerate(date_range):
            # Convert to tuple for immutability
            positions_tuple = tuple(self.positions)

            # Update progress tracker only for dates that are actually being processed
            if self.progress_tracker and i >= self.strategy.start_date_offset:
                self.progress_tracker.update(current_date=date)

            try:
                self.strategy.on_new_date(date, positions_tuple, self._add_position, self._remove_position)
            except Exception as e:
                error_msg = f"Error in on_new_date: {e}"
                if self.progress_tracker:
                    progress_print(error_msg, force=True)
                else:
                    print(error_msg)
                return False

        self._end()

        return True

    def _end(self):
        """
        On end, execute strategy and close any remaining positions.
        """
        if self.progress_tracker:
            progress_print(f"\n🏁 Closing backtest - {len(self.positions)} positions remaining", force=True)
        else:
            print(f"\n🏁 Closing backtest - {len(self.positions)} positions remaining")

        # Get the last available price from the data
        last_date = self.data.index[-1]
        last_price = self.data.loc[last_date, 'Close']

        self.benchmark.set_end_price(last_price)

        if self.progress_tracker:
            progress_print(f"   Last trading date: {last_date.date()}", force=True)
            progress_print(f"   Last closing price: ${last_price:.2f}", force=True)
        else:
            print(f"   Last trading date: {last_date.date()}") 
            print(f"   Last closing price: ${last_price:.2f}")

        # Create a wrapper function that handles the new _remove_position signature
        def remove_position_wrapper(date: datetime, position: Position, exit_price: float, current_volumes: list[int] = None):
            self._remove_position(date, position, exit_price, current_volumes=current_volumes)

        # Execute strategy's on_end method with the wrapper
        self.strategy.on_end(self.positions, remove_position_wrapper, last_date)

        # Calculate final performance metrics
        initial_capital = self.initial_capital  # Use the initial capital from the constructor
        final_return = self.capital - initial_capital
        final_return_pct = (final_return / initial_capital) * 100
        
        # Calculate Sharpe Ratio
        sharpe_ratio = self._calculate_sharpe_ratio()
        
        # Close progress tracker if it exists
        if self.progress_tracker:
            self.progress_tracker.close()
            set_global_progress_tracker(None)
        
        # Volume validation statistics
        if self.volume_config.enable_volume_validation:
            volume_summary = self.volume_stats.get_summary()
            print(f"\n📈 Volume Validation Statistics:")
            print(f"   Options checked: {volume_summary['options_checked']}")
            print(f"   Position opens rejected due to volume: {volume_summary['positions_rejected_volume']}")
            print(f"   Position closures rejected due to volume: {volume_summary['positions_rejected_closure_volume']}")
            print(f"   Skipped closures: {volume_summary['skipped_closures']}")
            print(f"   Total rejections: {volume_summary['total_rejections']}")
            print(f"   Volume rejection rate: {volume_summary['rejection_rate']:.1f}%")
        
        # Position performance statistics
        if self.closed_positions:
            self._print_position_statistics()

        print("\n📊 Backtest Results Summary:")
        print(f"   Benchmark return: {self.benchmark.get_return_percentage():+.2f}%")
        print(f"   Benchmark return dollars: ${self.benchmark.get_return_dollars():+.2f}\n")
        print(f"   Trading Days: {len(self.data.index)}")
        print(f"   Total positions: {self.total_positions}")
        print(f"   Final capital: ${self.capital:.2f}")
        print(f"   Total Return: ${final_return:+,.2f} ({final_return_pct:+.2f}%)")
        print(f"   Sharpe Ratio: {sharpe_ratio:.3f}")

    def _validate_data(self, data: pd.DataFrame) -> bool:
        """
        Validate the data and filter it to the specified date range.
        
        Args:
            data: DataFrame with market data and features
            
        Returns:
            bool: True if data is valid, False otherwise
        """
        if self.progress_tracker:
            progress_print(f"\n🔍 Validating data for backtest...", force=True)
            progress_print(f"   Original data shape: {data.shape}", force=True)
            progress_print(f"   Date range: {self.start_date} to {self.end_date}", force=True)
        else:
            print(f"\n🔍 Validating data for backtest...")
            print(f"   Original data shape: {data.shape}")
            print(f"   Date range: {self.start_date} to {self.end_date}")
        
        # Check if the data has the required columns
        required_columns = [
            'Open', 'High', 'Low', 'Close', 'Volume',  # Basic OHLCV data
            'Returns', 'Log_Returns', 'Volatility',     # Basic technical features
            'RSI', 'MACD_Hist', 'Volume_Ratio',         # Technical indicators
            'Market_State',                             # HMM market state
            'Put_Call_Ratio', 'Option_Volume_Ratio',    # Options features
            'Days_Until_Next_CPI', 'Days_Since_Last_CPI',  # Calendar features
            'Days_Until_Next_CC', 'Days_Since_Last_CC',
            'Days_Until_Next_FFR', 'Days_Since_Last_FFR'
        ]
        
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            if self.progress_tracker:
                progress_print(f"⚠️  Warning: Missing columns: {missing_columns}", force=True)
                progress_print(f"   Available columns: {list(data.columns)}", force=True)
            else:
                print(f"⚠️  Warning: Missing columns: {missing_columns}")
                print(f"   Available columns: {list(data.columns)}")
            return False
        else:
            if self.progress_tracker:
                progress_print(f"✅ All required columns present", force=True)
            else:
                print(f"✅ All required columns present")
        
        # Check if data has datetime index
        if not isinstance(data.index, pd.DatetimeIndex):
            error_msg = "❌ Error: Data must have a datetime index for backtesting"
            if self.progress_tracker:
                progress_print(error_msg, force=True)
            else:
                print(error_msg)
            return False
        
        # Filter data to the specified date range
        if self.progress_tracker:
            progress_print(f"   Data index range: {data.index.min()} to {data.index.max()}", force=True)
        else:
            print(f"   Data index range: {data.index.min()} to {data.index.max()}")

        # Convert start_date and end_date to datetime if they're not already
        if isinstance(self.start_date, datetime):
            start_date = self.start_date
        else:
            start_date = pd.to_datetime(self.start_date)

        if isinstance(self.end_date, datetime):
            end_date = self.end_date
        else:
            end_date = pd.to_datetime(self.end_date)

        # Filter data to the specified date range
        mask = (data.index >= start_date) & (data.index <= end_date)
        filtered_data = data[mask].copy()

        if len(filtered_data) == 0:
            print(f"❌ Error: No data available for the specified date range: {start_date} to {end_date}. ")
            print(f"   Available data range: {data.index.min()} to {data.index.max()}")
            print(f"   Requested start date: {start_date}")
            print(f"   Requested end date: {end_date}")
            print(f"   Total available data points: {len(data)}")

            # Check if the issue is with the date range
            if start_date > data.index.max():
                print(f"   ⚠️  Start date {start_date} is after the latest available data {data.index.max()}")
            if end_date < data.index.min():
                print(f"   ⚠️  End date {end_date} is before the earliest available data {data.index.min()}")
            
            return False
        
        # Update the data attribute
        self.data = filtered_data
        
        print(f"✅ Data validation complete:")
        print(f"   Final data shape: {self.data.shape}")
        print(f"   Date range: {self.data.index.min()} to {self.data.index.max()}")
        print(f"   Trading days: {len(self.data)}")
        
        # Check for gaps in the data (missing trading days)
        expected_business_days = len(pd.bdate_range(start=self.data.index.min(), end=self.data.index.max()))
        actual_trading_days = len(self.data)
        if actual_trading_days < expected_business_days * 0.9:  # Allow for some holidays
            print(f"⚠️  Warning: Data may have gaps. Expected ~{expected_business_days} business days, got {actual_trading_days}")
        
        return True

    def _add_position(self, position: Position):
        """
        Add a position to the positions list with volume validation.
        """
        
        # Volume validation - check all options in the spread
        if self.volume_config.enable_volume_validation and position.spread_options:
            for option in position.spread_options:
                if not self._validate_option_volume(option):
                    print(f"⚠️  Volume validation failed: {option.symbol} has insufficient volume")
                    self.volume_stats = self.volume_stats.increment_rejected_positions()
                    return  # Reject the position

        position_size = self._get_position_size(position)
        if position_size == 0:
            print(f"⚠️  Warning: Not enough capital to add position. Position size is 0.")
            return
        
        position.set_quantity(position_size)

        # For credit spreads, we need to reserve the maximum risk amount
        if position.strategy_type in [StrategyType.CALL_CREDIT_SPREAD, StrategyType.PUT_CREDIT_SPREAD]:
            # For credit spreads: Add the net credit received to capital
            # The net credit is already stored in position.entry_price
            credit_received = position.entry_price * position.quantity * 100
            self.capital += credit_received
            print(f"💰 Added net credit of ${credit_received:.2f} to capital")
        else:
            # For other position types, check if we have enough capital
            if self.capital < position.entry_price * position.quantity * 100:
                raise ValueError("Not enough capital to add position")

        print(f"Adding position: {position.__str__()}")
        
        self.positions.append(position)
        self.total_positions += 1

    def _remove_position(self, date: datetime, position: Position, exit_price: float, underlying_price: float = None, current_volumes: list[int] = None):
        """
        Remove a position from the positions list with enhanced current date volume validation.
        
        Args:
            date: Date at which the position is being closed
            position: Position to remove
            exit_price: Price at which the position is being closed
            underlying_price: Price of the underlying at the time of exit
            current_volumes: List of current volume data for each option in position.spread_options
        """
        # Enhanced volume validation - check all options in the spread with current date data
        if self.volume_config.enable_volume_validation and position.spread_options and current_volumes:
            volume_validation_failed = False
            failed_options = []
            
            # Increment options checked counter for each option in the spread
            for option in position.spread_options:
                self.volume_stats = self.volume_stats.increment_options_checked()
            
            for option, current_volume in zip(position.spread_options, current_volumes):
                if current_volume is None or current_volume < self.volume_config.min_volume:
                    volume_validation_failed = True
                    failed_options.append(option.symbol)
            
            if volume_validation_failed:
                print(f"⚠️  Volume validation failed for position closure: {', '.join(failed_options)} have insufficient volume")
                self.volume_stats = self.volume_stats.increment_rejected_closures()
                
                # Skip closing the position for this date due to insufficient volume
                print(f"⚠️  Skipping position closure for {date.date()} due to insufficient volume")
                return  # Skip closure and keep position open
        
        if position not in self.positions:
            print(f"⚠️  Warning: Position {position.__str__()} not found in positions list")
            return
        
        final_exit_price = exit_price
        if not exit_price:
            print(f"⚠️  Warning: Exit price not provided for position {position.__str__()}. Defaulting to 0.")
            final_exit_price = 0

        # Calculate the return for this position
        if position.get_days_to_expiration(date) < 1:
            if underlying_price is None:
                raise ValueError(f"Underlying price not provided for position {position.__str__()}")

            position_return = position.get_return_dollars_from_assignment(underlying_price)
            print(f"   Position closed by assignment: {position.__str__()}")
        else:
            position_return = position.get_return_dollars(final_exit_price)

        # Update capital based on position type
        if position.strategy_type in [StrategyType.CALL_CREDIT_SPREAD, StrategyType.PUT_CREDIT_SPREAD]:
            # For credit spreads: Subtract the cost to buy back the spread
            # The exit_price represents the cost to close the position
            cost_to_close = final_exit_price * position.quantity * 100
            self.capital -= cost_to_close
            print(f"💰 Subtracted cost to close of ${cost_to_close:.2f} from capital")
        else:
            # For other position types, add the return
            self.capital += position_return

        # Calculate daily return and add to tracking
        daily_return = (self.capital - self.previous_capital) / self.previous_capital
        self.daily_returns.append(daily_return)
        self.previous_capital = self.capital

        # Track closed position for statistics
        closed_position_data = {
            'strategy_type': position.strategy_type,
            'entry_date': position.entry_date,
            'exit_date': date,
            'entry_price': position.entry_price,
            'exit_price': final_exit_price,
            'return_dollars': position_return,
            'return_percentage': (position_return / position.get_max_risk()) * 100 if position.quantity else 0,
            'days_held': position.get_days_held(date),
            'max_risk': position.get_max_risk()
        }
        self.closed_positions.append(closed_position_data)
        
        # Remove the position
        self.positions.remove(position)
        
        # Log the position closure
        print(f"   Position closed: {position.__str__()}")
        print(f"     Entry: ${position.entry_price:.2f} | Exit: ${final_exit_price:.2f}")
        print(f"     Return: ${position_return:+.2f} | Capital: ${self.capital:.2f}")
    
    # TODO: Only works for credit spreads since using max risk
    def _get_position_size(self, position: Position) -> int:
        """
        Get the number of contracts to buy or sell for a position based on the max position size and the current capital.
        """
        if self.max_position_size is None:
            return 1
        
        max_position_capital = self.capital * self.max_position_size

        return int(max_position_capital / position.get_max_risk())

    def _calculate_sharpe_ratio(self) -> float:
        """
        Calculate the Sharpe Ratio based on daily returns.
        
        Returns:
            float: Sharpe Ratio (annualized)
        """
        if not self.daily_returns:
            return 0.0
        
        import numpy as np
        
        # Convert daily returns to numpy array
        returns = np.array(self.daily_returns)
        
        # Calculate mean and standard deviation of returns
        mean_return = np.mean(returns)
        std_return = np.std(returns, ddof=1)  # Use sample standard deviation
        
        # Avoid division by zero
        if std_return == 0:
            return 0.0
        
        # Calculate Sharpe Ratio (assuming risk-free rate of 0 for simplicity)
        # Annualize by multiplying by sqrt(252) for trading days
        sharpe_ratio = (mean_return / std_return) * np.sqrt(252)
        
        return sharpe_ratio
    
    def _validate_option_volume(self, option) -> bool:
        """
        Validate if an option has sufficient volume for trading.
        
        Note: Data fetching is handled by the Strategy class. This method
        only validates the volume data that is already present in the option.
        """
        self.volume_stats = self.volume_stats.increment_options_checked()
        
        if option.volume is None:
            return False
        return option.volume >= self.volume_config.min_volume

    def _print_position_statistics(self):
        """Print comprehensive position performance statistics"""
        overall_stats = self._calculate_overall_statistics()
        strategy_stats = self._calculate_strategy_statistics()
        
        self._print_overall_statistics(overall_stats)
        self._print_strategy_statistics(strategy_stats)
    
    def _calculate_overall_statistics(self) -> 'OverallPerformanceStats':
        """Calculate overall performance statistics"""
        total_positions = len(self.closed_positions)
        total_return = sum(pos['return_dollars'] for pos in self.closed_positions)
        winning_positions = [pos for pos in self.closed_positions if pos['return_dollars'] > 0]
        
        win_rate = len(winning_positions) / total_positions * 100 if total_positions > 0 else 0
        avg_return = total_return / total_positions if total_positions > 0 else 0
        avg_drawdown = self._calculate_average_drawdown(self.closed_positions, self.initial_capital)
        
        return OverallPerformanceStats(
            total_positions=total_positions,
            win_rate=win_rate,
            total_pnl=total_return,
            average_return=avg_return,
            average_drawdown=avg_drawdown
        )
    
    def _calculate_strategy_statistics(self) -> List['StrategyPerformanceStats']:
        """Calculate performance statistics for each strategy type"""
        strategy_data = {}
        
        # Group positions by strategy type
        for pos in self.closed_positions:
            strategy_type = pos['strategy_type']
            if strategy_type not in strategy_data:
                strategy_data[strategy_type] = []
            strategy_data[strategy_type].append(pos)
        
        # Calculate statistics for each strategy
        strategy_stats = []
        for strategy_type, positions in strategy_data.items():
            total_return = sum(pos['return_dollars'] for pos in positions)
            winning_positions = [pos for pos in positions if pos['return_dollars'] > 0]
            
            win_rate = len(winning_positions) / len(positions) * 100 if positions else 0
            avg_return = total_return / len(positions) if positions else 0
            avg_drawdown = self._calculate_average_drawdown(positions, self.initial_capital)
            
            stats = StrategyPerformanceStats(
                strategy_type=strategy_type,
                positions_count=len(positions),
                win_rate=win_rate,
                total_pnl=total_return,
                average_return=avg_return,
                average_drawdown=avg_drawdown
            )
            strategy_stats.append(stats)
        
        return strategy_stats
    
    def _calculate_average_drawdown(self, positions: List[dict], initial_capital: float) -> float:
        """Calculate average drawdown for a list of positions"""
        if not positions:
            return 0.0
        
        drawdowns = []
        peak_capital = initial_capital
        current_capital = initial_capital
        
        for pos in positions:
            current_capital += pos['return_dollars']
            if current_capital > peak_capital:
                peak_capital = current_capital
            
            # Only calculate drawdown if we have a positive peak and current value is below peak
            if peak_capital > 0 and current_capital < peak_capital:
                drawdown = (peak_capital - current_capital) / peak_capital * 100
                drawdowns.append(drawdown)
        
        return sum(drawdowns) / len(drawdowns) if drawdowns else 0.0
    
    def _print_overall_statistics(self, stats: 'OverallPerformanceStats'):
        """Print overall performance statistics"""
        print(f"\n📊 Position Performance Statistics:")
        print(f"   Total closed positions: {stats.total_positions}")
        print(f"   Overall win rate: {stats.win_rate:.1f}%")
        print(f"   Total P&L: ${stats.total_pnl:+,.2f}")
        print(f"   Average return per position: ${stats.average_return:+.2f}")
        print(f"   Average drawdown: {stats.average_drawdown:.1f}%")
    
    def _print_strategy_statistics(self, strategy_stats: List['StrategyPerformanceStats']):
        """Print strategy-specific performance statistics"""
        for stats in strategy_stats:
            print(f"\n   {stats.strategy_type.value.replace('_', ' ').title()}:")
            print(f"     Positions: {stats.positions_count}")
            print(f"     Win rate: {stats.win_rate:.1f}%")
            print(f"     Total P&L: ${stats.total_pnl:+,.2f}")
            print(f"     Average return: ${stats.average_return:+.2f}")
            print(f"     Average drawdown: {stats.average_drawdown:.1f}%")

    def _handle_insufficient_volume_closure(self, position: Position, date: datetime) -> bool:
        """
        Handle position closure when volume is insufficient.
        
        Args:
            position: The position to close
            date: The date of closure
            
        Returns:
            bool: True if closure should be skipped, False if closure should proceed
        """
        if self.volume_config.skip_closure_on_insufficient_volume:
            # Skip closure and keep position open
            print(f"⚠️  Skipping closure of position {position.__str__()} for {date.date()} due to insufficient volume")
            self.volume_stats = self.volume_stats.increment_skipped_closures()
            return True  # Indicate that closure should be skipped
        else:
            # Proceed with closure despite insufficient volume (for backward compatibility)
            print(f"⚠️  Proceeding with closure of position {position.__str__()} despite insufficient volume")
            return False  # Indicate that closure should proceed


def parse_arguments():
    """Parse command line arguments for backtest configuration"""
    parser = argparse.ArgumentParser(description='Run backtest with specified strategy')
    parser.add_argument('--strategy', 
                       choices=StrategyFactory.get_available_strategies(),
                       default='credit_spread',
                       help='Strategy to use for backtesting')
    parser.add_argument('--start-date-offset', type=int, default=60,
                       help='Start date offset for strategy')
    parser.add_argument('--stop-loss', type=float, default=0.6,
                       help='Stop loss percentage')
    parser.add_argument('--profit-target', type=float, default=None,
                       help='Profit target percentage')
    parser.add_argument('--initial-capital', type=float, default=5000,
                       help='Initial capital for backtesting')
    parser.add_argument('--max-position-size', type=float, default=0.20,
                       help='Maximum position size as fraction of capital')
    parser.add_argument('--start-date', type=str, default='2025-01-01',
                       help='Start date for backtest (YYYY-MM-DD)')
    parser.add_argument('--end-date', type=str, default='2025-07-15',
                       help='End date for backtest (YYYY-MM-DD)')
    parser.add_argument('--symbol', type=str, default='SPY',
                       help='Symbol to trade')
    parser.add_argument('--verbose', action='store_true', default=False,
                       help='Run in quiet mode')
    
    return parser.parse_args()

if __name__ == "__main__":
    # Parse command line arguments
    args = parse_arguments()
    
    # Convert date strings to datetime objects
    start_date = datetime.strptime(args.start_date, '%Y-%m-%d')
    end_date = datetime.strptime(args.end_date, '%Y-%m-%d')
    
    print(f"🚀 Starting backtest with strategy: {args.strategy}")
    print(f"   Date range: {start_date.date()} to {end_date.date()}")
    print(f"   Symbol: {args.symbol}")
    print(f"   Initial capital: ${args.initial_capital:,.2f}")
    print(f"   Max position size: {args.max_position_size * 100:.1f}%")
    print(f"   Stop loss: {args.stop_loss * 100:.1f}%")
    if args.profit_target:
        print(f"   Profit target: {args.profit_target * 100:.1f}%")
    print()

    data_retriever = DataRetriever(
        symbol=args.symbol, 
        hmm_start_date=start_date, 
        lstm_start_date=start_date, 
        use_free_tier=False, 
        quiet_mode=not args.verbose
    )

    # Load model directory from environment variable
    model_save_base_path = os.getenv('MODEL_SAVE_BASE_PATH', 'Trained_Models')
    model_dir = os.path.join(model_save_base_path, 'lstm_poc', args.symbol, 'latest')

    options_handler = data_retriever.options_handler

    try:
        hmm_model = load_hmm_model(model_dir)
        lstm_model, scaler = load_lstm_model(model_dir, return_lstm_instance=True)

        # Then prepare the data for LSTM
        data, options_data = data_retriever.prepare_data_for_lstm(state_classifier=hmm_model)

        # Create strategy using the builder pattern
        strategy = create_strategy_from_args(
            strategy_name=args.strategy,
            lstm_model=lstm_model,
            lstm_scaler=scaler,
            options_handler=options_handler,
            start_date_offset=args.start_date_offset,
            stop_loss=args.stop_loss,
            profit_target=args.profit_target
        )
        
        if strategy is None:
            print("❌ Failed to create strategy")
            exit(1)
        
        strategy.set_data(data, options_data)

        backtester = BacktestEngine(
            data=data, 
            strategy=strategy,
            initial_capital=args.initial_capital,
            start_date=start_date,
            end_date=end_date,
            max_position_size=args.max_position_size,
            quiet_mode=not args.verbose
        )
        
        success = backtester.run()
        if success:
            print("✅ Backtest completed successfully!")
        else:
            print("❌ Backtest failed!")
            
    except Exception as e:
        print(f"❌ Error during backtest: {e}")
        import traceback
        traceback.print_exc()
