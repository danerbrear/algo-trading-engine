import os
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Optional, Union
import argparse
import numpy as np

from algo_trading_engine.common.options_handler import OptionsHandler

from .models import Benchmark, Strategy, Position, StrategyType
from algo_trading_engine.common.data_retriever import DataRetriever
from algo_trading_engine.common.functions import load_hmm_model, load_lstm_model
from .config import VolumeConfig, VolumeStats, OverallPerformanceStats, StrategyPerformanceStats
from algo_trading_engine.common.progress_tracker import ProgressTracker, set_global_progress_tracker, progress_print
from .strategy_builder import StrategyFactory, create_strategy_from_args
from algo_trading_engine.core.engine import TradingEngine
from algo_trading_engine.models.config import BacktestConfig as BacktestConfigDTO
from algo_trading_engine.models.metrics import PerformanceMetrics, PositionStats

class BacktestEngine(TradingEngine):
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
        self._strategy = strategy
        self._capital = initial_capital
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

    @property
    def strategy(self) -> Strategy:
        """Get the strategy being used by this engine."""
        return self._strategy
    
    @strategy.setter
    def strategy(self, value: Strategy):
        """Set the strategy for this engine."""
        self._strategy = value
    
    @property
    def capital(self) -> float:
        """Get current capital."""
        return self._capital
    
    @capital.setter
    def capital(self, value: float):
        """Set the capital for this engine."""
        self._capital = value

    @classmethod
    def from_config(cls, config: BacktestConfigDTO) -> 'BacktestEngine':
        """
        Create BacktestEngine from configuration.
        
        Handles all data fetching, strategy creation, and setup internally.
        Child projects only need to provide configuration.
        
        Args:
            config: BacktestConfig DTO with all necessary parameters
            
        Returns:
            Configured BacktestEngine instance ready to run
            
        Raises:
            ValueError: If configuration is invalid or data fetching fails
        """
        # Internal: Calculate LSTM start date (days before backtest start)
        lstm_start_date = (config.start_date - timedelta(days=config.lstm_start_date_offset))
        
        # Internal: Create data retriever
        retriever = DataRetriever(
            symbol=config.symbol,
            lstm_start_date=lstm_start_date.strftime("%Y-%m-%d"),
            quiet_mode=config.quiet_mode,
            use_free_tier=config.use_free_tier
        )
        
        # Internal: Fetch data for backtest period
        data = retriever.fetch_data_for_period(
            config.start_date.strftime("%Y-%m-%d"),
            'backtest'
        )
        
        if data is None or len(data) == 0:
            raise ValueError(f"Failed to fetch data for {config.symbol} from {config.start_date.date()} to {config.end_date.date()}")
        
        # Internal: Create options handler
        options_handler = OptionsHandler(
            symbol=config.symbol,
            api_key=config.api_key,
            use_free_tier=config.use_free_tier
        )
        
        # Internal: Create or use provided strategy
        if isinstance(config.strategy_type, str):
            # Create strategy from string name
            strategy = create_strategy_from_args(
                strategy_name=config.strategy_type,
                symbol=config.symbol,
                options_handler=options_handler,
                stop_loss=config.stop_loss,
                profit_target=config.profit_target
            )
            if strategy is None:
                raise ValueError(f"Failed to create strategy: {config.strategy_type}")
        else:
            # Strategy instance provided - inject options_handler if it has that attribute
            strategy = config.strategy_type
            if hasattr(strategy, 'options_handler'):
                strategy.options_handler = options_handler
        
        # Internal: Set data on strategy
        strategy.set_data(data, retriever.treasury_rates)
        
        # Create and return engine
        return cls(
            data=data,
            strategy=strategy,
            initial_capital=config.initial_capital,
            start_date=config.start_date,
            end_date=config.end_date,
            max_position_size=config.max_position_size,
            volume_config=config.volume_config,
            enable_progress_tracking=config.enable_progress_tracking,
            quiet_mode=config.quiet_mode
        )

    def run(self) -> bool:
        """
        Run the backtest.
        """
        
        # Validate the data using the strategy's validation method
        if not self.strategy.validate_data(self.data):
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
    
    def get_performance_metrics(self) -> PerformanceMetrics:
        """
        Get performance statistics for the backtest.
        
        Returns:
            PerformanceMetrics object with all performance data
        """
        # Calculate overall stats
        overall_stats = self._calculate_overall_statistics()
        strategy_stats = self._calculate_strategy_statistics()
        
        # Calculate final return
        final_return = self.capital - self.initial_capital
        final_return_pct = (final_return / self.initial_capital) * 100
        
        # Calculate Sharpe Ratio
        sharpe_ratio = self._calculate_sharpe_ratio()
        
        # Calculate max drawdown
        max_drawdown = overall_stats.max_drawdown
        
        # Convert closed positions to PositionStats
        position_stats_list = []
        for pos in self.closed_positions:
            position_stats = PositionStats(
                strategy_type=pos['strategy_type'],
                entry_date=pos['entry_date'],
                exit_date=pos['exit_date'],
                entry_price=pos['entry_price'],
                exit_price=pos['exit_price'],
                return_dollars=pos['return_dollars'],
                return_percentage=pos['return_percentage'],
                days_held=pos['days_held'],
                max_risk=pos['max_risk']
            )
            position_stats_list.append(position_stats)
        
        return PerformanceMetrics(
            total_return=final_return,
            total_return_pct=final_return_pct,
            sharpe_ratio=sharpe_ratio,
            max_drawdown=max_drawdown,
            win_rate=overall_stats.win_rate,
            total_positions=self.total_positions,
            closed_positions=position_stats_list,
            strategy_stats=strategy_stats,
            overall_stats=overall_stats,
            benchmark_return=self.benchmark.get_return_dollars(),
            benchmark_return_pct=self.benchmark.get_return_percentage()
        )
    
    def get_positions(self) -> List[Position]:
        """
        Get current open positions.
        
        Returns:
            List of currently open Position objects
        """
        return self.positions.copy()


    def _add_position(self, position: Position):
        """
        Add a position to the positions. Rejects positions with insufficient volume and determines position size based
        on capital provided to the backtest.
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

        print(f"Adding position: {position.__str__()}\n")
        
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
                
                # Skip closing the position for this date due to insufficient volume unless expired
                if position.get_days_to_expiration(date) > 0:
                    print(f"⚠️  Skipping position closure for {date.date()} due to insufficient volume")
                    return  # Skip closure and keep position open

        if position not in self.positions:
            raise ValueError(f"Could not find position to close within open positions. {position.__str__()}")

        # Calculate the return for this position due to assignment
        if position.get_days_to_expiration(date) < 1:
            if underlying_price is None:
                raise ValueError(f"Underlying price not provided for position {position.__str__()}")

            position_return = position.get_return_dollars_from_assignment(underlying_price)
            print(f"   Position closed by assignment: {position.__str__()}")
        else:
            if exit_price is None: # Allows for exit price to be 0 - possible if ATM and OTM market price are equal
                raise ValueError("Exit price not provided for the unexpired position")
            position_return = position.get_return_dollars(exit_price)

        # Update capital based on position type
        if position.strategy_type in [StrategyType.CALL_CREDIT_SPREAD, StrategyType.PUT_CREDIT_SPREAD]:
            # For credit spreads: Subtract the cost to buy back the spread
            # The exit_price represents the cost to close the position
            cost_to_close = exit_price * position.quantity * 100
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
            'exit_price': exit_price,
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
        print(f"     Entry: ${position.entry_price:.2f} | Exit: ${exit_price:.2f}")
        print(f"     Return: ${position_return:+.2f} | Capital: ${self.capital:.2f}\n")
    
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
        min_dd, mean_dd, max_dd = self._calculate_drawdown_stats(self.closed_positions, self.initial_capital)
        
        return OverallPerformanceStats(
            total_positions=total_positions,
            win_rate=win_rate,
            total_pnl=total_return,
            average_return=avg_return,
            min_drawdown=min_dd,
            mean_drawdown=mean_dd,
            max_drawdown=max_dd
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
            min_dd, mean_dd, max_dd = self._calculate_drawdown_stats(positions, self.initial_capital)
            
            stats = StrategyPerformanceStats(
                strategy_type=strategy_type,
                positions_count=len(positions),
                win_rate=win_rate,
                total_pnl=total_return,
                average_return=avg_return,
                min_drawdown=min_dd,
                mean_drawdown=mean_dd,
                max_drawdown=max_dd
            )
            strategy_stats.append(stats)
        
        return strategy_stats
    
    def _calculate_drawdown_stats(self, positions: List[dict], initial_capital: float) -> tuple[float, float, float]:
        """
        Calculate drawdown statistics from closed positions.
        
        Returns:
            Tuple of (min_drawdown, mean_drawdown, max_drawdown) as percentages
        """
        if not positions:
            return 0.0, 0.0, 0.0
        
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
        
        if not drawdowns:
            return 0.0, 0.0, 0.0
        
        min_dd = min(drawdowns)
        mean_dd = sum(drawdowns) / len(drawdowns)
        max_dd = max(drawdowns)
        
        return min_dd, mean_dd, max_dd
    
    def _print_overall_statistics(self, stats: 'OverallPerformanceStats'):
        """Print overall performance statistics"""
        print(f"\n📊 Position Performance Statistics:")
        print(f"   Total closed positions: {stats.total_positions}")
        print(f"   Overall win rate: {stats.win_rate:.1f}%")
        print(f"   Total P&L: ${stats.total_pnl:+,.2f}")
        print(f"   Average return per position: ${stats.average_return:+.2f}")
        
        # Print drawdown stats
        if stats.max_drawdown > 0:
            print(f"   Drawdowns: Min: {stats.min_drawdown:.2f}% | Mean: {stats.mean_drawdown:.2f}% | Max: {stats.max_drawdown:.2f}%")
        else:
            print(f"   Drawdowns: No drawdowns detected")
    
    def _print_strategy_statistics(self, strategy_stats: List['StrategyPerformanceStats']):
        """Print strategy-specific performance statistics"""
        for stats in strategy_stats:
            print(f"\n   {stats.strategy_type.value.replace('_', ' ').title()}:")
            print(f"     Positions: {stats.positions_count}")
            print(f"     Win rate: {stats.win_rate:.1f}%")
            print(f"     Total P&L: ${stats.total_pnl:+,.2f}")
            print(f"     Average return: ${stats.average_return:+.2f}")
            
            # Print drawdown stats
            if stats.max_drawdown > 0:
                print(f"     Drawdowns: Min: {stats.min_drawdown:.2f}% | Mean: {stats.mean_drawdown:.2f}% | Max: {stats.max_drawdown:.2f}%")
            else:
                print(f"     Drawdowns: No drawdowns detected")

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
    parser.add_argument('--stop-loss', type=float, default=None,
                       help='Stop loss percentage')
    parser.add_argument('--profit-target', type=float, default=None,
                       help='Profit target percentage')
    parser.add_argument('--initial-capital', type=float, default=3000,
                       help='Initial capital for backtesting')
    parser.add_argument('--max-position-size', type=float, default=0.40,
                       help='Maximum position size as fraction of capital')
    parser.add_argument('--start-date', type=str, default='2024-08-01',
                       help='Start date for backtest (YYYY-MM-DD)')
    parser.add_argument('--end-date', type=str, default='2025-10-15',
                       help='End date for backtest (YYYY-MM-DD)')
    parser.add_argument('--symbol', type=str, default='SPY',
                       help='Symbol to trade')
    parser.add_argument('--verbose', action='store_true', default=False,
                       help='Run in quiet mode')
    parser.add_argument('-f', '--free', action='store_true', default=False,
                       help='Use free tier rate limiting (13 second timeout between API requests)')
    
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
    print(f"   Stop loss: {args.stop_loss * 100:.1f}%") if args.stop_loss else print("   Stop loss: None")
    if args.profit_target:
        print(f"   Profit target: {args.profit_target * 100:.1f}%")
    print()

    data_retriever = DataRetriever(
        symbol=args.symbol, 
        hmm_start_date=start_date, 
        lstm_start_date=start_date, 
        use_free_tier=args.free, 
        quiet_mode=not args.verbose
    )

    # Load treasury rates before starting backtest
    data_retriever.load_treasury_rates(start_date, end_date)

    try:
        data = data_retriever.fetch_data_for_period(start_date, 'backtest')

        # Create options handler to inject into strategy
        # Get API key from environment or args if provided
        api_key = getattr(args, 'api_key', None)
        options_handler = OptionsHandler(
            symbol=args.symbol,
            api_key=api_key,
            use_free_tier=args.free
        )

        strategy = create_strategy_from_args(
            strategy_name=args.strategy,
            symbol=args.symbol,
            options_handler=options_handler,
            start_date_offset=args.start_date_offset,
            stop_loss=args.stop_loss,
            profit_target=args.profit_target
        )
        
        if strategy is None:
            print("❌ Failed to create strategy")
            exit(1)

        if (args.strategy == 'credit_spread'):
            data = data_retriever.prepare_data_for_lstm()
        
        strategy.set_data(data, data_retriever.treasury_rates)

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
