"""
Trading engine interfaces and implementations.

This module provides the abstract base class for trading engines
and concrete implementations for backtesting and paper trading.
"""

from abc import ABC, abstractmethod, abstractclassmethod
from typing import List, Optional, Union, TYPE_CHECKING
from datetime import datetime
import pandas as pd

from .strategy import Strategy
from algo_trading_engine.models.config import PaperTradingConfig

if TYPE_CHECKING:
    from algo_trading_engine.backtest.models import Position
    from algo_trading_engine.common.models import OptionChain
    from algo_trading_engine.models.config import BacktestConfig


class TradingEngine(ABC):
    """
    Abstract base class for trading engines.
    
    Both backtesting and paper trading engines implement this interface,
    allowing for unified usage patterns.
    """

    def __init__(self, strategy: Strategy, data: pd.DataFrame):
        self._strategy = strategy
        self._data = data
        strategy.get_current_underlying_price = self._get_current_underlying_price
    
    @abstractmethod
    def run(self) -> bool:
        """
        Execute the trading simulation.
        
        Returns:
            True if execution completed successfully, False otherwise
        """
        pass
    
    @abstractmethod
    def get_positions(self) -> List['Position']:
        """
        Get current open positions.
        
        Returns:
            List of currently open Position objects
        """
        pass
    
    @property
    def data(self) -> pd.DataFrame:
        """Get the market data."""
        return self._data
    
    @property
    @abstractmethod
    def strategy(self) -> Strategy:
        """Get the strategy being used by this engine."""
        pass
    
    @abstractclassmethod
    def from_config(cls, config: Union['BacktestConfig', PaperTradingConfig]) -> 'TradingEngine':
        """
        Create trading engine from configuration.
        
        Factory method that handles all data fetching, strategy creation, and setup.
        Child projects only need to provide configuration.
        
        Args:
            config: Configuration DTO (BacktestConfig or PaperTradingConfig)
            
        Returns:
            Configured TradingEngine instance ready to run
            
        Raises:
            ValueError: If configuration is invalid or data fetching fails
        """
        pass

    def _get_current_underlying_price(self, date: datetime, symbol: str) -> Optional[float]:
        """
        Fetch and return the live price if the date is the current date, otherwise return last_price for the date.
        
        This method is injected into strategies so they can get current underlying prices
        without needing to manage DataRetriever themselves.
        
        Args:
            date: Date to get price for
            symbol: Symbol to fetch price for (e.g., 'SPY')
            
        Returns:
            Current underlying price as float, or None if unavailable
            
        Raises:
            ValueError: If live price fetch fails and date is current date
        """
        current_date = datetime.now().date()
        if date.date() == current_date:
            try:
                from algo_trading_engine.common.data_retriever import DataRetriever
                data_retriever = DataRetriever(symbol=symbol, use_free_tier=True, quiet_mode=True)
                live_price = data_retriever.get_live_price()
            except Exception as e:
                raise ValueError(f"Failed to fetch live price from DataRetriever: {e}")

            if live_price is not None:
                return live_price
            else:
                raise ValueError("Failed to fetch live price from DataRetriever.")
        else:
            # Historical date - return Close price from data
            try:
                return float(self.data.loc[date]['Close'])
            except (KeyError, IndexError):
                # If exact date not found, try to get closest available date
                try:
                    return float(self.data.loc[self.data.index <= date]['Close'].iloc[-1])
                except (IndexError, KeyError):
                    raise ValueError(f"Could not find price data for date {date.date()}")
    
        
    def get_current_volumes_for_position(self, position: 'Position', date: datetime) -> list[int]:
        """
        Fetch current date volume data for all options in a position using options_retriever.
        """
        current_volumes = []
        
        # Check if position has spread_options
        if not hasattr(position, 'spread_options') or position.spread_options is None:
            return current_volumes
            
        for option in position.spread_options:
            try:
                # Get current volume data from strategy's callable if available
                if hasattr(self.strategy, 'get_option_bar') and callable(self.strategy.get_option_bar):
                    bar_data = self.strategy.get_option_bar(option, date)
                else:
                    print(f"⚠️  No option bar data available for {option.ticker} on {date.date()}")
                    current_volumes.append(None)
                    continue
                
                if bar_data and hasattr(bar_data, 'volume') and bar_data.volume is not None:
                    current_volumes.append(bar_data.volume)
                    print(f"📡 Fetched volume data for {option.ticker} on {date.date()}: {bar_data.volume}")
                else:
                    current_volumes.append(None)
                    print(f"⚠️  No volume data available for {option.ticker} on {date.date()}")
                    
            except Exception as e:
                print(f"⚠️  Error fetching volume data for {option.symbol}: {e}")
                current_volumes.append(None)
        return current_volumes
    
    def compute_exit_price(self, position: 'Position', date: datetime) -> Optional[float]:
        """
        Compute exit price for a position on a specific date.
        
        Args:
            position: Position to compute exit price for
            date: Date to compute exit price on
            
        Returns:
            Exit price or None if unavailable
        """
        try:
            if not position.spread_options or len(position.spread_options) != 2:
                return None
                
            atm_option, otm_option = position.spread_options
            
            # Get bar data for both options using strategy's callable if available
            if hasattr(self.strategy, 'get_option_bar') and callable(self.strategy.get_option_bar):
                atm_bar = self.strategy.get_option_bar(atm_option, date)
                otm_bar = self.strategy.get_option_bar(otm_option, date)
            else:
                return None
            
            if not atm_bar or not otm_bar:
                return None
            
            # Use the position's method to calculate exit price from bars
            exit_price = position.calculate_exit_price_from_bars(atm_bar, otm_bar)
            return exit_price
            
        except Exception as e:
            print(f"⚠️  Error computing exit price: {e}")
            return None
    
    def check_univeral_close_conditions(self, date: datetime):
        """
        Check if the position should be closed due to universal close conditions.
        """
        # Get symbol from strategy if available, otherwise default to 'SPY'
        symbol = getattr(self.strategy, 'symbol', 'SPY')
        current_underlying_price = self.strategy.get_current_underlying_price(date, symbol)
        for position in self.get_positions():
            # Get current volumes for this specific position
            current_volumes = self.get_current_volumes_for_position(position, date)
            
            # Compute exit price for profit target and stop loss checks
            exit_price = self.compute_exit_price(position, date)
            
            if self._should_close_due_to_assignment(position, date):
                print(f"⏰ Position {position.__str__()} expired or near expiration (days to exp: {position.get_days_to_expiration(date)})")
                self._remove_position(date, position, 0.0, underlying_price=current_underlying_price, current_volumes=current_volumes)
            elif self._should_close_due_to_profit_target(position, exit_price):
                print(f"💰 Profit target hit for {position.__str__()} at exit {exit_price}")
                self._remove_position(date, position, exit_price if exit_price is not None else 0.0, current_volumes=current_volumes)
            elif self._should_close_due_to_stop(position, exit_price):
                print(f"💰 Stop loss hit for {position.__str__()} at exit {exit_price}")
                self._remove_position(date, position, exit_price if exit_price is not None else 0.0, current_volumes=current_volumes)
    
    def _should_close_due_to_assignment(self, position: 'Position', date: datetime) -> bool:
        try:
            return position.get_days_to_expiration(date) < 1
        except Exception:
            return False

    def _should_close_due_to_profit_target(self, position: 'Position', exit_price: Optional[float]) -> bool:
        if exit_price is None or self.strategy.profit_target is None:
            return False
        return position.profit_target_hit(self.strategy.profit_target, exit_price)

    def _should_close_due_to_stop(self, position: 'Position', exit_price: Optional[float]) -> bool:
        if exit_price is None or self.strategy.stop_loss is None:
            return False
        return position.stop_loss_hit(self.strategy.stop_loss, exit_price)

# BacktestEngine is defined in backtest.main and implements TradingEngine
# We'll import it here for convenience, but it's defined in backtest/main.py
# to avoid circular imports


class PaperTradingEngine(TradingEngine):
    """
    Paper trading engine implementation.
    
    This engine runs strategies against live market data in real-time,
    simulating trades without actually executing them.
    """
    
    def __init__(
        self,
        strategy: Strategy,
        config: PaperTradingConfig,
        options_handler=None
    ):
        """
        Initialize paper trading engine.
        
        Args:
            strategy: Trading strategy to execute
            config: Paper trading configuration
            options_handler: Options handler instance (optional, will be extracted from strategy if not provided)
        """
        # PaperTradingEngine doesn't have its own data - use strategy's data
        # If strategy doesn't have data yet, create empty DataFrame
        strategy_data = getattr(strategy, 'data', None)
        if strategy_data is None:
            import pandas as pd
            strategy_data = pd.DataFrame()
        super().__init__(strategy, strategy_data)
        self._config = config
        self._positions: List['Position'] = []
        self._closed_positions: List[dict] = []
        self._running = False
        
        # Store options_handler for use in run()
        if options_handler is not None:
            self._options_handler = options_handler
        elif hasattr(strategy, 'options_handler'):
            self._options_handler = strategy.options_handler
        else:
            self._options_handler = None
    
    @property
    def strategy(self) -> Strategy:
        """Get the strategy being used by this engine."""
        return self._strategy
    
    def run(self) -> bool:
        """
        Execute paper trading using the recommendation engine.
        
        Similar to recommend_cli.py, this runs the InteractiveStrategyRecommender
        to produce recommendations for the current date.
        
        Returns:
            True if execution completed successfully, False otherwise
        """
        from datetime import datetime
        from algo_trading_engine.prediction.decision_store import JsonDecisionStore
        from algo_trading_engine.prediction.capital_manager import CapitalManager
        from algo_trading_engine.prediction.recommendation_engine import InteractiveStrategyRecommender
        
        # Get options handler
        if self._options_handler is None:
            print("❌ ERROR: Options handler not available")
            return False
        
        # Load capital allocation configuration
        config_path = "config/strategies/capital_allocations.json"
        
        # Get strategy name before trying to load config
        strategy_name = self._get_strategy_name_from_class()
        
        try:
            # Ensure config file exists and strategy is initialized
            CapitalManager.initialize_config_for_strategy(
                config_path,
                strategy_name,
                default_capital=10000.0,
                default_max_risk_pct=0.05
            )
            
            store = JsonDecisionStore()
            capital_manager = CapitalManager.from_config_file(config_path, store)
        except Exception as e:
            print(f"❌ ERROR: Failed to load capital allocation config: {e}")
            return False
        
        # Get current date
        run_date = datetime.now()
        
        # Create recommender (needed for position status checks)
        recommender = InteractiveStrategyRecommender(
            self._strategy,
            store,
            capital_manager,
            auto_yes=False
        )
        
        # Check for open positions and display status
        open_records = store.get_open_positions(symbol=self._config.symbol)
        if open_records:
            print(f"📊 Open positions found: {len(open_records)}")
            statuses = recommender.get_open_positions_status(run_date)
            if statuses:
                print("\n📈 Open position status:")
                for s in statuses:
                    pnl_dollars = f"${s['pnl_dollars']:.2f}" if s.get('pnl_dollars') is not None else "N/A"
                    pnl_pct = f"{s['pnl_percent']:.1%}" if s.get('pnl_percent') is not None else "N/A"
                    print(
                        f"  - {s['symbol']} {s['strategy_type']} x{s['quantity']} | "
                        f"Entry ${s['entry_price']:.2f}  Exit ${s['exit_price']:.2f} | "
                        f"P&L {pnl_dollars} ({pnl_pct}) | Held {s['days_held']}d  DTE {s['dte']}d"
                    )
                print()
        
        print(f"📅 Running recommendation flow for {run_date.date()}")
        
        # Display capital status
        print(capital_manager.get_status_summary(strategy_name))
        print()
        
        try:
            # Run full recommendation flow (both open and close recommendations)
            recommender.run(run_date, auto_yes=False)
            self.check_univeral_close_conditions(run_date)
            return True
        except Exception as e:
            print(f"❌ ERROR: Failed to run recommendation engine: {e}")
            return False
    
    def _get_strategy_name_from_class(self) -> str:
        """
        Get strategy name from strategy class for capital manager.
        
        Uses the same logic as InteractiveStrategyRecommender to ensure consistency.
        
        Returns:
            Strategy name string (e.g., 'credit_spread', 'velocity_momentum', 'my_custom')
        """
        import re
        
        # Remove "Strategy" suffix and convert CamelCase to snake_case
        class_name = self._strategy.__class__.__name__.replace("Strategy", "")
        strategy_name = re.sub(r'(?<!^)(?=[A-Z])', '_', class_name).lower()
        
        # Map class names to config keys for built-in strategies
        name_mapping = {
            "credit_spread": "credit_spread",
            "velocity_signal_momentum": "velocity_momentum",
        }
        
        return name_mapping.get(strategy_name, strategy_name)
    
    def get_positions(self) -> List['Position']:
        """Get current open positions."""
        return self._positions.copy()
    
    @classmethod
    def from_config(cls, config: PaperTradingConfig) -> 'PaperTradingEngine':
        """
        Create PaperTradingEngine from configuration.
        
        Handles all data fetching, strategy creation, and setup internally.
        Child projects only need to provide configuration.
        
        Args:
            config: PaperTradingConfig DTO with all necessary parameters
            
        Returns:
            Configured PaperTradingEngine instance ready to run
            
        Raises:
            ValueError: If configuration is invalid or data fetching fails
        """
        # Internal: Create data retriever
        # For paper trading, we need recent historical data for strategy initialization
        # Use a default lookback period (e.g., 120 days) for LSTM data
        from datetime import timedelta
        from algo_trading_engine.common.data_retriever import DataRetriever
        from algo_trading_engine.common.options_handler import OptionsHandler
        from algo_trading_engine.backtest.strategy_builder import create_strategy_from_args
        
        # Calculate start date for data retrieval (120 days back from today)
        today = datetime.now()
        lstm_start_date = (today - timedelta(days=120)).strftime("%Y-%m-%d")
        
        retriever = DataRetriever(
            symbol=config.symbol,
            lstm_start_date=lstm_start_date,
            quiet_mode=True,
            use_free_tier=config.use_free_tier,
            bar_interval=config.bar_interval
        )
        
        # Internal: Fetch recent data for strategy initialization
        # For paper trading, we fetch data up to today
        data = retriever.fetch_data_for_period(lstm_start_date)
        
        if data is None or len(data) == 0:
            raise ValueError(f"Failed to fetch data for {config.symbol}")
        
        # Internal: Create options handler
        options_handler = OptionsHandler(
            symbol=config.symbol,
            api_key=config.api_key,
            use_free_tier=config.use_free_tier
        )
        
        # Internal: Extract methods as callables (no imports needed by child repos)
        get_contract_list_for_date = options_handler.get_contract_list_for_date
        get_option_bar = options_handler.get_option_bar
        get_options_chain = options_handler.get_options_chain
        
        # Internal: Create or use provided strategy
        if isinstance(config.strategy_type, str):
            # Create strategy from string name
            strategy = create_strategy_from_args(
                strategy_name=config.strategy_type,
                symbol=config.symbol,
                get_contract_list_for_date=get_contract_list_for_date,
                get_option_bar=get_option_bar,
                get_options_chain=get_options_chain,
                get_current_volumes_for_position=cls.get_current_volumes_for_position,
                compute_exit_price=cls.compute_exit_price,
                options_handler=options_handler,  # Needed for CreditSpreadStrategy with LSTM
                stop_loss=config.stop_loss,
                profit_target=config.profit_target
            )
            if strategy is None:
                raise ValueError(f"Failed to create strategy: {config.strategy_type}")
        else:
            # Strategy instance provided - inject callables if strategy expects them
            strategy = config.strategy_type
            if hasattr(strategy, 'get_contract_list_for_date'):
                strategy.get_contract_list_for_date = get_contract_list_for_date
                strategy.get_option_bar = get_option_bar
                strategy.get_options_chain = get_options_chain
                strategy.get_current_volumes_for_position = cls.get_current_volumes_for_position
                strategy.compute_exit_price = cls.compute_exit_price
            elif hasattr(strategy, 'options_handler'):
                # Backward compatibility: if strategy still uses options_handler, inject it
                strategy.options_handler = options_handler
        
        # Internal: Set data on strategy
        strategy.set_data(data, retriever.treasury_rates)
        
        # Create and return engine
        return cls(
            strategy=strategy,
            config=config,
            options_handler=options_handler
        )

