"""
Paper trading engine (public API).
"""

from __future__ import annotations

from datetime import datetime
from typing import List, TYPE_CHECKING

from algo_trading_engine._internal.common.logger import configure_logger, get_logger, log_and_echo
from algo_trading_engine._internal.common.trading_engine import (
    DEFAULT_PAPER_TRADING_LSTM_LOOKBACK_DAYS,
    TradingEngine,
    compute_paper_trading_fetch_start_date,
    make_rt_option_bar,
)

__all__ = [
    "DEFAULT_PAPER_TRADING_LSTM_LOOKBACK_DAYS",
    "PaperTradingEngine",
    "compute_paper_trading_fetch_start_date",
]

from algo_trading_engine.models.config import PaperTradingConfig
from algo_trading_engine.strategy import IndicatorUpdateError, Strategy

if TYPE_CHECKING:
    from algo_trading_engine.vo import Position

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

        super().__init__(strategy, strategy_data, config, bar_interval=config.bar_interval)

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
        from algo_trading_engine.database.decision_store import JsonDecisionStore
        from algo_trading_engine._internal.trade.capital_manager import CapitalManager
        from algo_trading_engine._internal.trade.recommendation_engine import InteractiveStrategyRecommender
        
        # Logger already configured in from_config(); ensure it's set for this run
        configure_logger("trade", log_level="info", observer=self._config.observer)

        # Get options handler
        if self._options_handler is None:
            get_logger().error("Options handler not available")
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
                default_max_risk_pct=0.05,
                create_dirs=self._config.use_cache
            )
            
            store = self._config.decision_store or JsonDecisionStore()
            capital_manager = CapitalManager.from_config_file(config_path, store)
        except Exception as e:
            get_logger().error(f"Failed to load capital allocation config: {e}")
            return False
        
        # Get current date. Sub-second precision cannot be represented in coarser
        # bar-index resolutions, which breaks indicator writes keyed on this date.
        run_date = datetime.now().replace(microsecond=0)
        
        # Create recommender (needed for position status checks)
        recommender = InteractiveStrategyRecommender(
            self._strategy,
            store,
            capital_manager,
            auto_yes=self._config.auto_yes
        )
        
        # Check for open positions and display status (recommendation-relevant -> stdout via log_and_echo).
        # Scope by strategy_name so one strategy's Lambda run does not load another strategy's Dynamo rows.
        open_records = store.get_open_positions(
            symbol=self._config.symbol,
            strategy_name=strategy_name,
        )
        if open_records:
            log_and_echo(f"Open positions found: {len(open_records)}")
            statuses = recommender.get_open_positions_status(run_date)
            if statuses:
                log_and_echo("\nOpen position status:")
                for s in statuses:
                    pnl_dollars = f"${s['pnl_dollars']:.2f}" if s.get('pnl_dollars') is not None else "N/A"
                    pnl_pct = f"{s['pnl_percent']:.1%}" if s.get('pnl_percent') is not None else "N/A"
                    log_and_echo(
                        f"  - {s['symbol']} {s['strategy_type']} x{s['quantity']} | "
                        f"Entry ${s['entry_price']:.2f}  Exit ${s['exit_price']:.2f} | "
                        f"P&L {pnl_dollars} ({pnl_pct}) | Held {s['days_held']}d  DTE {s['dte']}d"
                    )
                log_and_echo("")

        log_and_echo(f"Running recommendation flow for {run_date.date()}")

        # Display capital status (recommendation-relevant)
        log_and_echo(capital_manager.get_status_summary(strategy_name))
        log_and_echo("")

        try:
            recommender.run(run_date)
            self.check_univeral_close_conditions(run_date)
            return True
        except IndicatorUpdateError:
            # Surfaced to the caller so the failure notification carries the real cause.
            log_and_echo("ERROR: Indicator update failed, aborting run")
            raise
        except Exception as e:
            log_and_echo(f"ERROR: Failed to run recommendation engine: {e}")
            get_logger().error(f"Failed to run recommendation engine: {e}")
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
        # Configure logger first so data fetch and all setup log to trade.log (not stdout)
        configure_logger("trade", log_level="info", observer=config.observer)

        from algo_trading_engine.data_retriever import DataRetriever
        from algo_trading_engine.options_handler import OptionsHandler
        from algo_trading_engine.backtest._strategy_builder import create_strategy_from_args

        today = datetime.now()

        # Internal: Create options handler (strategy needs option callables before first fetch)
        options_handler = OptionsHandler(
            symbol=config.symbol,
            api_key=config.api_key,
            use_free_tier=config.use_free_tier,
            use_cache=config.use_cache
        )

        # Internal: Extract methods as callables (no imports needed by child repos)
        get_contract_list_for_date = options_handler.get_contract_list_for_date
        get_option_bar = options_handler.get_option_bar
        get_options_chain = options_handler.get_options_chain

        # Internal: Create or use provided strategy (needed to compute warm-up window like backtest)
        if isinstance(config.strategy_type, str):
            strategy = create_strategy_from_args(
                strategy_name=config.strategy_type,
                symbol=config.symbol,
                get_contract_list_for_date=get_contract_list_for_date,
                get_option_bar=get_option_bar,
                get_options_chain=get_options_chain,
                get_current_volumes_for_position=cls.get_current_volumes_for_position,
                compute_exit_price=cls.compute_exit_price,
                options_handler=options_handler,
                stop_loss=config.stop_loss,
                profit_target=config.profit_target
            )
            if strategy is None:
                raise ValueError(f"Failed to create strategy: {config.strategy_type}")
        else:
            strategy = config.strategy_type
            strategy.symbol = config.symbol
            if hasattr(strategy, 'get_contract_list_for_date'):
                strategy.get_contract_list_for_date = get_contract_list_for_date
                strategy.get_option_bar = get_option_bar
                strategy.get_options_chain = get_options_chain
                strategy.get_current_volumes_for_position = cls.get_current_volumes_for_position
                strategy.compute_exit_price = cls.compute_exit_price
            elif hasattr(strategy, 'options_handler'):
                strategy.options_handler = options_handler

        strategy.use_snapshot_for_current_bar = True

        fetch_start_date = compute_paper_trading_fetch_start_date(
            today, strategy, config.bar_interval
        )

        retriever = DataRetriever(
            symbol=config.symbol,
            lstm_start_date=fetch_start_date,
            bar_interval=config.bar_interval,
            use_cache=config.use_cache
        )

        data = retriever.fetch_data_for_period(fetch_start_date)
        if data is None or len(data) == 0:
            raise ValueError(f"Failed to fetch data for {config.symbol}")
        get_logger().info(f"Fetched {len(data)} data points for {config.symbol} from {data.index[0]} to {data.index[-1]}")

        from algo_trading_engine._internal.common.ml_pipeline import (
            is_credit_spread_strategy,
            prepare_credit_spread_backtest_data,
        )

        if is_credit_spread_strategy(config.strategy_type):
            data = prepare_credit_spread_backtest_data(data, retriever, config.symbol)

        strategy.set_data(data, retriever.treasury_rates)
        strategy.warm_up_indicators()

        engine = cls(
            strategy=strategy,
            config=config,
            options_handler=options_handler
        )

        # Inject engine methods into strategy (mirrors BacktestEngine.from_config)
        if hasattr(strategy, 'compute_exit_price'):
            strategy.compute_exit_price = engine.compute_exit_price
        if hasattr(strategy, 'get_current_volumes_for_position'):
            strategy.get_current_volumes_for_position = engine.get_current_volumes_for_position

        strategy.get_rt_option_bar = make_rt_option_bar(options_handler)

        return engine

