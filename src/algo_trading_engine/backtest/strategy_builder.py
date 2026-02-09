"""
Strategy Builder Pattern for backtesting system.

This module provides a builder pattern implementation for creating and configuring
trading strategies with flexible parameter selection and validation.
"""

from abc import ABC, abstractmethod
from typing import Dict, Type, List, Callable

try:
    from ..core.strategy import Strategy
except ImportError:
    # Fallback for direct execution
    from algo_trading_engine.core.strategy import Strategy


class StrategyBuilder(ABC):
    """Abstract base class for strategy builders"""
    
    def __init__(self):
        self.reset()
    
    @abstractmethod
    def reset(self):
        """Reset the builder to initial state"""
        pass

    @abstractmethod
    def set_options_callables(self, get_contract_list_for_date: Callable, get_option_bar: Callable, get_options_chain: Callable, get_current_volumes_for_position: Callable, options_handler=None):
        """Set the options callables (methods from OptionsHandler as callables)"""
        pass
    
    @abstractmethod
    def set_start_date_offset(self, offset: int):
        """Set the start date offset"""
        pass
    
    @abstractmethod
    def set_stop_loss(self, stop_loss: float):
        """Set the stop loss"""
        pass
    
    @abstractmethod
    def set_profit_target(self, profit_target: float):
        """Set the profit target"""
        pass
    
    @abstractmethod
    def build(self) -> Strategy:
        """Build and return the strategy"""
        pass


class CreditSpreadStrategyBuilder(StrategyBuilder):
    """Builder for CreditSpreadStrategy"""
    
    def reset(self):
        self._lstm_model = None
        self._lstm_scaler = None
        self._symbol = None
        self._get_contract_list_for_date = None
        self._get_option_bar = None
        self._get_options_chain = None
        self._options_handler = None
        self._start_date_offset = 0
        self._stop_loss = 0.6
        self._profit_target = None
    
    def set_lstm_model(self, model):
        self._lstm_model = model
        return self
    
    def set_lstm_scaler(self, scaler):
        self._lstm_scaler = scaler
        return self
    
    def set_symbol(self, symbol: str):
        """Set the symbol for options data retrieval"""
        self._symbol = symbol
        return self
    
    def set_options_callables(self, get_contract_list_for_date: Callable, get_option_bar: Callable, get_options_chain: Callable, get_current_volumes_for_position: Callable = None, compute_exit_price: Callable = None, options_handler=None):
        """Set the options callables (methods from OptionsHandler as callables)
        
        Note: get_current_volumes_for_position and compute_exit_price are injected from the engine after creation,
        so they are optional here for backward compatibility.
        """
        self._get_contract_list_for_date = get_contract_list_for_date
        self._get_option_bar = get_option_bar
        self._get_options_chain = get_options_chain
        self._get_current_volumes_for_position = get_current_volumes_for_position
        self._compute_exit_price = compute_exit_price
        self._options_handler = options_handler
        return self
    
    def set_start_date_offset(self, offset: int):
        self._start_date_offset = offset
        return self
    
    def set_stop_loss(self, stop_loss: float):
        self._stop_loss = stop_loss
        return self
    
    def set_profit_target(self, profit_target: float):
        self._profit_target = profit_target
        return self
    
    def build(self) -> Strategy:
        try:
            from ..strategies.credit_spread_minimal import CreditSpreadStrategy
        except ImportError:
            from algo_trading_engine.strategies.credit_spread_minimal import CreditSpreadStrategy
        
        # Symbol is required for CreditSpreadStrategy
        if not self._symbol:
            raise ValueError("Symbol is required for CreditSpreadStrategy. Use set_symbol() to provide it.")
        
        # Load LSTM model and scaler if not already provided
        if self._lstm_model is None or self._lstm_scaler is None:
            try:
                from ..common.functions import get_model_directory, load_lstm_model
            except ImportError:
                from algo_trading_engine.common.functions import get_model_directory, load_lstm_model
            
            model_dir = get_model_directory(symbol=self._symbol)
            try:
                self._lstm_model, self._lstm_scaler = load_lstm_model(model_dir, return_lstm_instance=True)
            except Exception as e:
                raise ValueError(f"Failed to load LSTM model for symbol {self._symbol}: {e}")
        
        strategy = CreditSpreadStrategy(
            get_contract_list_for_date=self._get_contract_list_for_date,
            get_option_bar=self._get_option_bar,
            get_options_chain=self._get_options_chain,
            lstm_model=self._lstm_model,
            lstm_scaler=self._lstm_scaler,
            symbol=self._symbol,
            start_date_offset=self._start_date_offset,
            options_handler=self._options_handler
        )
        
        if self._profit_target:
            strategy.set_profit_target(self._profit_target)
        
        self.reset()
        return strategy


class VelocitySignalMomentumStrategyBuilder(StrategyBuilder):
    """Builder for VelocitySignalMomentumStrategy"""
    
    def reset(self):
        self._symbol = 'SPY'
        self._get_contract_list_for_date = None
        self._get_option_bar = None
        self._get_options_chain = None
        self._start_date_offset = 60
        self._stop_loss = None
        self._profit_target = None
    
    def set_symbol(self, symbol: str):
        """Set the symbol for the strategy"""
        self._symbol = symbol
        return self
    
    def set_options_callables(self, get_contract_list_for_date: Callable, get_option_bar: Callable, get_options_chain: Callable, get_current_volumes_for_position: Callable = None, compute_exit_price: Callable = None, options_handler=None):
        """Set the options callables (methods from OptionsHandler as callables)
        
        Note: get_current_volumes_for_position and compute_exit_price are injected from the engine after creation,
        so they are optional here for backward compatibility.
        """
        self._get_contract_list_for_date = get_contract_list_for_date
        self._get_option_bar = get_option_bar
        self._get_options_chain = get_options_chain
        self._get_current_volumes_for_position = get_current_volumes_for_position
        self._compute_exit_price = compute_exit_price
        return self
    
    def set_start_date_offset(self, offset: int):
        self._start_date_offset = offset
        return self
    
    def set_stop_loss(self, stop_loss: float):
        self._stop_loss = stop_loss
        return self

    def set_profit_target(self, profit_target: float):
        # Not used for this strategy but required by interface
        self._profit_target = profit_target
        return self
    
    def build(self) -> Strategy:
        try:
            from ..strategies.velocity_signal_momentum_strategy import VelocitySignalMomentumStrategy
        except ImportError:
            from algo_trading_engine.strategies.velocity_signal_momentum_strategy import VelocitySignalMomentumStrategy
        
        strategy = VelocitySignalMomentumStrategy(
            get_contract_list_for_date=self._get_contract_list_for_date,
            get_option_bar=self._get_option_bar,
            get_options_chain=self._get_options_chain,
            start_date_offset=self._start_date_offset,
            stop_loss=self._stop_loss,
            profit_target=self._profit_target,
            symbol=self._symbol
        )
        
        self.reset()
        return strategy


class StrategyFactory:
    """Factory for creating strategies using the builder pattern"""
    
    _builders: Dict[str, Type[StrategyBuilder]] = {
        'credit_spread': CreditSpreadStrategyBuilder,
        'velocity_momentum': VelocitySignalMomentumStrategyBuilder,
    }
    
    @classmethod
    def get_available_strategies(cls) -> List[str]:
        """Get list of available strategy names"""
        return list(cls._builders.keys())
    
    @classmethod
    def create_strategy(cls, strategy_name: str, **kwargs) -> Strategy:
        """
        Create a strategy using the builder pattern
        
        Args:
            strategy_name: Name of the strategy to create
            **kwargs: Parameters to configure the strategy
            
        Returns:
            Configured Strategy instance
        """
        if strategy_name not in cls._builders:
            raise ValueError(f"Unknown strategy: {strategy_name}. Available: {cls.get_available_strategies()}")
        
        builder_class = cls._builders[strategy_name]
        builder = builder_class()

        # Apply configuration from kwargs
        for key, value in kwargs.items():
            if hasattr(builder, f'set_{key}'):
                getattr(builder, f'set_{key}')(value)
        
        return builder.build()
    
    @classmethod
    def get_builder(cls, strategy_name: str) -> StrategyBuilder:
        """
        Get a builder instance for manual configuration
        
        Args:
            strategy_name: Name of the strategy
            
        Returns:
            StrategyBuilder instance
        """
        if strategy_name not in cls._builders:
            raise ValueError(f"Unknown strategy: {strategy_name}. Available: {cls.get_available_strategies()}")
        
        return cls._builders[strategy_name]()

def create_strategy_from_args(strategy_name: str, **kwargs):
    """
    Create strategy based on command line argument or configuration
    
    Args:
        strategy_name: Name of the strategy to create
        **kwargs: Additional configuration parameters
            - symbol: Required for credit_spread strategy
            - get_contract_list_for_date: Callable for getting contract list
            - get_option_bar: Callable for getting option bar
            - get_options_chain: Callable for getting options chain
            - get_current_volumes_for_position: Callable for getting current volumes for position
            - compute_exit_price: Callable for computing exit price for position
            - options_handler: Optional, OptionsHandler instance (needed for CreditSpreadStrategy with LSTM)
            - start_date_offset: Optional, defaults to 60
            - stop_loss: Optional
            - profit_target: Optional
        
    Returns:
        Configured Strategy instance or None if creation fails
    """
    try:
        builder = StrategyFactory.get_builder(strategy_name)
        
        # Set common parameters
        if all(k in kwargs for k in ['get_contract_list_for_date', 'get_option_bar', 'get_options_chain']):
            builder.set_options_callables(
                kwargs['get_contract_list_for_date'],
                kwargs['get_option_bar'],
                kwargs['get_options_chain'],
                kwargs.get('get_current_volumes_for_position'),  # Optional: injected from engine
                kwargs.get('compute_exit_price'),  # Optional: injected from engine
                kwargs.get('options_handler')  # Backward compatibility: if strategy still uses options_handler, inject it
            )
        if 'symbol' in kwargs:
            builder.set_symbol(kwargs['symbol'])
        if 'start_date_offset' in kwargs:
            builder.set_start_date_offset(kwargs['start_date_offset'])
        if 'stop_loss' in kwargs and kwargs['stop_loss'] is not None:
            builder.set_stop_loss(kwargs['stop_loss'])
        if 'profit_target' in kwargs and kwargs['profit_target'] is not None:
            builder.set_profit_target(kwargs['profit_target'])
        
        strategy = builder.build()
        return strategy
    except ValueError as e:
        print(f"❌ Strategy creation failed: {e}")
        print(f"Available strategies: {StrategyFactory.get_available_strategies()}")
        return None
