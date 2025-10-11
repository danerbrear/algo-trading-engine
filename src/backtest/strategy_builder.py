"""
Strategy Builder Pattern for backtesting system.

This module provides a builder pattern implementation for creating and configuring
trading strategies with flexible parameter selection and validation.
"""

from abc import ABC, abstractmethod
from typing import Dict, Type, List
try:
    from .models import Strategy
    from ..model.options_handler import OptionsHandler
except ImportError:
    # Fallback for direct execution
    from src.backtest.models import Strategy
    from src.model.options_handler import OptionsHandler


class StrategyBuilder(ABC):
    """Abstract base class for strategy builders"""
    
    def __init__(self):
        self.reset()
    
    @abstractmethod
    def reset(self):
        """Reset the builder to initial state"""
        pass
    
    @abstractmethod
    def set_lstm_model(self, model):
        """Set the LSTM model"""
        pass
    
    @abstractmethod
    def set_lstm_scaler(self, scaler):
        """Set the LSTM scaler"""
        pass
    
    @abstractmethod
    def set_options_handler(self, handler: OptionsHandler):
        """Set the options handler"""
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
    
    def set_options_handler(self, handler: OptionsHandler):
        self._options_handler = handler
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
        if not all([self._lstm_model, self._lstm_scaler, self._options_handler]):
            raise ValueError("Missing required parameters: lstm_model, lstm_scaler, options_handler")
        
        try:
            from ..strategies.credit_spread_minimal import CreditSpreadStrategy
        except ImportError:
            from src.strategies.credit_spread_minimal import CreditSpreadStrategy
        
        strategy = CreditSpreadStrategy(
            lstm_model=self._lstm_model,
            lstm_scaler=self._lstm_scaler,
            options_handler=self._options_handler,
            start_date_offset=self._start_date_offset
        )
        
        if self._profit_target:
            strategy.set_profit_target(self._profit_target)
        
        self.reset()
        return strategy


class VelocitySignalMomentumStrategyBuilder(StrategyBuilder):
    """Builder for VelocitySignalMomentumStrategy"""
    
    def reset(self):
        self._symbol = 'SPY'
        self._start_date_offset = 60
        self._lstm_model = None
        self._lstm_scaler = None
        self._options_handler = None
        self._stop_loss = None
        self._profit_target = None
    
    def set_symbol(self, symbol: str):
        self._symbol = symbol
        return self
    
    def set_start_date_offset(self, offset: int):
        self._start_date_offset = offset
        return self
    
    def set_lstm_model(self, model):
        # Not used for this strategy but required by interface
        self._lstm_model = model
        return self
    
    def set_lstm_scaler(self, scaler):
        # Not used for this strategy but required by interface
        self._lstm_scaler = scaler
        return self
    
    def set_options_handler(self, handler: OptionsHandler):
        self._options_handler = handler
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
            from src.strategies.velocity_signal_momentum_strategy import VelocitySignalMomentumStrategy
        
        strategy = VelocitySignalMomentumStrategy(
            options_handler=self._options_handler,
            start_date_offset=self._start_date_offset,
            stop_loss=self._stop_loss
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

def create_strategy_from_args(strategy_name: str, lstm_model, lstm_scaler, options_handler, **kwargs):
    """
    Create strategy based on command line argument or configuration
    
    Args:
        strategy_name: Name of the strategy to create
        lstm_model: LSTM model instance
        lstm_scaler: LSTM scaler instance
        options_handler: Options handler instance
        **kwargs: Additional configuration parameters
        
    Returns:
        Configured Strategy instance or None if creation fails
    """
    try:
        strategy = StrategyFactory.create_strategy(
            strategy_name=strategy_name,
            lstm_model=lstm_model,
            lstm_scaler=lstm_scaler,
            options_handler=options_handler,
            start_date_offset=kwargs.get('start_date_offset', 60),
            stop_loss=kwargs.get('stop_loss', None),
            profit_target=kwargs.get('profit_target', None)
        )
        return strategy
    except ValueError as e:
        print(f"❌ Strategy creation failed: {e}")
        print(f"Available strategies: {StrategyFactory.get_available_strategies()}")
        return None
