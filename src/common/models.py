from datetime import datetime
from typing import Optional, Dict, Any
from dataclasses import dataclass, field
from enum import Enum
from decimal import Decimal
import pandas as pd

class OptionType(Enum):
    """Enum for option types."""
    CALL = "call"
    PUT = "put"

class MarketStateType(Enum):
    """Enum for market state types identified by HMM"""
    LOW_VOLATILITY_UPTREND = "low_volatility_uptrend"
    MOMENTUM_UPTREND = "momentum_uptrend"
    CONSOLIDATION = "consolidation"
    HIGH_VOLATILITY_DOWNTREND = "high_volatility_downtrend"
    HIGH_VOLATILITY_RALLY = "high_volatility_rally"

class SignalType(Enum):
    """Enum for trading signal types"""
    HOLD = "hold"
    CALL_CREDIT_SPREAD = "call_credit_spread"
    PUT_CREDIT_SPREAD = "put_credit_spread"
    LONG_STOCK = "long_stock"
    LONG_CALL = "long_call"
    SHORT_CALL = "short_call"
    LONG_PUT = "long_put"
    SHORT_PUT = "short_put"

@dataclass(frozen=True)
class Option:
    """
    Immutable Data Transfer Object for an individual option contract, matching the cache format.
    """
    ticker: str
    symbol: str
    strike: float
    expiration: str  # Always store as string for cache compatibility
    option_type: OptionType
    last_price: float
    bid: Optional[float] = None
    ask: Optional[float] = None
    mid_price: Optional[float] = None
    volume: Optional[int] = None
    open_interest: Optional[int] = None
    implied_volatility: Optional[float] = None
    delta: Optional[float] = None
    gamma: Optional[float] = None
    theta: Optional[float] = None
    vega: Optional[float] = None
    moneyness: Optional[float] = None

    def __post_init__(self):
        # Ensure option_type is an OptionType enum
        if isinstance(self.option_type, str):
            object.__setattr__(self, 'option_type', OptionType(self.option_type))
        elif not isinstance(self.option_type, OptionType):
            raise ValueError(f"option_type must be a string or OptionType enum, got {type(self.option_type)}")
        
        # Ensure expiration is a string
        if isinstance(self.expiration, (datetime,)):
            object.__setattr__(self, 'expiration', self.expiration.strftime('%Y-%m-%d'))
        
        # Calculate mid_price if not provided but bid/ask are available
        if self.mid_price is None and self.bid is not None and self.ask is not None:
            object.__setattr__(self, 'mid_price', (self.bid + self.ask) / 2)
        
        # Validate required fields
        if self.strike <= 0:
            raise ValueError("Strike price must be positive")
        if self.last_price < 0:
            raise ValueError("Last price cannot be negative")
        if self.volume is not None and self.volume < 0:
            raise ValueError("Volume cannot be negative")
        if self.open_interest is not None and self.open_interest < 0:
            raise ValueError("Open interest cannot be negative")

    @property
    def is_call(self) -> bool:
        return self.option_type == OptionType.CALL

    @property
    def is_put(self) -> bool:
        return self.option_type == OptionType.PUT

    @property
    def is_atm(self) -> bool:
        if self.moneyness is None:
            return False
        return 0.95 <= self.moneyness <= 1.05

    @property
    def is_itm(self) -> bool:
        if self.moneyness is None:
            return False
        if self.is_call:
            return self.moneyness > 1.0
        else:
            return self.moneyness < 1.0

    @property
    def is_otm(self) -> bool:
        if self.moneyness is None:
            return False
        if self.is_call:
            return self.moneyness < 1.0
        else:
            return self.moneyness > 1.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'ticker': self.ticker,
            'strike': self.strike,
            'expiration': self.expiration,
            'type': self.option_type.value,
            'symbol': self.symbol,
            'volume': self.volume,
            'open_interest': self.open_interest,
            'implied_volatility': self.implied_volatility,
            'delta': self.delta,
            'gamma': self.gamma,
            'theta': self.theta,
            'vega': self.vega,
            'last_price': self.last_price,
            'bid': self.bid,
            'ask': self.ask,
            'mid_price': self.mid_price,
            'moneyness': self.moneyness,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Option':
        """Create from dictionary for deserialization"""
        return cls(
            ticker=data.get('ticker', ''),
            symbol=data['symbol'],
            strike=data['strike'],
            expiration=data['expiration'],
            option_type=data['type'],
            last_price=data['last_price'],
            bid=data.get('bid'),
            ask=data.get('ask'),
            mid_price=data.get('mid_price'),
            volume=data.get('volume'),
            open_interest=data.get('open_interest'),
            delta=data.get('delta'),
            gamma=data.get('gamma'),
            theta=data.get('theta'),
            vega=data.get('vega'),
            implied_volatility=data.get('implied_volatility'),
            moneyness=data.get('moneyness'),
        )

    def __eq__(self, other) -> bool:
        """
        Check if two options are equal based on key attributes.
        
        Two options are considered equal if they have:
        - Same ticker
        - Same symbol
        - Same strike price
        - Same expiration date
        - Same option type
        
        Args:
            other: Another Option object to compare against
            
        Returns:
            bool: True if options are equal, False otherwise
        """
        if not isinstance(other, Option):
            return False
            
        return (self.ticker == other.ticker and
                self.symbol == other.symbol and
                self.strike == other.strike and
                self.expiration == other.expiration and
                self.option_type == other.option_type)

    def __str__(self) -> str:
        return f"{self.symbol} {self.option_type.value.upper()} {self.strike} @ {self.last_price:.2f}"

    def __repr__(self) -> str:
        return (f"Option(symbol='{self.symbol}', strike={self.strike}, "
                f"expiration={self.expiration}, "
                f"type={self.option_type.value}, price={self.last_price:.2f})")

    @classmethod
    def from_contract_and_bar(cls, contract, bar) -> 'Option':
        """
        Create an Option from an OptionContractDTO and OptionBarDTO.
        
        Args:
            contract: The option contract metadata
            bar: The option bar/price data
            
        Returns:
            Option: Created Option instance
        """
        from src.common.options_dtos import OptionContractDTO, OptionBarDTO
        
        # Convert contract type from common.models.OptionType to the same module's OptionType
        option_type = OptionType.CALL if contract.contract_type.value == 'call' else OptionType.PUT
        
        return cls(
            ticker=contract.ticker,
            symbol=contract.underlying_ticker,
            strike=float(contract.strike_price.value),
            expiration=str(contract.expiration_date),
            option_type=option_type,
            last_price=float(bar.close_price),
            volume=bar.volume,
            # Use VWAP as mid_price if available
            mid_price=float(bar.volume_weighted_avg_price) if bar.volume_weighted_avg_price else None
        )

@dataclass(frozen=True)
class OptionChain:
    """
    Immutable Data Transfer Object for an option chain matching the cache format.
    Only 'calls' and 'puts' lists are required.
    """
    calls: tuple[Option, ...] = field(default_factory=tuple)
    puts: tuple[Option, ...] = field(default_factory=tuple)
    # Optional metadata for in-memory use, not present in cache
    underlying_symbol: Optional[str] = None
    expiration_date: Optional[str] = None
    current_price: Optional[float] = None
    date: Optional[str] = None
    source: Optional[str] = None

    def __post_init__(self):
        # Convert lists to tuples for immutability and ensure all items are Option instances
        calls_list = [opt if isinstance(opt, Option) else Option.from_dict(opt) for opt in self.calls]
        puts_list = [opt if isinstance(opt, Option) else Option.from_dict(opt) for opt in self.puts]
        
        object.__setattr__(self, 'calls', tuple(calls_list))
        object.__setattr__(self, 'puts', tuple(puts_list))

    @property
    def total_calls(self) -> int:
        return len(self.calls)

    @property
    def total_puts(self) -> int:
        return len(self.puts)

    @property
    def total_options(self) -> int:
        return self.total_calls + self.total_puts

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'calls': [opt.to_dict() for opt in self.calls],
            'puts': [opt.to_dict() for opt in self.puts],
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'OptionChain':
        """Create from dictionary for deserialization"""
        return cls(
            calls=[Option.from_dict(opt) for opt in data.get('calls', [])],
            puts=[Option.from_dict(opt) for opt in data.get('puts', [])],
        )
    
    @classmethod
    def from_dict_w_options(cls, data: Dict[Option, Any]) -> 'OptionChain':
        """Create from dictionary with existing Option objects"""
        return cls(
            calls=data.get('calls', []),
            puts=data.get('puts', []),
        )
    
    def get_option_data_for_option(self, option: Option) -> Optional[Option]:
        """
        Find the current option data for a given option by matching strike and expiration.
        
        Args:
            option: The option to find current data for
            
        Returns:
            Option: Current option data if found, None otherwise
        """
        # Determine which list to search based on option type
        options_list = self.calls if option.is_call else self.puts
        
        # Find the matching option by strike and expiration
        for current_option in options_list:
            if (current_option.strike == option.strike and 
                current_option.expiration == option.expiration):
                return current_option
        
        return None

    def add_option(self, option: Option) -> 'OptionChain':
        """
        Create a new OptionChain with the added option.
        
        Args:
            option: The option to add
            
        Returns:
            OptionChain: New OptionChain with the added option
        """
        if option.is_call:
            new_calls = self.calls + (option,)
            return OptionChain(
                calls=new_calls,
                puts=self.puts,
                underlying_symbol=self.underlying_symbol,
                expiration_date=self.expiration_date,
                current_price=self.current_price,
                date=self.date,
                source=self.source
            )
        else:
            new_puts = self.puts + (option,)
            return OptionChain(
                calls=self.calls,
                puts=new_puts,
                underlying_symbol=self.underlying_symbol,
                expiration_date=self.expiration_date,
                current_price=self.current_price,
                date=self.date,
                source=self.source
            )

    def __str__(self) -> str:
        """Return a descriptive string listing all options in the chain."""
        result = f"OptionChain({self.total_calls} calls, {self.total_puts} puts)"
        
        if self.calls:
            result += "\n  Calls:"
            for call in self.calls:
                result += f"\n    {call}"
        
        if self.puts:
            result += "\n  Puts:"
            for put in self.puts:
                result += f"\n    {put}"
        
        return result

    def __repr__(self) -> str:
        return f"OptionChain(calls={self.total_calls}, puts={self.total_puts})"

@dataclass(frozen=True)
class TreasuryRates:
    """
    Value Object representing treasury rates data.
    
    This encapsulates treasury yield data and provides domain-specific
    methods for accessing risk-free rates.
    """
    rates_data: pd.DataFrame  # The underlying treasury rates DataFrame
    
    def __post_init__(self):
        """Validate treasury rates data"""
        if self.rates_data is None or len(self.rates_data) == 0:
            raise ValueError("Treasury rates data cannot be empty")
        
        # Validate required columns exist
        required_columns = ['IRX_1Y', 'TNX_10Y']
        missing_columns = [col for col in required_columns if col not in self.rates_data.columns]
        if missing_columns:
            raise ValueError(f"Missing required treasury rate columns: {missing_columns}")
    
    def __eq__(self, other):
        """Value-based equality for TreasuryRates"""
        if not isinstance(other, TreasuryRates):
            return False
        return self.rates_data.equals(other.rates_data)
    
    def __hash__(self):
        """Hash based on the data content"""
        # Convert DataFrame to a hashable format
        data_hash = hash((
            tuple(self.rates_data.index),
            tuple(self.rates_data.columns),
            tuple(self.rates_data.values.flatten())
        ))
        return hash((TreasuryRates, data_hash))
    
    def get_risk_free_rate(self, date: datetime) -> Decimal:
        """
        Get the risk-free rate for a specific date.
        
        Args:
            date: The date to get the risk-free rate for
            
        Returns:
            Decimal: Risk-free rate (1-year Treasury yield)
        """
        try:
            # Try to get the 1-year rate (IRX_1Y) for the specific date
            if date in self.rates_data.index:
                rate = self.rates_data.loc[date, 'IRX_1Y']
                return Decimal(str(rate))
            
            # If exact date not found, use the closest available date
            available_dates = self.rates_data.index
            if len(available_dates) > 0:
                # Find the closest date
                closest_date = min(available_dates, key=lambda x: abs((x - date).days))
                rate = self.rates_data.loc[closest_date, 'IRX_1Y']
                return Decimal(str(rate))
                
        except (KeyError, IndexError, ValueError):
            pass
            
        return Decimal('0.0')  # Default fallback
    
    def get_10_year_rate(self, date: datetime) -> Decimal:
        """
        Get the 10-year Treasury rate for a specific date.
        
        Args:
            date: The date to get the 10-year rate for
            
        Returns:
            Decimal: 10-year Treasury yield
        """
        try:
            if date in self.rates_data.index:
                rate = self.rates_data.loc[date, 'TNX_10Y']
                return Decimal(str(rate))
            
            available_dates = self.rates_data.index
            if len(available_dates) > 0:
                closest_date = min(available_dates, key=lambda x: abs((x - date).days))
                rate = self.rates_data.loc[closest_date, 'TNX_10Y']
                return Decimal(str(rate))
                
        except (KeyError, IndexError, ValueError):
            pass
            
        return Decimal('0.0')
    
    def get_date_range(self) -> tuple[datetime, datetime]:
        """
        Get the date range covered by this treasury rates data.
        
        Returns:
            tuple: (start_date, end_date)
        """
        return self.rates_data.index.min(), self.rates_data.index.max()
    
    def is_empty(self) -> bool:
        """Check if treasury rates data is empty"""
        return len(self.rates_data) == 0

@dataclass(frozen=True)
class MarketState:
    """Represents a market state with characteristics"""
    state_type: MarketStateType
    volatility: Decimal
    average_return: Decimal
    confidence: Decimal
    
    def __post_init__(self):
        if not 0 <= self.confidence <= 1:
            raise ValueError("Confidence must be between 0 and 1")
        if self.volatility < 0:
            raise ValueError("Volatility cannot be negative")
    
    def is_bullish(self) -> bool:
        """Check if this market state is generally bullish"""
        return self.state_type in [
            MarketStateType.LOW_VOLATILITY_UPTREND,
            MarketStateType.MOMENTUM_UPTREND,
            MarketStateType.HIGH_VOLATILITY_RALLY
        ]
    
    def is_bearish(self) -> bool:
        """Check if this market state is generally bearish"""
        return self.state_type == MarketStateType.HIGH_VOLATILITY_DOWNTREND
    
    def is_consolidation(self) -> bool:
        """Check if this market state represents consolidation"""
        return self.state_type == MarketStateType.CONSOLIDATION

@dataclass(frozen=True)
class TradingSignal:
    """Represents a trading signal with strategy and confidence"""
    signal_type: SignalType
    confidence: Decimal
    ticker: str
    expiration_date: Optional[str] = None
    strike_prices: Optional[tuple[Decimal, Decimal]] = None
    
    def __post_init__(self):
        if not 0 <= self.confidence <= 1:
            raise ValueError("Confidence must be between 0 and 1")
        if self.signal_type != SignalType.HOLD and not self.expiration_date:
            raise ValueError("Expiration date required for option strategies")
    
    def is_option_strategy(self) -> bool:
        """Check if this signal represents an option strategy"""
        return self.signal_type in [
            SignalType.CALL_CREDIT_SPREAD, 
            SignalType.PUT_CREDIT_SPREAD,
            SignalType.LONG_CALL,
            SignalType.SHORT_CALL,
            SignalType.LONG_PUT,
            SignalType.SHORT_PUT
        ]
    
    def is_credit_spread(self) -> bool:
        """Check if this signal represents a credit spread strategy"""
        return self.signal_type in [
            SignalType.CALL_CREDIT_SPREAD,
            SignalType.PUT_CREDIT_SPREAD
        ]

@dataclass(frozen=True)
class PriceRange:
    """Represents a price range with validation"""
    low: Decimal
    high: Decimal
    
    def __post_init__(self):
        if self.low > self.high:
            raise ValueError("Low price cannot be greater than high price")
        if self.low < 0:
            raise ValueError("Price cannot be negative")
    
    def contains(self, price: Decimal) -> bool:
        """Check if a price falls within this range"""
        return self.low <= price <= self.high
    
    def spread(self) -> Decimal:
        """Calculate the spread between high and low prices"""
        return self.high - self.low
    
    def midpoint(self) -> Decimal:
        """Calculate the midpoint of the price range"""
        return (self.low + self.high) / 2

@dataclass(frozen=True)
class Volatility:
    """Represents volatility with validation"""
    value: Decimal
    period: int  # days
    
    def __post_init__(self):
        if self.value < 0:
            raise ValueError("Volatility cannot be negative")
        if self.period <= 0:
            raise ValueError("Period must be positive")
    
    def is_high(self) -> bool:
        """Check if volatility is considered high (>30%)"""
        return self.value > Decimal('0.3')
    
    def is_low(self) -> bool:
        """Check if volatility is considered low (<10%)"""
        return self.value < Decimal('0.1')
    
    def is_moderate(self) -> bool:
        """Check if volatility is moderate (10-30%)"""
        return not self.is_high() and not self.is_low()
