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

    def __str__(self) -> str:
        return f"{self.symbol} {self.option_type.value.upper()} {self.strike} @ {self.last_price:.2f}"

    def __repr__(self) -> str:
        return (f"Option(symbol='{self.symbol}', strike={self.strike}, "
                f"expiration={self.expiration}, "
                f"type={self.option_type.value}, price={self.last_price:.2f})")

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
