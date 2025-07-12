from datetime import datetime
from typing import Optional, Dict, Any
from dataclasses import dataclass, field
from enum import Enum

class OptionType(Enum):
    """Enum for option types."""
    CALL = "call"
    PUT = "put"

@dataclass
class Option:
    """
    Data Transfer Object for an individual option contract, matching the cache format.
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
            self.option_type = OptionType(self.option_type)
        elif not isinstance(self.option_type, OptionType):
            raise ValueError(f"option_type must be a string or OptionType enum, got {type(self.option_type)}")
        # Ensure expiration is a string
        if isinstance(self.expiration, (datetime,)):
            self.expiration = self.expiration.strftime('%Y-%m-%d')
        # Calculate mid_price if not provided but bid/ask are available
        if self.mid_price is None and self.bid is not None and self.ask is not None:
            self.mid_price = (self.bid + self.ask) / 2

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


@dataclass
class OptionChain:
    """
    Data Transfer Object for an option chain matching the cache format.
    Only 'calls' and 'puts' lists are required.
    """
    calls: list[Option] = field(default_factory=list)
    puts: list[Option] = field(default_factory=list)
    # Optional metadata for in-memory use, not present in cache
    underlying_symbol: Optional[str] = None
    expiration_date: Optional[str] = None
    current_price: Optional[float] = None
    date: Optional[str] = None
    source: Optional[str] = None

    def __post_init__(self):
        # Ensure all items in calls and puts are Option instances
        self.calls = [opt if isinstance(opt, Option) else Option.from_dict(opt) for opt in self.calls]
        self.puts = [opt if isinstance(opt, Option) else Option.from_dict(opt) for opt in self.puts]

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
        return {
            'calls': [opt.to_dict() for opt in self.calls],
            'puts': [opt.to_dict() for opt in self.puts],
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'OptionChain':
        return cls(
            calls=[Option.from_dict(opt) for opt in data.get('calls', [])],
            puts=[Option.from_dict(opt) for opt in data.get('puts', [])],
        )
    
    @classmethod
    def from_dict_w_options(cls, data: Dict[Option, Any]) -> 'OptionChain':
        return cls(
            calls=data.get('calls', []),
            puts=data.get('puts', []),
        )
    
    def get_option_data_for_option(self, option: Option) -> Option:
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
