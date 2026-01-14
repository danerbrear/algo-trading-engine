"""
Public Value Objects (VOs) for the Algo Trading Engine.

These VOs are part of the public API and can be imported via:
    from algo_trading_engine.vo import StrikePrice, ExpirationDate, MarketState, etc.

Value Objects represent domain concepts with validation and behavior.
They are immutable and provide value-based equality.
"""

from dataclasses import dataclass
from datetime import date, timedelta
from decimal import Decimal
from typing import Optional

from algo_trading_engine.enums import OptionType, MarketStateType, SignalType


@dataclass(frozen=True)
class StrikePrice:
    """
    Value Object representing a strike price with validation.
    
    This encapsulates strike price logic and provides domain-specific
    methods for strike price operations.
    """
    value: Decimal
    
    def __post_init__(self):
        if self.value <= 0:
            raise ValueError("Strike price must be positive")
        if self.value > Decimal('10000'):  # Reasonable upper limit
            raise ValueError("Strike price cannot exceed $10,000")
    
    def __str__(self) -> str:
        return f"${self.value}"
    
    def __repr__(self) -> str:
        return f"StrikePrice({self.value})"
    
    def is_atm(self, current_price: Decimal, tolerance: Decimal = Decimal('0.01')) -> bool:
        """Check if this strike is at-the-money within tolerance."""
        return abs(self.value - current_price) <= tolerance
    
    def is_itm(self, current_price: Decimal, option_type: OptionType) -> bool:
        """Check if this strike is in-the-money."""
        if option_type == OptionType.CALL:
            return self.value < current_price
        else:  # PUT
            return self.value > current_price
    
    def is_otm(self, current_price: Decimal, option_type: OptionType) -> bool:
        """Check if this strike is out-of-the-money."""
        return not self.is_itm(current_price, option_type) and not self.is_atm(current_price)


@dataclass(frozen=True)
class ExpirationDate:
    """
    Value Object representing an expiration date with validation.
    
    This encapsulates expiration date logic and provides domain-specific
    methods for expiration date operations.
    """
    date: date
    
    def __post_init__(self):
        pass
    
    def __str__(self) -> str:
        return self.date.strftime('%Y-%m-%d')
    
    def __repr__(self) -> str:
        return f"ExpirationDate({self.date})"
    
    def days_to_expiration(self, current_date: Optional['date'] = None) -> int:
        """Calculate days to expiration from current date."""
        if current_date is None:
            current_date = date.today()
        return (self.date - current_date).days
    
    def is_weekly(self) -> bool:
        """Check if this is a weekly expiration (Friday)."""
        return self.date.weekday() == 4  # Friday
    
    def is_monthly(self) -> bool:
        """Check if this is a monthly expiration (third Friday)."""
        if not self.is_weekly():
            return False
        # Third Friday of the month
        first_day = self.date.replace(day=1)
        first_friday = first_day + timedelta(days=(4 - first_day.weekday()) % 7)
        third_friday = first_friday + timedelta(days=14)
        return self.date == third_friday


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
