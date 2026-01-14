"""
Public Data Transfer Objects (DTOs) for Options data.

These DTOs are part of the public API and can be imported via:
    from algo_trading_engine.dto import OptionContractDTO, OptionBarDTO, etc.

Note: StrikePrice and ExpirationDate are Value Objects, not DTOs,
and are located in algo_trading_engine.vo.value_objects
"""

from dataclasses import dataclass, field
from datetime import datetime, date
from decimal import Decimal
from typing import Optional, List, Dict, Any
import re

from algo_trading_engine.vo import StrikePrice, ExpirationDate
from algo_trading_engine.common.models import OptionType


@dataclass(frozen=True)
class OptionContractDTO:
    """
    Data Transfer Object for option contract metadata.
    
    Based on Polygon.io /v3/reference/options/contracts API response.
    """
    ticker: str
    underlying_ticker: str
    contract_type: OptionType
    strike_price: StrikePrice
    expiration_date: ExpirationDate
    exercise_style: str = "american"
    shares_per_contract: int = 100
    primary_exchange: Optional[str] = None
    cfi: Optional[str] = None
    additional_underlyings: Optional[List[Dict[str, Any]]] = None
    
    def __post_init__(self):
        # Validate ticker format
        if not re.match(r'^O:[A-Z0-9]+$', self.ticker):
            raise ValueError(f"Invalid ticker format: {self.ticker}")
        
        # Validate underlying ticker
        if not re.match(r'^[A-Z0-9]+$', self.underlying_ticker):
            raise ValueError(f"Invalid underlying ticker: {self.underlying_ticker}")
        
        # Validate exercise style
        if self.exercise_style not in ["american", "european"]:
            raise ValueError(f"Invalid exercise style: {self.exercise_style}")
        
        # Validate shares per contract
        if self.shares_per_contract <= 0:
            raise ValueError("Shares per contract must be positive")
    
    def is_call(self) -> bool:
        """Check if this is a call option."""
        return self.contract_type == OptionType.CALL
    
    def is_put(self) -> bool:
        """Check if this is a put option."""
        return self.contract_type == OptionType.PUT
    
    def days_to_expiration(self, current_date: Optional['date'] = None) -> int:
        """Get days to expiration."""
        return self.expiration_date.days_to_expiration(current_date)
    
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "ticker": self.ticker,
            "underlying_ticker": self.underlying_ticker,
            "contract_type": self.contract_type.value,
            "strike_price": float(self.strike_price.value),
            "expiration_date": str(self.expiration_date),
            "exercise_style": self.exercise_style,
            "shares_per_contract": self.shares_per_contract,
            "primary_exchange": self.primary_exchange,
            "cfi": self.cfi,
            "additional_underlyings": self.additional_underlyings
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'OptionContractDTO':
        """Create from dictionary."""
        return cls(
            ticker=data["ticker"],
            underlying_ticker=data["underlying_ticker"],
            contract_type=OptionType(data["contract_type"]),
            strike_price=StrikePrice(Decimal(str(data["strike_price"]))),
            expiration_date=ExpirationDate(datetime.strptime(data["expiration_date"], '%Y-%m-%d').date()),
            exercise_style=data.get("exercise_style", "american"),
            shares_per_contract=data.get("shares_per_contract", 100),
            primary_exchange=data.get("primary_exchange"),
            cfi=data.get("cfi"),
            additional_underlyings=data.get("additional_underlyings")
        )


@dataclass(frozen=True)
class OptionBarDTO:
    """
    Data Transfer Object for option bar/price data.
    
    Based on Polygon.io /v2/aggs/ticker/{optionsTicker}/range/{multiplier}/{timespan}/{from}/{to} API response.
    """
    ticker: str
    timestamp: datetime
    open_price: Decimal
    high_price: Decimal
    low_price: Decimal
    close_price: Decimal
    volume: int
    volume_weighted_avg_price: Decimal
    number_of_transactions: int
    adjusted: bool = True
    
    def __post_init__(self):
        # Validate prices
        for price_name, price in [
            ("open", self.open_price),
            ("high", self.high_price),
            ("low", self.low_price),
            ("close", self.close_price),
            ("vwap", self.volume_weighted_avg_price)
        ]:
            if price < 0:
                raise ValueError(f"{price_name} price cannot be negative")
        
        # Validate high >= low
        if self.high_price < self.low_price:
            raise ValueError("High price cannot be less than low price")
        
        # Validate volume
        if self.volume < 0:
            raise ValueError("Volume cannot be negative")
        
        # Validate transactions
        if self.number_of_transactions < 0:
            raise ValueError("Number of transactions cannot be negative")
    
    def price_range(self) -> Decimal:
        """Calculate the price range (high - low)."""
        return self.high_price - self.low_price
    
    def is_green_candle(self) -> bool:
        """Check if this is a green (bullish) candle."""
        return self.close_price > self.open_price
    
    def is_red_candle(self) -> bool:
        """Check if this is a red (bearish) candle."""
        return self.close_price < self.open_price
    
    def is_doji(self, tolerance: Decimal = Decimal('0.01')) -> bool:
        """Check if this is a doji candle (open ≈ close)."""
        return abs(self.close_price - self.open_price) <= tolerance
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "ticker": self.ticker,
            "timestamp": self.timestamp.isoformat(),
            "open_price": float(self.open_price),
            "high_price": float(self.high_price),
            "low_price": float(self.low_price),
            "close_price": float(self.close_price),
            "volume": self.volume,
            "volume_weighted_avg_price": float(self.volume_weighted_avg_price),
            "number_of_transactions": self.number_of_transactions,
            "adjusted": self.adjusted
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'OptionBarDTO':
        """Create from dictionary."""
        return cls(
            ticker=data["ticker"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            open_price=Decimal(str(data["open_price"])),
            high_price=Decimal(str(data["high_price"])),
            low_price=Decimal(str(data["low_price"])),
            close_price=Decimal(str(data["close_price"])),
            volume=data["volume"],
            volume_weighted_avg_price=Decimal(str(data["volume_weighted_avg_price"])),
            number_of_transactions=data["number_of_transactions"],
            adjusted=data.get("adjusted", True)
        )


@dataclass(frozen=True)
class StrikeRangeDTO:
    """
    Data Transfer Object for strike price filtering criteria.
    """
    min_strike: Optional[StrikePrice] = None
    max_strike: Optional[StrikePrice] = None
    target_strike: Optional[StrikePrice] = None
    tolerance: Optional[Decimal] = None
    
    def __post_init__(self):
        # Validate range
        if self.min_strike is not None and self.max_strike is not None:
            if self.min_strike.value > self.max_strike.value:
                raise ValueError("Min strike cannot be greater than max strike")
        
        # Validate tolerance
        if self.tolerance is not None and self.tolerance < 0:
            raise ValueError("Tolerance cannot be negative")
    
    def contains_strike(self, strike: StrikePrice) -> bool:
        """Check if a strike price falls within this range."""
        if self.min_strike is not None and strike.value < self.min_strike.value:
            return False
        if self.max_strike is not None and strike.value > self.max_strike.value:
            return False
        return True
    
    def is_target_within_tolerance(self, strike: StrikePrice) -> bool:
        """Check if a strike is within tolerance of target."""
        if self.target_strike is None or self.tolerance is None:
            return False
        return abs(strike.value - self.target_strike.value) <= self.tolerance


@dataclass(frozen=True)
class ExpirationRangeDTO:
    """
    Data Transfer Object for expiration date filtering criteria.
    """
    min_days: Optional[int] = None
    max_days: Optional[int] = None
    target_date: Optional[ExpirationDate] = None
    current_date: Optional['date'] = None
    
    def __post_init__(self):
        # Validate range
        if self.min_days is not None and self.max_days is not None:
            if self.min_days > self.max_days:
                raise ValueError("Min days cannot be greater than max days")
        
        # Validate days
        if self.min_days is not None and self.min_days < 0:
            raise ValueError("Min days cannot be negative")
        if self.max_days is not None and self.max_days < 0:
            raise ValueError("Max days cannot be negative")
    
    def contains_expiration(self, expiration: ExpirationDate, current_date: Optional['date'] = None) -> bool:
        """Check if an expiration date falls within this range."""
        if current_date is None:
            current_date = self.current_date or date.today()
        
        days_to_exp = expiration.days_to_expiration(current_date)
        
        if self.min_days is not None and days_to_exp < self.min_days:
            return False
        if self.max_days is not None and days_to_exp > self.max_days:
            return False
        return True
    
    def contains_expiration_with_tolerance(self, expiration: ExpirationDate, current_date: Optional['date'] = None) -> bool:
        """Check if an expiration is within tolerance of target date."""
        if self.target_date is None or current_date is None:
            return False
        
        target_days = self.target_date.days_to_expiration(current_date)
        expiration_days = expiration.days_to_expiration(current_date)
        
        # Allow 1 day tolerance
        return abs(target_days - expiration_days) <= 1


@dataclass(frozen=True)
class OptionsChainDTO:
    """
    Data Transfer Object for a complete option chain.
    """
    underlying_symbol: str
    current_price: Decimal
    date: date
    contracts: List[OptionContractDTO] = field(default_factory=list)
    bars: Dict[str, OptionBarDTO] = field(default_factory=dict)
    
    def __post_init__(self):
        # Validate current price
        if self.current_price <= 0:
            raise ValueError("Current price must be positive")
        
        # Validate underlying symbol
        if not re.match(r'^[A-Z0-9]+$', self.underlying_symbol):
            raise ValueError(f"Invalid underlying symbol: {self.underlying_symbol}")
    
    def get_calls(self) -> List[OptionContractDTO]:
        """Get all call contracts."""
        return [contract for contract in self.contracts if contract.is_call()]
    
    def get_puts(self) -> List[OptionContractDTO]:
        """Get all put contracts."""
        return [contract for contract in self.contracts if contract.is_put()]
    
    def get_contracts_by_strike(self, strike_range: StrikeRangeDTO) -> List[OptionContractDTO]:
        """Get contracts within strike range."""
        return [contract for contract in self.contracts if strike_range.contains_strike(contract.strike_price)]
    
    def get_contracts_by_expiration(self, expiration_range: ExpirationRangeDTO) -> List[OptionContractDTO]:
        """Get contracts within expiration range."""
        return [contract for contract in self.contracts if expiration_range.contains_expiration(contract.expiration_date)]
    
    def get_atm_contracts(self, tolerance: Decimal = Decimal('0.01')) -> List[OptionContractDTO]:
        """Get at-the-money contracts."""
        return [contract for contract in self.contracts if contract.strike_price.is_atm(self.current_price, tolerance)]
    
    def get_bar_for_contract(self, contract: OptionContractDTO) -> Optional[OptionBarDTO]:
        """Get bar data for a specific contract."""
        return self.bars.get(contract.ticker)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "underlying_symbol": self.underlying_symbol,
            "current_price": float(self.current_price),
            "date": self.date.isoformat(),
            "contracts": [contract.to_dict() for contract in self.contracts],
            "bars": {ticker: bar.to_dict() for ticker, bar in self.bars.items()}
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'OptionsChainDTO':
        """Create from dictionary."""
        return cls(
            underlying_symbol=data["underlying_symbol"],
            current_price=Decimal(str(data["current_price"])),
            date=datetime.strptime(data["date"], '%Y-%m-%d').date(),
            contracts=[OptionContractDTO.from_dict(contract) for contract in data.get("contracts", [])],
            bars={ticker: OptionBarDTO.from_dict(bar) for ticker, bar in data.get("bars", {}).items()}
        )
