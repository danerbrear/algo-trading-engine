from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Any, Dict, Optional
from vollib.black_scholes.implied_volatility import implied_volatility

from algo_trading_engine._internal.util.datetime_helpers import EXPIRATION_DATE_FORMAT, to_readable_str, to_readable_expiration_str

from algo_trading_engine.enums import OptionType

if TYPE_CHECKING:
    from algo_trading_engine.dto.options_dtos import OptionBarDTO, OptionContractDTO


@dataclass(frozen=True)
class Option:
    """
    Value object for an individual option contract snapshot.
    """

    ticker: str
    timestamp: datetime
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
        if isinstance(self.timestamp, str):
            object.__setattr__(self, "timestamp", datetime.fromisoformat(self.timestamp))

        if isinstance(self.option_type, str):
            object.__setattr__(self, "option_type", OptionType(self.option_type))
        elif hasattr(self.option_type, "value") and self.option_type.value in ("call", "put"):
            object.__setattr__(self, "option_type", OptionType(self.option_type.value))
        elif not isinstance(self.option_type, OptionType):
            raise ValueError(
                f"option_type must be a string or OptionType enum, got {type(self.option_type)}"
            )

        if isinstance(self.expiration, datetime):
            object.__setattr__(self, "expiration", to_readable_expiration_str(self.expiration))

        if self.mid_price is None and self.bid is not None and self.ask is not None:
            object.__setattr__(self, "mid_price", (self.bid + self.ask) / 2)

        if self.strike <= 0:
            raise ValueError("Strike price must be positive")
        if self.last_price < 0:
            raise ValueError("Last price cannot be negative")
        if self.timestamp is None:
            raise ValueError("Must associate a timestamp with an option since price data is attached.")
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
        return self.moneyness < 1.0

    @property
    def is_otm(self) -> bool:
        if self.moneyness is None:
            return False
        if self.is_call:
            return self.moneyness < 1.0
        return self.moneyness > 1.0

    @property
    def time_to_expiration(self) -> float:
        """Time to expiration in years"""
        expiration_dt = datetime.strptime(self.expiration, EXPIRATION_DATE_FORMAT)
        diff = expiration_dt - self.timestamp
        return diff.days / 365

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ticker": self.ticker,
            "timestamp": self.timestamp.isoformat(),
            "strike": self.strike,
            "expiration": self.expiration,
            "type": self.option_type.value,
            "symbol": self.symbol,
            "volume": self.volume,
            "open_interest": self.open_interest,
            "implied_volatility": self.implied_volatility,
            "delta": self.delta,
            "gamma": self.gamma,
            "theta": self.theta,
            "vega": self.vega,
            "last_price": self.last_price,
            "bid": self.bid,
            "ask": self.ask,
            "mid_price": self.mid_price,
            "moneyness": self.moneyness,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Option:
        return cls(
            ticker=data.get("ticker", ""),
            timestamp=data.get("timestamp"),
            symbol=data["symbol"],
            strike=data["strike"],
            expiration=data["expiration"],
            option_type=data["type"],
            last_price=data["last_price"],
            bid=data.get("bid"),
            ask=data.get("ask"),
            mid_price=data.get("mid_price"),
            volume=data.get("volume"),
            open_interest=data.get("open_interest"),
            delta=data.get("delta"),
            gamma=data.get("gamma"),
            theta=data.get("theta"),
            vega=data.get("vega"),
            implied_volatility=data.get("implied_volatility"),
            moneyness=data.get("moneyness"),
        )

    def __eq__(self, other) -> bool:
        if not isinstance(other, Option):
            return False
        return (
            self.ticker == other.ticker
            and self.timestamp == other.timestamp
            and self.symbol == other.symbol
            and self.strike == other.strike
            and self.expiration == other.expiration
            and self.option_type == other.option_type
        )

    def __str__(self) -> str:
        return f"{self.symbol} {self.option_type.value.upper()} {self.strike} @ {self.last_price:.2f} as of {to_readable_str(self.timestamp)}"

    def __repr__(self) -> str:
        return (
            f"Option(symbol='{self.symbol}', strike={self.strike}, "
            f"expiration={self.expiration}, "
            f"type={self.option_type.value}, price={self.last_price:.2f})"
        )

    @classmethod
    def from_contract_and_bar(cls, contract: OptionContractDTO, option_bar: OptionBarDTO) -> Option:
        """Create a complete option object from contract information and pricing data"""
        option_type = (
            OptionType.CALL if contract.contract_type.value == "call" else OptionType.PUT
        )
        return cls(
            ticker=contract.ticker,
            timestamp=option_bar.timestamp,
            symbol=contract.underlying_ticker,
            strike=float(contract.strike_price.value),
            expiration=str(contract.expiration_date),
            option_type=option_type,
            last_price=float(option_bar.close_price),
            volume=option_bar.volume,
            mid_price=float(option_bar.volume_weighted_avg_price)
            if option_bar.volume_weighted_avg_price
            else None,
        )

    def get_black_scholes_implied_vol(self, underlying_price: float, treasury_rate: float):
        """Returns the implied volatility, calculated using Black-Scholes."""
        return implied_volatility(
            price=self.last_price,
            S=underlying_price,
            K=self.strike,
            t=self.time_to_expiration,
            r=treasury_rate,
            flag="c" if self.option_type == OptionType.CALL else "p"
        )


@dataclass(frozen=True)
class OptionChain:
    """
    Immutable Data Transfer Object for an option chain matching the cache format.
    Only 'calls' and 'puts' lists are required.
    """

    calls: tuple[Option, ...] = field(default_factory=tuple)
    puts: tuple[Option, ...] = field(default_factory=tuple)
    underlying_symbol: Optional[str] = None
    expiration_date: Optional[str] = None
    current_price: Optional[float] = None
    date: Optional[str] = None
    source: Optional[str] = None

    def __post_init__(self):
        calls_list = [
            opt if isinstance(opt, Option) else Option.from_dict(opt) for opt in self.calls
        ]
        puts_list = [
            opt if isinstance(opt, Option) else Option.from_dict(opt) for opt in self.puts
        ]
        object.__setattr__(self, "calls", tuple(calls_list))
        object.__setattr__(self, "puts", tuple(puts_list))

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
            "calls": [opt.to_dict() for opt in self.calls],
            "puts": [opt.to_dict() for opt in self.puts],
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> OptionChain:
        return cls(
            calls=[Option.from_dict(opt) for opt in data.get("calls", [])],
            puts=[Option.from_dict(opt) for opt in data.get("puts", [])],
        )

    @classmethod
    def from_dict_w_options(cls, data: Dict[Option, Any]) -> OptionChain:
        return cls(
            calls=data.get("calls", []),
            puts=data.get("puts", []),
        )

    def get_option_data_for_option(self, option: Option) -> Optional[Option]:
        options_list = self.calls if option.is_call else self.puts
        for current_option in options_list:
            if (
                current_option.strike == option.strike
                and current_option.expiration == option.expiration
            ):
                return current_option
        return None

    def add_option(self, option: Option) -> OptionChain:
        if option.is_call:
            new_calls = self.calls + (option,)
            return OptionChain(
                calls=new_calls,
                puts=self.puts,
                underlying_symbol=self.underlying_symbol,
                expiration_date=self.expiration_date,
                current_price=self.current_price,
                date=self.date,
                source=self.source,
            )
        new_puts = self.puts + (option,)
        return OptionChain(
            calls=self.calls,
            puts=new_puts,
            underlying_symbol=self.underlying_symbol,
            expiration_date=self.expiration_date,
            current_price=self.current_price,
            date=self.date,
            source=self.source,
        )

    def __str__(self) -> str:
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
