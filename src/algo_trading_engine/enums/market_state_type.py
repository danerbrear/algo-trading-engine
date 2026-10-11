from enum import Enum


class MarketStateType(Enum):
    """Enum for market state types identified by HMM."""

    LOW_VOLATILITY_UPTREND = "low_volatility_uptrend"
    MOMENTUM_UPTREND = "momentum_uptrend"
    CONSOLIDATION = "consolidation"
    HIGH_VOLATILITY_DOWNTREND = "high_volatility_downtrend"
    HIGH_VOLATILITY_RALLY = "high_volatility_rally"
