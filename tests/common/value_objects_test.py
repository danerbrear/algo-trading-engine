"""
Unit tests for Value Objects defined in src/common/models.py
"""
import pytest
from decimal import Decimal
from algo_trading_engine.vo import (
    MarketState, TradingSignal, PriceRange, Volatility
)
from algo_trading_engine.enums import MarketStateType, SignalType


class TestMarketState:
    """Test cases for MarketState Value Object"""
    
    def test_valid_creation(self):
        """Test valid MarketState creation"""
        state = MarketState(
            state_type=MarketStateType.LOW_VOLATILITY_UPTREND,
            volatility=Decimal('0.15'),
            average_return=Decimal('0.02'),
            confidence=Decimal('0.8')
        )
        assert state.state_type == MarketStateType.LOW_VOLATILITY_UPTREND
        assert state.volatility == Decimal('0.15')
        assert state.confidence == Decimal('0.8')
    
    def test_invalid_confidence_raises_error(self):
        """Test that invalid confidence values raise ValueError"""
        with pytest.raises(ValueError, match="Confidence must be between 0 and 1"):
            MarketState(
                state_type=MarketStateType.LOW_VOLATILITY_UPTREND,
                volatility=Decimal('0.15'),
                average_return=Decimal('0.02'),
                confidence=Decimal('1.5')  # Invalid
            )
    
    def test_negative_volatility_raises_error(self):
        """Test that negative volatility raises ValueError"""
        with pytest.raises(ValueError, match="Volatility cannot be negative"):
            MarketState(
                state_type=MarketStateType.LOW_VOLATILITY_UPTREND,
                volatility=Decimal('-0.1'),  # Invalid
                average_return=Decimal('0.02'),
                confidence=Decimal('0.8')
            )
    
    def test_is_bullish(self):
        """Test bullish state detection"""
        bullish_state = MarketState(
            state_type=MarketStateType.MOMENTUM_UPTREND,
            volatility=Decimal('0.15'),
            average_return=Decimal('0.02'),
            confidence=Decimal('0.8')
        )
        assert bullish_state.is_bullish() is True
        assert bullish_state.is_bearish() is False
    
    def test_is_bearish(self):
        """Test bearish state detection"""
        bearish_state = MarketState(
            state_type=MarketStateType.HIGH_VOLATILITY_DOWNTREND,
            volatility=Decimal('0.25'),
            average_return=Decimal('-0.02'),
            confidence=Decimal('0.7')
        )
        assert bearish_state.is_bearish() is True
        assert bearish_state.is_bullish() is False
    
    def test_is_consolidation(self):
        """Test consolidation state detection"""
        consolidation_state = MarketState(
            state_type=MarketStateType.CONSOLIDATION,
            volatility=Decimal('0.10'),
            average_return=Decimal('0.001'),
            confidence=Decimal('0.6')
        )
        assert consolidation_state.is_consolidation() is True
        assert consolidation_state.is_bullish() is False
        assert consolidation_state.is_bearish() is False


class TestTradingSignal:
    """Test cases for TradingSignal Value Object"""
    
    def test_valid_creation(self):
        """Test valid TradingSignal creation"""
        signal = TradingSignal(
            signal_type=SignalType.CALL_CREDIT_SPREAD,
            confidence=Decimal('0.75'),
            ticker='SPY',
            expiration_date='2024-01-19',
            strike_prices=(Decimal('450'), Decimal('455'))
        )
        assert signal.signal_type == SignalType.CALL_CREDIT_SPREAD
        assert signal.confidence == Decimal('0.75')
        assert signal.ticker == 'SPY'
    
    def test_invalid_confidence_raises_error(self):
        """Test that invalid confidence values raise ValueError"""
        with pytest.raises(ValueError, match="Confidence must be between 0 and 1"):
            TradingSignal(
                signal_type=SignalType.CALL_CREDIT_SPREAD,
                confidence=Decimal('1.2'),  # Invalid
                ticker='SPY'
            )
    
    def test_missing_expiration_for_option_strategy_raises_error(self):
        """Test that option strategies require expiration date"""
        with pytest.raises(ValueError, match="Expiration date required for option strategies"):
            TradingSignal(
                signal_type=SignalType.CALL_CREDIT_SPREAD,
                confidence=Decimal('0.75'),
                ticker='SPY'
                # Missing expiration_date
            )
    
    def test_hold_signal_does_not_require_expiration(self):
        """Test that HOLD signals don't require expiration date"""
        signal = TradingSignal(
            signal_type=SignalType.HOLD,
            confidence=Decimal('0.5'),
            ticker='SPY'
            # No expiration_date required
        )
        assert signal.signal_type == SignalType.HOLD
    
    def test_is_option_strategy(self):
        """Test option strategy detection"""
        option_signal = TradingSignal(
            signal_type=SignalType.CALL_CREDIT_SPREAD,
            confidence=Decimal('0.75'),
            ticker='SPY',
            expiration_date='2024-01-19'
        )
        assert option_signal.is_option_strategy() is True
        
        stock_signal = TradingSignal(
            signal_type=SignalType.HOLD,
            confidence=Decimal('0.5'),
            ticker='SPY'
        )
        assert stock_signal.is_option_strategy() is False
    
    def test_is_credit_spread(self):
        """Test credit spread strategy detection"""
        credit_spread_signal = TradingSignal(
            signal_type=SignalType.PUT_CREDIT_SPREAD,
            confidence=Decimal('0.75'),
            ticker='SPY',
            expiration_date='2024-01-19'
        )
        assert credit_spread_signal.is_credit_spread() is True
        
        # LSTM only outputs HOLD and credit spreads; HOLD is not a credit spread
        hold_signal = TradingSignal(
            signal_type=SignalType.HOLD,
            confidence=Decimal('0.5'),
            ticker='SPY'
        )
        assert hold_signal.is_credit_spread() is False


class TestPriceRange:
    """Test cases for PriceRange Value Object"""
    
    def test_valid_creation(self):
        """Test valid PriceRange creation"""
        price_range = PriceRange(
            low=Decimal('100.00'),
            high=Decimal('110.00')
        )
        assert price_range.low == Decimal('100.00')
        assert price_range.high == Decimal('110.00')
    
    def test_low_greater_than_high_raises_error(self):
        """Test that low > high raises ValueError"""
        with pytest.raises(ValueError, match="Low price cannot be greater than high price"):
            PriceRange(
                low=Decimal('110.00'),
                high=Decimal('100.00')  # Invalid
            )
    
    def test_negative_price_raises_error(self):
        """Test that negative prices raise ValueError"""
        with pytest.raises(ValueError, match="Price cannot be negative"):
            PriceRange(
                low=Decimal('-10.00'),  # Invalid
                high=Decimal('100.00')
            )
    
    def test_contains(self):
        """Test price containment check"""
        price_range = PriceRange(
            low=Decimal('100.00'),
            high=Decimal('110.00')
        )
        assert price_range.contains(Decimal('105.00')) is True
        assert price_range.contains(Decimal('100.00')) is True  # Edge case
        assert price_range.contains(Decimal('110.00')) is True  # Edge case
        assert price_range.contains(Decimal('95.00')) is False
        assert price_range.contains(Decimal('115.00')) is False
    
    def test_spread(self):
        """Test spread calculation"""
        price_range = PriceRange(
            low=Decimal('100.00'),
            high=Decimal('110.00')
        )
        assert price_range.spread() == Decimal('10.00')
    
    def test_midpoint(self):
        """Test midpoint calculation"""
        price_range = PriceRange(
            low=Decimal('100.00'),
            high=Decimal('110.00')
        )
        assert price_range.midpoint() == Decimal('105.00')


class TestVolatility:
    """Test cases for Volatility Value Object"""
    
    def test_valid_creation(self):
        """Test valid Volatility creation"""
        volatility = Volatility(
            value=Decimal('0.20'),
            period=30
        )
        assert volatility.value == Decimal('0.20')
        assert volatility.period == 30
    
    def test_negative_volatility_raises_error(self):
        """Test that negative volatility raises ValueError"""
        with pytest.raises(ValueError, match="Volatility cannot be negative"):
            Volatility(
                value=Decimal('-0.1'),  # Invalid
                period=30
            )
    
    def test_zero_period_raises_error(self):
        """Test that zero period raises ValueError"""
        with pytest.raises(ValueError, match="Period must be positive"):
            Volatility(
                value=Decimal('0.20'),
                period=0  # Invalid
            )
    
    def test_is_high(self):
        """Test high volatility detection"""
        high_vol = Volatility(value=Decimal('0.35'), period=30)
        assert high_vol.is_high() is True
        assert high_vol.is_low() is False
        assert high_vol.is_moderate() is False
    
    def test_is_low(self):
        """Test low volatility detection"""
        low_vol = Volatility(value=Decimal('0.05'), period=30)
        assert low_vol.is_low() is True
        assert low_vol.is_high() is False
        assert low_vol.is_moderate() is False
    
    def test_is_moderate(self):
        """Test moderate volatility detection"""
        moderate_vol = Volatility(value=Decimal('0.15'), period=30)
        assert moderate_vol.is_moderate() is True
        assert moderate_vol.is_high() is False
        assert moderate_vol.is_low() is False


class TestValueObjectImmutability:
    """Test that all Value Objects are immutable"""
    
    def test_market_state_immutability(self):
        """Test that MarketState is immutable"""
        state = MarketState(
            state_type=MarketStateType.LOW_VOLATILITY_UPTREND,
            volatility=Decimal('0.15'),
            average_return=Decimal('0.02'),
            confidence=Decimal('0.8')
        )
        with pytest.raises(AttributeError):
            state.confidence = Decimal('0.9')
    
    def test_trading_signal_immutability(self):
        """Test that TradingSignal is immutable"""
        signal = TradingSignal(
            signal_type=SignalType.HOLD,
            confidence=Decimal('0.5'),
            ticker='SPY'
        )
        with pytest.raises(AttributeError):
            signal.ticker = 'QQQ'
    
    def test_price_range_immutability(self):
        """Test that PriceRange is immutable"""
        price_range = PriceRange(
            low=Decimal('100.00'),
            high=Decimal('110.00')
        )
        with pytest.raises(AttributeError):
            price_range.low = Decimal('95.00')
    
    def test_volatility_immutability(self):
        """Test that Volatility is immutable"""
        volatility = Volatility(
            value=Decimal('0.20'),
            period=30
        )
        with pytest.raises(AttributeError):
            volatility.value = Decimal('0.25')
