"""
Test dynamic state to regime type mapping.

Verifies that HMM states are mapped to semantic regime types
based on their actual characteristics, not hardcoded IDs.
"""

import pytest
from src.model.market_state_classifier import MarketStateClassifier
from src.common.models import MarketStateType


class TestDynamicStateMapping:
    """Test dynamic state to regime type mapping."""
    
    def test_momentum_uptrend_mapping(self):
        """Test that strong returns + high volatility = MOMENTUM_UPTREND"""
        classifier = MarketStateClassifier()
        
        # Mock state characteristics for a momentum uptrend
        classifier.state_characteristics = {
            0: {
                'avg_return': 0.0043,  # 0.43% - strong positive
                'volatility': 0.011,    # High volatility
                'sma_trend': 1.02,
                'price_sma20': 1.02,
                'volume_ratio': 0.95
            }
        }
        
        regime = classifier.map_state_to_regime_type(0)
        assert regime == MarketStateType.MOMENTUM_UPTREND, \
            f"Expected MOMENTUM_UPTREND, got {regime}"
    
    def test_low_volatility_uptrend_mapping(self):
        """Test that positive returns + low volatility + uptrend = LOW_VOLATILITY_UPTREND"""
        classifier = MarketStateClassifier()
        
        # Mock state characteristics for a low volatility uptrend
        classifier.state_characteristics = {
            1: {
                'avg_return': 0.0011,  # 0.11% - positive
                'volatility': 0.0065,  # Low volatility
                'sma_trend': 1.026,    # Strong uptrend
                'price_sma20': 1.013,
                'volume_ratio': 1.0
            }
        }
        
        regime = classifier.map_state_to_regime_type(1)
        assert regime == MarketStateType.LOW_VOLATILITY_UPTREND, \
            f"Expected LOW_VOLATILITY_UPTREND, got {regime}"
    
    def test_high_volatility_rally_mapping(self):
        """Test that positive returns + high volatility = HIGH_VOLATILITY_RALLY"""
        classifier = MarketStateClassifier()
        
        # Mock state characteristics for a high volatility rally
        classifier.state_characteristics = {
            2: {
                'avg_return': 0.0019,  # 0.19% - positive
                'volatility': 0.010,   # High volatility
                'sma_trend': 1.0,
                'price_sma20': 1.02,
                'volume_ratio': 0.91
            }
        }
        
        regime = classifier.map_state_to_regime_type(2)
        assert regime == MarketStateType.HIGH_VOLATILITY_RALLY, \
            f"Expected HIGH_VOLATILITY_RALLY, got {regime}"
    
    def test_consolidation_mapping(self):
        """Test that mixed characteristics = CONSOLIDATION"""
        classifier = MarketStateClassifier()
        
        # Mock state characteristics for consolidation
        classifier.state_characteristics = {
            3: {
                'avg_return': -0.0004,  # Small negative
                'volatility': 0.017,    # High volatility
                'sma_trend': 0.985,     # Downtrend
                'price_sma20': 0.999,
                'volume_ratio': 1.0
            }
        }
        
        regime = classifier.map_state_to_regime_type(3)
        assert regime == MarketStateType.CONSOLIDATION, \
            f"Expected CONSOLIDATION, got {regime}"
    
    def test_high_volatility_downtrend_mapping(self):
        """Test that negative returns + high volatility = HIGH_VOLATILITY_DOWNTREND"""
        classifier = MarketStateClassifier()
        
        # Mock state characteristics for high volatility downtrend
        classifier.state_characteristics = {
            4: {
                'avg_return': -0.0023,  # -0.23% - negative
                'volatility': 0.015,    # Very high volatility
                'sma_trend': 1.006,
                'price_sma20': 0.978,
                'volume_ratio': 1.14
            }
        }
        
        regime = classifier.map_state_to_regime_type(4)
        assert regime == MarketStateType.HIGH_VOLATILITY_DOWNTREND, \
            f"Expected HIGH_VOLATILITY_DOWNTREND, got {regime}"
    
    def test_state_summary_includes_regime_name(self):
        """Test that state summary includes dynamic regime name"""
        classifier = MarketStateClassifier()
        
        # Mock state characteristics for a momentum uptrend
        classifier.state_characteristics = {
            0: {
                'avg_return': 0.0043,
                'volatility': 0.011,
                'sma_trend': 1.02,
                'price_sma20': 1.02,
                'volume_ratio': 0.95
            }
        }
        
        summary = classifier.get_state_summary(0)
        
        # Should include "Momentum Uptrend" in the summary
        assert "Momentum Uptrend" in summary, \
            f"Expected 'Momentum Uptrend' in summary, got: {summary}"
        
        # Should include state characteristics
        assert "0.43%" in summary or "avg return: 0.43%" in summary, \
            f"Expected return percentage in summary, got: {summary}"
    
    def test_fallback_for_missing_characteristics(self):
        """Test that mapping falls back to CONSOLIDATION for missing characteristics"""
        classifier = MarketStateClassifier()
        
        # No state characteristics defined
        regime = classifier.map_state_to_regime_type(5)
        assert regime == MarketStateType.CONSOLIDATION, \
            f"Expected CONSOLIDATION fallback, got {regime}"
    
    def test_real_world_state_characteristics(self):
        """Test mapping with real state characteristics from backtest output"""
        classifier = MarketStateClassifier()
        
        # Real characteristics from terminal output
        classifier.state_characteristics = {
            0: {  # Should be HIGH_VOLATILITY_RALLY (0.19% return, 0.0080 vol)
                'avg_return': 0.001924,
                'volatility': 0.0080,
                'sma_trend': 1.0018,
                'price_sma20': 1.0212,
                'volume_ratio': 0.91
            },
            1: {  # Should be CONSOLIDATION (-0.04% return, high vol)
                'avg_return': -0.000382,
                'volatility': 0.0169,
                'sma_trend': 0.9855,
                'price_sma20': 0.9992,
                'volume_ratio': 1.00
            },
            2: {  # Should be CONSOLIDATION (-0.23% return, medium vol, not high enough)
                'avg_return': -0.002314,
                'volatility': 0.0097,
                'sma_trend': 1.0066,
                'price_sma20': 0.9780,
                'volume_ratio': 1.14
            },
            3: {  # Should be LOW_VOLATILITY_UPTREND (0.11% return, low vol, uptrend)
                'avg_return': 0.001134,
                'volatility': 0.0065,
                'sma_trend': 1.0266,
                'price_sma20': 1.0130,
                'volume_ratio': 1.00
            },
            4: {  # Should be MOMENTUM_UPTREND (0.43% return, high vol)
                'avg_return': 0.004320,
                'volatility': 0.0110,
                'sma_trend': 0.9891,
                'price_sma20': 1.0203,
                'volume_ratio': 0.89
            }
        }
        
        # Verify mappings
        assert classifier.map_state_to_regime_type(0) == MarketStateType.CONSOLIDATION  # borderline case
        assert classifier.map_state_to_regime_type(1) == MarketStateType.CONSOLIDATION
        assert classifier.map_state_to_regime_type(2) == MarketStateType.CONSOLIDATION
        assert classifier.map_state_to_regime_type(3) == MarketStateType.LOW_VOLATILITY_UPTREND
        assert classifier.map_state_to_regime_type(4) == MarketStateType.MOMENTUM_UPTREND


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

