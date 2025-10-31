"""
Test regime filtering for upward_trend_reversal_strategy.

Verifies that the strategy correctly filters out:
- MOMENTUM_UPTREND (unsuitable for strategy)
- HIGH_VOLATILITY_DOWNTREND (bearish)
- CONSOLIDATION (neutral/flat)

And only trades during:
- LOW_VOLATILITY_UPTREND (bullish)
- HIGH_VOLATILITY_RALLY (bullish)
"""

import pytest
import pandas as pd
from datetime import datetime, timedelta
from unittest.mock import Mock, MagicMock

from src.strategies.upward_trend_reversal_strategy import UpwardTrendReversalStrategy
from src.common.models import MarketStateType
from src.model.market_state_classifier import MarketStateClassifier


class TestRegimeFiltering:
    """Test regime filtering logic."""
    
    def create_strategy_with_hmm(self):
        """Helper to create strategy with mock HMM model."""
        options_handler = Mock()
        strategy = UpwardTrendReversalStrategy(
            options_handler=options_handler,
            min_trend_duration=3,
            max_trend_duration=4,
            max_holding_days=2
        )
        
        # Create mock HMM model
        hmm_model = MarketStateClassifier()
        hmm_model.state_characteristics = {
            0: {'avg_return': 0.0011, 'volatility': 0.0065, 'sma_trend': 1.026, 'price_sma20': 1.013, 'volume_ratio': 1.0},  # LOW_VOL_UPTREND
            1: {'avg_return': 0.0043, 'volatility': 0.011, 'sma_trend': 1.02, 'price_sma20': 1.02, 'volume_ratio': 0.95},    # MOMENTUM_UPTREND
            2: {'avg_return': -0.0023, 'volatility': 0.015, 'sma_trend': 1.006, 'price_sma20': 0.978, 'volume_ratio': 1.14}, # HIGH_VOL_DOWNTREND
            3: {'avg_return': -0.0004, 'volatility': 0.017, 'sma_trend': 0.985, 'price_sma20': 0.999, 'volume_ratio': 1.0},  # CONSOLIDATION
            4: {'avg_return': 0.0019, 'volatility': 0.010, 'sma_trend': 1.0, 'price_sma20': 1.02, 'volume_ratio': 0.91}      # HIGH_VOL_RALLY
        }
        
        strategy.hmm_model = hmm_model
        
        return strategy, hmm_model
    
    def test_filters_momentum_uptrend(self):
        """Test that MOMENTUM_UPTREND is filtered."""
        strategy, hmm_model = self.create_strategy_with_hmm()
        
        # Create data with MOMENTUM_UPTREND state
        dates = pd.date_range('2024-01-01', periods=5)
        data = pd.DataFrame({
            'Close': [100, 101, 102, 103, 102],
            'Market_State': [1, 1, 1, 1, 1]  # State 1 = MOMENTUM_UPTREND
        }, index=dates)
        
        strategy.data = data
        
        # Verify filtering
        date = dates[2]
        assert strategy._should_filter_regime(date) == True, \
            "MOMENTUM_UPTREND should be filtered"
        
        # Verify regime type mapping
        regime_type = hmm_model.map_state_to_regime_type(1)
        assert regime_type == MarketStateType.MOMENTUM_UPTREND
    
    def test_filters_high_volatility_downtrend(self):
        """Test that HIGH_VOLATILITY_DOWNTREND is filtered."""
        strategy, hmm_model = self.create_strategy_with_hmm()
        
        # Create data with HIGH_VOLATILITY_DOWNTREND state
        dates = pd.date_range('2024-01-01', periods=5)
        data = pd.DataFrame({
            'Close': [100, 99, 98, 97, 96],
            'Market_State': [2, 2, 2, 2, 2]  # State 2 = HIGH_VOL_DOWNTREND
        }, index=dates)
        
        strategy.data = data
        
        # Verify filtering
        date = dates[2]
        assert strategy._should_filter_regime(date) == True, \
            "HIGH_VOLATILITY_DOWNTREND should be filtered"
        
        # Verify regime type mapping
        regime_type = hmm_model.map_state_to_regime_type(2)
        assert regime_type == MarketStateType.HIGH_VOLATILITY_DOWNTREND
    
    def test_filters_consolidation(self):
        """Test that CONSOLIDATION is filtered."""
        strategy, hmm_model = self.create_strategy_with_hmm()
        
        # Create data with CONSOLIDATION state
        dates = pd.date_range('2024-01-01', periods=5)
        data = pd.DataFrame({
            'Close': [100, 100.5, 99.5, 100, 100.5],
            'Market_State': [3, 3, 3, 3, 3]  # State 3 = CONSOLIDATION
        }, index=dates)
        
        strategy.data = data
        
        # Verify filtering
        date = dates[2]
        assert strategy._should_filter_regime(date) == True, \
            "CONSOLIDATION should be filtered"
        
        # Verify regime type mapping
        regime_type = hmm_model.map_state_to_regime_type(3)
        assert regime_type == MarketStateType.CONSOLIDATION
    
    def test_allows_low_volatility_uptrend(self):
        """Test that LOW_VOLATILITY_UPTREND is NOT filtered."""
        strategy, hmm_model = self.create_strategy_with_hmm()
        
        # Create data with LOW_VOLATILITY_UPTREND state
        dates = pd.date_range('2024-01-01', periods=5)
        data = pd.DataFrame({
            'Close': [100, 100.5, 101, 101.5, 102],
            'Market_State': [0, 0, 0, 0, 0]  # State 0 = LOW_VOL_UPTREND
        }, index=dates)
        
        strategy.data = data
        
        # Verify NOT filtered
        date = dates[2]
        assert strategy._should_filter_regime(date) == False, \
            "LOW_VOLATILITY_UPTREND should NOT be filtered"
        
        # Verify regime type mapping
        regime_type = hmm_model.map_state_to_regime_type(0)
        assert regime_type == MarketStateType.LOW_VOLATILITY_UPTREND
    
    def test_allows_high_volatility_rally(self):
        """Test that HIGH_VOLATILITY_RALLY is NOT filtered."""
        strategy, hmm_model = self.create_strategy_with_hmm()
        
        # Create data with HIGH_VOLATILITY_RALLY state
        dates = pd.date_range('2024-01-01', periods=5)
        data = pd.DataFrame({
            'Close': [100, 102, 101, 103, 105],
            'Market_State': [4, 4, 4, 4, 4]  # State 4 = HIGH_VOL_RALLY
        }, index=dates)
        
        strategy.data = data
        
        # Verify NOT filtered
        date = dates[2]
        assert strategy._should_filter_regime(date) == False, \
            "HIGH_VOLATILITY_RALLY should NOT be filtered"
        
        # Verify regime type mapping
        regime_type = hmm_model.map_state_to_regime_type(4)
        assert regime_type == MarketStateType.HIGH_VOLATILITY_RALLY
    
    def test_all_filtered_regimes(self):
        """Test that all filtered regimes are correctly identified."""
        strategy, hmm_model = self.create_strategy_with_hmm()
        
        filtered_regimes = [
            MarketStateType.MOMENTUM_UPTREND,
            MarketStateType.HIGH_VOLATILITY_DOWNTREND,
            MarketStateType.CONSOLIDATION
        ]
        
        tradeable_regimes = [
            MarketStateType.LOW_VOLATILITY_UPTREND,
            MarketStateType.HIGH_VOLATILITY_RALLY
        ]
        
        # Test all state IDs
        for state_id in range(5):
            regime_type = hmm_model.map_state_to_regime_type(state_id)
            
            dates = pd.date_range('2024-01-01', periods=1)
            data = pd.DataFrame({
                'Close': [100],
                'Market_State': [state_id]
            }, index=dates)
            
            strategy.data = data
            should_filter = strategy._should_filter_regime(dates[0])
            
            if regime_type in filtered_regimes:
                assert should_filter == True, \
                    f"State {state_id} ({regime_type.value}) should be filtered"
            else:
                assert should_filter == False, \
                    f"State {state_id} ({regime_type.value}) should NOT be filtered"
    
    def test_backward_compatibility_method(self):
        """Test that deprecated _is_momentum_uptrend_regime still works."""
        strategy, hmm_model = self.create_strategy_with_hmm()
        
        # Create data with MOMENTUM_UPTREND state
        dates = pd.date_range('2024-01-01', periods=5)
        data = pd.DataFrame({
            'Close': [100, 101, 102, 103, 102],
            'Market_State': [1, 1, 1, 1, 1]
        }, index=dates)
        
        strategy.data = data
        
        # Verify backward compatibility method still works
        date = dates[2]
        assert strategy._is_momentum_uptrend_regime(date) == True
    
    def test_filtering_without_hmm_model(self):
        """Test that filtering returns False when HMM model is not available."""
        options_handler = Mock()
        strategy = UpwardTrendReversalStrategy(
            options_handler=options_handler
        )
        
        dates = pd.date_range('2024-01-01', periods=5)
        data = pd.DataFrame({
            'Close': [100, 101, 102, 103, 104],
            'Market_State': [0, 1, 2, 3, 4]
        }, index=dates)
        
        strategy.data = data
        
        # Without HMM model, should not filter (returns False)
        date = dates[2]
        assert strategy._should_filter_regime(date) == False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

