"""
Test for HMM features bug.

This test reproduces the bug where Volume_Change feature is missing
when trying to apply HMM predictions to backtest data.
"""

import pytest
import pandas as pd
from datetime import datetime, timedelta
from unittest.mock import Mock

from src.common.data_retriever import DataRetriever
from src.strategies.upward_trend_reversal_strategy import UpwardTrendReversalStrategy


class TestHMMFeaturesBug:
    """Test HMM features calculation bug."""
    
    def test_calculate_features_includes_hmm_features(self):
        """Test that calculate_features_for_data includes all HMM required features."""
        # Create DataRetriever
        retriever = DataRetriever(symbol='SPY', quiet_mode=True)
        
        # Create sample data
        dates = pd.date_range('2024-01-01', periods=100)
        data = pd.DataFrame({
            'Close': [100 + i * 0.5 for i in range(100)],
            'Volume': [1000000 + i * 10000 for i in range(100)],
            'Open': [99 + i * 0.5 for i in range(100)],
            'High': [101 + i * 0.5 for i in range(100)],
            'Low': [98 + i * 0.5 for i in range(100)]
        }, index=dates)
        
        # Calculate features
        retriever.calculate_features_for_data(data)
        
        # Verify required HMM features are present (from MarketStateClassifier)
        required_hmm_features = ['Returns', 'Volatility', 'Price_to_SMA20', 'SMA20_to_SMA50', 'Volume_Ratio']
        
        for feature in required_hmm_features:
            assert feature in data.columns, f"Missing required HMM feature: {feature}"
    
    def test_hmm_training_with_features(self):
        """Test that HMM training works when all features are present."""
        from src.strategies.hmm_strategy import HMMStrategy
        
        # Create mock retriever
        mock_retriever = Mock()
        
        # Create sample historical data WITH all required features
        dates = pd.date_range('2022-01-01', periods=500)
        historical_data = pd.DataFrame({
            'Close': [100 + i * 0.5 for i in range(500)],
            'Returns': [0.005] * 500,
            'Volatility': [0.01] * 500,
            'Price_to_SMA20': [1.0] * 500,
            'SMA20_to_SMA50': [1.0] * 500,
            'Volume_Ratio': [1.0] * 500
        }, index=dates)
        
        # Mock fetch_data_for_period to return our sample data
        mock_retriever.fetch_data_for_period.return_value = historical_data
        mock_retriever.calculate_features_for_data = Mock()
        
        # Create strategy with HMM training enabled
        mock_handler = Mock()
        strategy = UpwardTrendReversalStrategy(
            options_handler=mock_handler,
            data_retriever=mock_retriever,
            train_hmm=True,
            hmm_training_years=2
        )
        
        # Create backtest data WITH all required features
        backtest_dates = pd.date_range('2024-01-01', periods=100)
        backtest_data = pd.DataFrame({
            'Close': [150 + i * 0.5 for i in range(100)],
            'Returns': [0.005] * 100,
            'Volatility': [0.01] * 100,
            'Price_to_SMA20': [1.0] * 100,
            'SMA20_to_SMA50': [1.0] * 100,
            'Volume_Ratio': [1.0] * 100,
            'Market_State': [0] * 100  # Will be overwritten by HMM
        }, index=backtest_dates)
        
        # This should not raise an error
        try:
            strategy.set_data(backtest_data, None, None)
            # If we get here, HMM was applied successfully
            assert 'Market_State' in strategy.data.columns
        except ValueError as e:
            if 'Missing required features' in str(e):
                pytest.fail(f"HMM training failed due to missing features: {e}")
            raise
    
    def test_hmm_training_fails_without_features(self):
        """Test that HMM training raises clear error when features are missing."""
        from src.strategies.hmm_strategy import HMMStrategy
        
        # Create mock retriever
        mock_retriever = Mock()
        
        # Create sample historical data WITH all features
        dates = pd.date_range('2022-01-01', periods=500)
        historical_data = pd.DataFrame({
            'Close': [100 + i * 0.5 for i in range(500)],
            'Returns': [0.005] * 500,
            'Volatility': [0.01] * 500,
            'Price_to_SMA20': [1.0] * 500,
            'SMA20_to_SMA50': [1.0] * 500,
            'Volume_Ratio': [1.0] * 500
        }, index=dates)
        
        mock_retriever.fetch_data_for_period.return_value = historical_data
        mock_retriever.calculate_features_for_data = Mock()
        
        # Create strategy
        mock_handler = Mock()
        strategy = UpwardTrendReversalStrategy(
            options_handler=mock_handler,
            data_retriever=mock_retriever,
            train_hmm=True,
            hmm_training_years=2
        )
        
        # Create backtest data WITHOUT required features
        backtest_dates = pd.date_range('2024-01-01', periods=100)
        backtest_data = pd.DataFrame({
            'Close': [150 + i * 0.5 for i in range(100)],
            'Returns': [0.005] * 100,
            'Volatility': [0.01] * 100
            # Missing Price_to_SMA20, SMA20_to_SMA50, Volume_Ratio!
        }, index=backtest_dates)
        
        # This should raise a clear error
        with pytest.raises(ValueError, match="Cannot apply HMM: Missing required features"):
            strategy.set_data(backtest_data, None, None)
    
    def test_backtest_data_has_all_hmm_features(self):
        """Test that backtest data preparation includes all HMM features."""
        retriever = DataRetriever(symbol='SPY', quiet_mode=True)
        
        # Create minimal backtest data
        dates = pd.date_range('2024-01-01', periods=100)
        data = pd.DataFrame({
            'Close': [100 + i * 0.5 for i in range(100)],
            'Volume': [1000000 + i * 10000 for i in range(100)],
            'Open': [99 + i * 0.5 for i in range(100)],
            'High': [101 + i * 0.5 for i in range(100)],
            'Low': [98 + i * 0.5 for i in range(100)]
        }, index=dates)
        
        # Calculate features (simulating main.py workflow)
        retriever.calculate_features_for_data(data)
        
        # Verify all HMM features exist (from MarketStateClassifier)
        hmm_features = ['Returns', 'Volatility', 'Price_to_SMA20', 'SMA20_to_SMA50', 'Volume_Ratio']
        
        for feature in hmm_features:
            assert feature in data.columns, f"{feature} feature missing (BUG!)"
            # Verify they have valid values (not all NaN)
            assert not data[feature].isna().all(), f"{feature} values are all NaN"

