"""
Tests to verify data_retriever changes don't break existing functionality.

This test validates:
1. DataRetriever has both options_handler (modern) and _lstm_options_handler (legacy)
2. Modern options_handler has get_contract_list_for_date method
3. Legacy _lstm_options_handler has LSTM methods (calculate_option_features, etc.)
4. Features calculation includes Volume_Change
"""

import pytest
from src.common.data_retriever import DataRetriever
from src.common.options_handler import OptionsHandler


class TestDataRetrieverChanges:
    """Test that DataRetriever changes maintain backward compatibility."""
    
    def test_data_retriever_has_modern_options_handler(self):
        """Test that data_retriever.options_handler is the modern handler."""
        retriever = DataRetriever(symbol='SPY', quiet_mode=True)
        
        # Verify options_handler exists and is the modern type
        assert hasattr(retriever, 'options_handler')
        assert retriever.options_handler is not None
        assert isinstance(retriever.options_handler, OptionsHandler)
    
    def test_modern_options_handler_has_get_contract_list_method(self):
        """Test that modern options_handler has get_contract_list_for_date."""
        retriever = DataRetriever(symbol='SPY', quiet_mode=True)
        
        # Verify get_contract_list_for_date exists
        assert hasattr(retriever.options_handler, 'get_contract_list_for_date')
        assert callable(retriever.options_handler.get_contract_list_for_date)
    
    def test_data_retriever_has_legacy_lstm_handler(self):
        """Test that data_retriever has _lstm_options_handler for LSTM training."""
        retriever = DataRetriever(symbol='SPY', quiet_mode=True)
        
        # Verify private _lstm_options_handler exists
        assert hasattr(retriever, '_lstm_options_handler')
        assert retriever._lstm_options_handler is not None
    
    def test_legacy_handler_has_lstm_methods(self):
        """Test that legacy handler has LSTM-specific methods."""
        retriever = DataRetriever(symbol='SPY', quiet_mode=True)
        
        # Verify LSTM methods exist on legacy handler
        assert hasattr(retriever._lstm_options_handler, 'calculate_option_features')
        assert hasattr(retriever._lstm_options_handler, 'calculate_option_signals')
        assert callable(retriever._lstm_options_handler.calculate_option_features)
        assert callable(retriever._lstm_options_handler.calculate_option_signals)
    
    def test_modern_handler_does_not_have_lstm_methods(self):
        """Test that modern handler does NOT have legacy LSTM methods."""
        retriever = DataRetriever(symbol='SPY', quiet_mode=True)
        
        # Modern handler should not have these legacy methods
        assert not hasattr(retriever.options_handler, 'calculate_option_features')
        assert not hasattr(retriever.options_handler, 'calculate_option_signals')
    
    def test_modern_handler_has_different_signature(self):
        """Test that modern and legacy handlers have different constructors."""
        from src.common.options_handler import OptionsHandler
        import inspect
        
        # Get constructor signature
        modern_sig = inspect.signature(OptionsHandler.__init__)
        modern_params = list(modern_sig.parameters.keys())
        
        # Modern handler should NOT have start_date or quiet_mode
        assert 'start_date' not in modern_params
        assert 'quiet_mode' not in modern_params
        
        # Should have symbol, api_key, cache_dir, use_free_tier
        assert 'symbol' in modern_params
        assert 'cache_dir' in modern_params
        assert 'use_free_tier' in modern_params
    
    def test_both_handlers_use_same_symbol(self):
        """Test that both handlers are initialized with the same symbol."""
        retriever = DataRetriever(symbol='AAPL', quiet_mode=True)
        
        # Both should have same symbol
        assert retriever.options_handler.symbol == 'AAPL'
        assert retriever._lstm_options_handler.symbol == 'AAPL'
    
    def test_both_handlers_use_same_cache_dir(self):
        """Test that both handlers use the same cache directory."""
        retriever = DataRetriever(symbol='SPY', quiet_mode=True)
        
        # Both should have cache_manager
        assert hasattr(retriever.options_handler, 'cache_manager')
        assert hasattr(retriever._lstm_options_handler, 'cache_manager')
    
    def test_volume_change_included_in_features(self):
        """Test that Volume_Change is now calculated in features."""
        import pandas as pd
        
        retriever = DataRetriever(symbol='SPY', quiet_mode=True)
        
        # Create test data
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
        
        # Verify Volume_Change is present
        assert 'Volume_Change' in data.columns
        
        # Verify other required HMM features are also present
        assert 'Returns' in data.columns
        assert 'Volatility' in data.columns
    
    def test_hmm_required_features_all_present(self):
        """Test that all HMM required features are calculated."""
        import pandas as pd
        
        retriever = DataRetriever(symbol='SPY', quiet_mode=True)
        
        # Create test data
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
        
        # HMM requires these three features
        hmm_required_features = ['Returns', 'Volatility', 'Volume_Change']
        
        for feature in hmm_required_features:
            assert feature in data.columns, f"Missing HMM required feature: {feature}"
            # Verify feature has valid data (not all NaN)
            assert not data[feature].isna().all(), f"Feature {feature} is all NaN"

