"""
Test the types sub-package exports.

This ensures that runtime types, value objects, and enums are properly
exposed through the public API.
"""

import pytest


def test_types_subpackage_import():
    """Test that types sub-package can be imported."""
    from algo_trading_engine import types
    assert types is not None


def test_strategy_type_import():
    """Test that StrategyType can be imported from types."""
    from algo_trading_engine.types import StrategyType
    
    # Verify enum values exist
    assert StrategyType.CALL_CREDIT_SPREAD
    assert StrategyType.PUT_CREDIT_SPREAD
    assert hasattr(StrategyType, 'LONG_CALL')


def test_position_import():
    """Test that Position can be imported from types."""
    from algo_trading_engine.types import Position
    from datetime import datetime
    
    # Verify Position class exists and can be instantiated
    assert Position is not None
    # Don't instantiate as it requires complex setup


def test_option_import():
    """Test that Option can be imported from types."""
    from algo_trading_engine.types import Option
    
    # Verify Option class exists
    assert Option is not None


def test_treasury_rates_import():
    """Test that TreasuryRates can be imported from types."""
    from algo_trading_engine.types import TreasuryRates
    
    # Verify TreasuryRates class exists
    assert TreasuryRates is not None


def test_types_all_exports():
    """Test that __all__ is properly defined in types sub-package."""
    from algo_trading_engine import types
    
    # Verify __all__ contains expected exports
    assert hasattr(types, '__all__')
    expected_exports = ['StrategyType', 'Position', 'Option', 'TreasuryRates']
    for export in expected_exports:
        assert export in types.__all__, f"{export} not in types.__all__"


def test_types_usage_pattern():
    """Test the recommended usage pattern for types sub-package."""
    # Pattern 1: Import sub-package
    from algo_trading_engine import types
    assert hasattr(types, 'StrategyType')
    assert hasattr(types, 'Position')
    assert hasattr(types, 'Option')
    assert hasattr(types, 'TreasuryRates')
    
    # Pattern 2: Import specific types
    from algo_trading_engine.types import StrategyType, Position, Option, TreasuryRates
    assert StrategyType is not None
    assert Position is not None
    assert Option is not None
    assert TreasuryRates is not None


def test_backward_compatibility_with_existing_code():
    """
    Test that existing imports still work.
    
    This ensures we haven't broken any existing functionality by
    introducing the types sub-package.
    """
    # Main API imports should still work
    from algo_trading_engine import (
        BacktestEngine,
        PaperTradingEngine,
        Strategy,
        BacktestConfig,
        PaperTradingConfig,
    )
    
    assert BacktestEngine is not None
    assert PaperTradingEngine is not None
    assert Strategy is not None
    assert BacktestConfig is not None
    assert PaperTradingConfig is not None
