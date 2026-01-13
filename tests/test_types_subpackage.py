"""
Test the vo and enums sub-package exports.

This ensures that runtime types, value objects, and enums are properly
exposed through the public API.
"""

import pytest


def test_vo_subpackage_import():
    """Test that vo sub-package can be imported."""
    from algo_trading_engine import vo
    assert vo is not None


def test_enums_subpackage_import():
    """Test that enums sub-package can be imported."""
    from algo_trading_engine import enums
    assert enums is not None


def test_strategy_type_import():
    """Test that StrategyType can be imported from enums."""
    from algo_trading_engine.enums import StrategyType
    
    # Verify enum values exist
    assert StrategyType.CALL_CREDIT_SPREAD
    assert StrategyType.PUT_CREDIT_SPREAD
    assert hasattr(StrategyType, 'LONG_CALL')


def test_position_import():
    """Test that Position can be imported from vo."""
    from algo_trading_engine.vo import Position
    from datetime import datetime
    
    # Verify Position class exists and can be instantiated
    assert Position is not None
    # Don't instantiate as it requires complex setup


def test_option_import():
    """Test that Option can be imported from vo."""
    from algo_trading_engine.vo import Option
    
    # Verify Option class exists
    assert Option is not None


def test_treasury_rates_import():
    """Test that TreasuryRates can be imported from vo."""
    from algo_trading_engine.vo import TreasuryRates
    
    # Verify TreasuryRates class exists
    assert TreasuryRates is not None


def test_vo_all_exports():
    """Test that __all__ is properly defined in vo sub-package."""
    from algo_trading_engine import vo
    
    # Verify __all__ contains expected exports
    assert hasattr(vo, '__all__')
    expected_exports = ['Position', 'Option', 'TreasuryRates', 'StrikePrice', 'ExpirationDate']
    for export in expected_exports:
        assert export in vo.__all__, f"{export} not in vo.__all__"


def test_enums_all_exports():
    """Test that __all__ is properly defined in enums sub-package."""
    from algo_trading_engine import enums
    
    # Verify __all__ contains expected exports
    assert hasattr(enums, '__all__')
    expected_exports = ['StrategyType', 'OptionType', 'MarketStateType', 'SignalType']
    for export in expected_exports:
        assert export in enums.__all__, f"{export} not in enums.__all__"


def test_vo_and_enums_usage_pattern():
    """Test the recommended usage pattern for vo and enums sub-packages."""
    # Pattern 1: Import sub-packages
    from algo_trading_engine import vo, enums
    assert hasattr(enums, 'StrategyType')
    assert hasattr(vo, 'Position')
    assert hasattr(vo, 'Option')
    assert hasattr(vo, 'TreasuryRates')
    
    # Pattern 2: Import specific types
    from algo_trading_engine.enums import StrategyType
    from algo_trading_engine.vo import Position, Option, TreasuryRates
    assert StrategyType is not None
    assert Position is not None
    assert Option is not None
    assert TreasuryRates is not None


def test_backward_compatibility_with_existing_code():
    """
    Test that existing imports still work.
    
    This ensures we haven't broken any existing functionality by
    introducing the vo and enums sub-packages.
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
