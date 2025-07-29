"""
Tests for volume validation functionality.
"""

import pytest
from src.backtest.volume_validator import VolumeValidator
from src.backtest.config import VolumeConfig, VolumeStats
from src.common.models import Option, OptionType


class TestVolumeValidator:
    """Test cases for VolumeValidator static methods."""
    
    def test_validate_option_volume_with_sufficient_volume(self):
        """Test volume validation with sufficient volume."""
        option = Option(
            ticker="SPY",
            symbol="SPY230616C00400000",
            strike=400.0,
            expiration="2023-06-16",
            option_type=OptionType.CALL,
            last_price=5.50,
            volume=15
        )
        
        assert VolumeValidator.validate_option_volume(option, min_volume=10) == True
    
    def test_validate_option_volume_with_insufficient_volume(self):
        """Test volume validation with insufficient volume."""
        option = Option(
            ticker="SPY",
            symbol="SPY230616C00400000",
            strike=400.0,
            expiration="2023-06-16",
            option_type=OptionType.CALL,
            last_price=5.50,
            volume=5
        )
        
        assert VolumeValidator.validate_option_volume(option, min_volume=10) == False
    
    def test_validate_option_volume_with_no_volume_data(self):
        """Test volume validation with no volume data."""
        option = Option(
            ticker="SPY",
            symbol="SPY230616C00400000",
            strike=400.0,
            expiration="2023-06-16",
            option_type=OptionType.CALL,
            last_price=5.50,
            volume=None
        )
        
        assert VolumeValidator.validate_option_volume(option, min_volume=10) == False
    
    def test_validate_spread_volume_with_valid_spread(self):
        """Test spread volume validation with valid options."""
        option1 = Option(
            ticker="SPY",
            symbol="SPY230616C00400000",
            strike=400.0,
            expiration="2023-06-16",
            option_type=OptionType.CALL,
            last_price=5.50,
            volume=15
        )
        option2 = Option(
            ticker="SPY",
            symbol="SPY230616C00405000",
            strike=405.0,
            expiration="2023-06-16",
            option_type=OptionType.CALL,
            last_price=3.20,
            volume=12
        )
        
        spread_options = [option1, option2]
        assert VolumeValidator.validate_spread_volume(spread_options, min_volume=10) == True
    
    def test_validate_spread_volume_with_invalid_spread(self):
        """Test spread volume validation with invalid options."""
        option1 = Option(
            ticker="SPY",
            symbol="SPY230616C00400000",
            strike=400.0,
            expiration="2023-06-16",
            option_type=OptionType.CALL,
            last_price=5.50,
            volume=15
        )
        option2 = Option(
            ticker="SPY",
            symbol="SPY230616C00405000",
            strike=405.0,
            expiration="2023-06-16",
            option_type=OptionType.CALL,
            last_price=3.20,
            volume=5  # Insufficient volume
        )
        
        spread_options = [option1, option2]
        assert VolumeValidator.validate_spread_volume(spread_options, min_volume=10) == False
    
    def test_validate_spread_volume_with_empty_spread(self):
        """Test spread volume validation with empty spread."""
        spread_options = []
        assert VolumeValidator.validate_spread_volume(spread_options, min_volume=10) == False
    
    def test_get_volume_status(self):
        """Test getting volume status for an option."""
        option = Option(
            ticker="SPY",
            symbol="SPY230616C00400000",
            strike=400.0,
            expiration="2023-06-16",
            option_type=OptionType.CALL,
            last_price=5.50,
            volume=15
        )
        
        status = VolumeValidator.get_volume_status(option)
        
        assert status['symbol'] == "SPY230616C00400000"
        assert status['volume'] == 15
        assert status['has_volume_data'] == True
        assert status['is_liquid'] == True
        assert status['strike'] == 400.0
        assert status['expiration'] == "2023-06-16"
        assert status['option_type'] == "call"
    
    def test_get_spread_volume_status(self):
        """Test getting volume status for a spread."""
        option1 = Option(
            ticker="SPY",
            symbol="SPY230616C00400000",
            strike=400.0,
            expiration="2023-06-16",
            option_type=OptionType.CALL,
            last_price=5.50,
            volume=15
        )
        option2 = Option(
            ticker="SPY",
            symbol="SPY230616C00405000",
            strike=405.0,
            expiration="2023-06-16",
            option_type=OptionType.CALL,
            last_price=3.20,
            volume=5
        )
        
        spread_options = [option1, option2]
        status = VolumeValidator.get_spread_volume_status(spread_options)
        
        assert status['is_valid'] == False
        assert status['total_options'] == 2
        assert status['valid_options'] == 1
        assert len(status['options_status']) == 2


class TestVolumeConfig:
    """Test cases for VolumeConfig DTO."""
    
    def test_valid_config(self):
        """Test creating a valid volume config."""
        config = VolumeConfig(min_volume=15, enable_volume_validation=True)
        assert config.min_volume == 15
        assert config.enable_volume_validation == True
    
    def test_default_config(self):
        """Test creating config with default values."""
        config = VolumeConfig()
        assert config.min_volume == 10
        assert config.enable_volume_validation == True
    
    def test_invalid_negative_volume(self):
        """Test that negative volume raises ValueError."""
        with pytest.raises(ValueError, match="Minimum volume cannot be negative"):
            VolumeConfig(min_volume=-5)
    
    def test_invalid_zero_volume(self):
        """Test that zero volume raises ValueError."""
        with pytest.raises(ValueError, match="Minimum volume must be greater than 0"):
            VolumeConfig(min_volume=0)


class TestVolumeStats:
    """Test cases for VolumeStats DTO."""
    
    def test_default_stats(self):
        """Test creating default volume stats."""
        stats = VolumeStats()
        assert stats.positions_rejected_volume == 0
        assert stats.options_checked == 0
        assert stats.api_fetch_failures == 0
        assert stats.api_errors == 0
        assert stats.cache_updates == 0
    
    def test_increment_rejected_positions(self):
        """Test incrementing rejected positions count."""
        stats = VolumeStats()
        new_stats = stats.increment_rejected_positions()
        
        assert new_stats.positions_rejected_volume == 1
        assert new_stats.options_checked == 0
        assert new_stats.api_fetch_failures == 0
        assert new_stats.api_errors == 0
        assert new_stats.cache_updates == 0
    
    def test_increment_options_checked(self):
        """Test incrementing options checked count."""
        stats = VolumeStats()
        new_stats = stats.increment_options_checked()
        
        assert new_stats.positions_rejected_volume == 0
        assert new_stats.options_checked == 1
        assert new_stats.api_fetch_failures == 0
        assert new_stats.api_errors == 0
        assert new_stats.cache_updates == 0
    
    def test_get_summary(self):
        """Test getting summary statistics."""
        stats = VolumeStats(
            positions_rejected_volume=5,
            options_checked=20,
            api_fetch_failures=2,
            api_errors=1,
            cache_updates=3
        )
        
        summary = stats.get_summary()
        
        assert summary['positions_rejected_volume'] == 5
        assert summary['options_checked'] == 20
        assert summary['api_fetch_failures'] == 2
        assert summary['api_errors'] == 1
        assert summary['cache_updates'] == 3
        assert summary['rejection_rate'] == 25.0  # 5/20 * 100
        assert summary['api_success_rate'] == 85.0  # (20-2-1)/20 * 100
    
    def test_get_summary_with_zero_options(self):
        """Test summary with zero options checked."""
        stats = VolumeStats()
        summary = stats.get_summary()
        
        assert summary['rejection_rate'] == 0.0
        assert summary['api_success_rate'] == 0.0 