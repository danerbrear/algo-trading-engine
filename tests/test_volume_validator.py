"""
Unit tests for volume validation functionality.

This module tests the volume validation logic, configuration, and statistics tracking.
"""

import pytest
from datetime import datetime
from unittest.mock import Mock
import dataclasses

from src.backtest.volume_validator import VolumeValidator
from src.backtest.config import VolumeConfig, VolumeStats
from src.common.models import Option, OptionType


class TestVolumeValidator:
    """Test cases for VolumeValidator static methods."""
    
    def test_validate_option_volume_with_sufficient_volume(self):
        """Test volume validation with sufficient volume."""
        option = Mock(spec=Option)
        option.volume = 15
        
        result = VolumeValidator.validate_option_volume(option, min_volume=10)
        assert result is True
    
    def test_validate_option_volume_with_insufficient_volume(self):
        """Test volume validation with insufficient volume."""
        option = Mock(spec=Option)
        option.volume = 5
        
        result = VolumeValidator.validate_option_volume(option, min_volume=10)
        assert result is False
    
    def test_validate_option_volume_with_exact_minimum(self):
        """Test volume validation with exact minimum volume."""
        option = Mock(spec=Option)
        option.volume = 10
        
        result = VolumeValidator.validate_option_volume(option, min_volume=10)
        assert result is True
    
    def test_validate_option_volume_with_no_volume_data(self):
        """Test volume validation with missing volume data."""
        option = Mock(spec=Option)
        option.volume = None
        
        result = VolumeValidator.validate_option_volume(option, min_volume=10)
        assert result is False
    
    def test_validate_option_volume_with_zero_volume(self):
        """Test volume validation with zero volume."""
        option = Mock(spec=Option)
        option.volume = 0
        
        result = VolumeValidator.validate_option_volume(option, min_volume=10)
        assert result is False
    
    def test_validate_spread_volume_with_all_valid_options(self):
        """Test spread volume validation with all valid options."""
        option1 = Mock(spec=Option)
        option1.volume = 15
        option2 = Mock(spec=Option)
        option2.volume = 20
        
        spread_options = [option1, option2]
        
        result = VolumeValidator.validate_spread_volume(spread_options, min_volume=10)
        assert result is True
    
    def test_validate_spread_volume_with_one_invalid_option(self):
        """Test spread volume validation with one invalid option."""
        option1 = Mock(spec=Option)
        option1.volume = 15
        option2 = Mock(spec=Option)
        option2.volume = 5  # Below minimum
        
        spread_options = [option1, option2]
        
        result = VolumeValidator.validate_spread_volume(spread_options, min_volume=10)
        assert result is False
    
    def test_validate_spread_volume_with_empty_list(self):
        """Test spread volume validation with empty options list."""
        spread_options = []
        
        result = VolumeValidator.validate_spread_volume(spread_options, min_volume=10)
        assert result is False
    
    def test_validate_spread_volume_with_none_list(self):
        """Test spread volume validation with None options list."""
        result = VolumeValidator.validate_spread_volume(None, min_volume=10)
        assert result is False
    
    def test_get_volume_status_with_valid_option(self):
        """Test getting volume status for a valid option."""
        option = Mock(spec=Option)
        option.symbol = "SPY240315C00500000"
        option.volume = 15
        option.strike = 500.0
        option.expiration = "2024-03-15"
        option.option_type = Mock()
        option.option_type.value = "C"
        
        status = VolumeValidator.get_volume_status(option)
        
        assert status['symbol'] == "SPY240315C00500000"
        assert status['volume'] == 15
        assert status['has_volume_data'] is True
        assert status['is_liquid'] is True
        assert status['strike'] == 500.0
        assert status['expiration'] == "2024-03-15"
        assert status['option_type'] == "C"
    
    def test_get_volume_status_with_invalid_option(self):
        """Test getting volume status for an invalid option."""
        option = Mock(spec=Option)
        option.symbol = "SPY240315C00500000"
        option.volume = 5
        option.strike = 500.0
        option.expiration = "2024-03-15"
        option.option_type = Mock()
        option.option_type.value = "C"
        
        status = VolumeValidator.get_volume_status(option)
        
        assert status['symbol'] == "SPY240315C00500000"
        assert status['volume'] == 5
        assert status['has_volume_data'] is True
        assert status['is_liquid'] is False
    
    def test_get_volume_status_with_no_volume_data(self):
        """Test getting volume status for option with no volume data."""
        option = Mock(spec=Option)
        option.symbol = "SPY240315C00500000"
        option.volume = None
        option.strike = 500.0
        option.expiration = "2024-03-15"
        option.option_type = Mock()
        option.option_type.value = "C"
        
        status = VolumeValidator.get_volume_status(option)
        
        assert status['symbol'] == "SPY240315C00500000"
        assert status['volume'] is None
        assert status['has_volume_data'] is False
        assert status['is_liquid'] is False
    
    def test_get_spread_volume_status_with_valid_spread(self):
        """Test getting volume status for a valid spread."""
        option1 = Mock(spec=Option)
        option1.symbol = "SPY240315C00500000"
        option1.volume = 15
        option1.strike = 500.0
        option1.expiration = "2024-03-15"
        option1.option_type = Mock()
        option1.option_type.value = "C"
        
        option2 = Mock(spec=Option)
        option2.symbol = "SPY240315C00510000"
        option2.volume = 20
        option2.strike = 510.0
        option2.expiration = "2024-03-15"
        option2.option_type = Mock()
        option2.option_type.value = "C"
        
        spread_options = [option1, option2]
        
        status = VolumeValidator.get_spread_volume_status(spread_options)
        
        assert status['is_valid'] is True
        assert status['total_options'] == 2
        assert status['valid_options'] == 2
        assert len(status['options_status']) == 2
        assert status['options_status'][0]['is_liquid'] is True
        assert status['options_status'][1]['is_liquid'] is True
    
    def test_get_spread_volume_status_with_invalid_spread(self):
        """Test getting volume status for an invalid spread."""
        option1 = Mock(spec=Option)
        option1.symbol = "SPY240315C00500000"
        option1.volume = 15
        option1.strike = 500.0
        option1.expiration = "2024-03-15"
        option1.option_type = Mock()
        option1.option_type.value = "C"
        
        option2 = Mock(spec=Option)
        option2.symbol = "SPY240315C00510000"
        option2.volume = 5  # Below minimum
        option2.strike = 510.0
        option2.expiration = "2024-03-15"
        option2.option_type = Mock()
        option2.option_type.value = "C"
        
        spread_options = [option1, option2]
        
        status = VolumeValidator.get_spread_volume_status(spread_options)
        
        assert status['is_valid'] is False
        assert status['total_options'] == 2
        assert status['valid_options'] == 1
        assert len(status['options_status']) == 2
        assert status['options_status'][0]['is_liquid'] is True
        assert status['options_status'][1]['is_liquid'] is False
    
    def test_get_spread_volume_status_with_empty_list(self):
        """Test getting volume status for empty spread options."""
        spread_options = []
        
        status = VolumeValidator.get_spread_volume_status(spread_options)
        
        assert status['is_valid'] is False
        assert status['reason'] == 'No options provided'
        assert status['options_status'] == []
        assert status['total_options'] == 0
        assert status['valid_options'] == 0


class TestVolumeConfig:
    """Test cases for VolumeConfig DTO."""
    
    def test_volume_config_default_values(self):
        """Test VolumeConfig with default values."""
        config = VolumeConfig()
        
        assert config.min_volume == 10
        assert config.enable_volume_validation is True
    
    def test_volume_config_custom_values(self):
        """Test VolumeConfig with custom values."""
        config = VolumeConfig(min_volume=20, enable_volume_validation=False)
        
        assert config.min_volume == 20
        assert config.enable_volume_validation is False
    
    def test_volume_config_negative_min_volume_raises_error(self):
        """Test that negative min_volume raises ValueError."""
        with pytest.raises(ValueError, match="Minimum volume cannot be negative"):
            VolumeConfig(min_volume=-5)
    
    def test_volume_config_zero_min_volume_raises_error(self):
        """Test that zero min_volume raises ValueError."""
        with pytest.raises(ValueError, match="Minimum volume must be greater than 0"):
            VolumeConfig(min_volume=0)
    
    def test_volume_config_immutability(self):
        """Test that VolumeConfig is immutable."""
        config = VolumeConfig()
        
        with pytest.raises(dataclasses.FrozenInstanceError):
            config.min_volume = 20


class TestVolumeStats:
    """Test cases for VolumeStats DTO."""
    
    def test_volume_stats_default_values(self):
        """Test VolumeStats with default values."""
        stats = VolumeStats()
        
        assert stats.positions_rejected_volume == 0
        assert stats.options_checked == 0
    
    def test_volume_stats_custom_values(self):
        """Test VolumeStats with custom values."""
        stats = VolumeStats(positions_rejected_volume=5, options_checked=10)
        
        assert stats.positions_rejected_volume == 5
        assert stats.options_checked == 10
    
    def test_increment_rejected_positions(self):
        """Test incrementing rejected positions count."""
        stats = VolumeStats(positions_rejected_volume=3, options_checked=10)
        
        new_stats = stats.increment_rejected_positions()
        
        assert new_stats.positions_rejected_volume == 4
        assert new_stats.options_checked == 10
        assert stats.positions_rejected_volume == 3  # Original unchanged
    
    def test_increment_options_checked(self):
        """Test incrementing options checked count."""
        stats = VolumeStats(positions_rejected_volume=3, options_checked=10)
        
        new_stats = stats.increment_options_checked()
        
        assert new_stats.positions_rejected_volume == 3
        assert new_stats.options_checked == 11
        assert stats.options_checked == 10  # Original unchanged
    
    def test_get_summary(self):
        """Test getting summary statistics."""
        stats = VolumeStats(positions_rejected_volume=5, options_checked=20)
        
        summary = stats.get_summary()
        
        assert summary['positions_rejected_volume'] == 5
        assert summary['options_checked'] == 20
        assert summary['rejection_rate'] == 25.0  # 5/20 * 100
    
    def test_get_summary_with_zero_options_checked(self):
        """Test getting summary with zero options checked."""
        stats = VolumeStats(positions_rejected_volume=0, options_checked=0)
        
        summary = stats.get_summary()
        
        assert summary['positions_rejected_volume'] == 0
        assert summary['options_checked'] == 0
        assert summary['rejection_rate'] == 0.0  # 0/1 * 100 (max(0, 1) = 1)
    
    def test_volume_stats_immutability(self):
        """Test that VolumeStats is immutable."""
        stats = VolumeStats()
        
        with pytest.raises(dataclasses.FrozenInstanceError):
            stats.positions_rejected_volume = 5


if __name__ == "__main__":
    pytest.main([__file__]) 