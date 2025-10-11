"""
Unit tests for volume validation functionality.

This module tests the volume validation logic, configuration, and statistics tracking.
"""

import pytest
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
        assert config.skip_closure_on_insufficient_volume is True
    
    def test_volume_config_custom_values(self):
        """Test VolumeConfig with custom values."""
        config = VolumeConfig(min_volume=20, enable_volume_validation=False, skip_closure_on_insufficient_volume=False)
        
        assert config.min_volume == 20
        assert config.enable_volume_validation is False
        assert config.skip_closure_on_insufficient_volume is False
    
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
        assert stats.positions_rejected_closure_volume == 0
        assert stats.options_checked == 0
        assert stats.skipped_closures == 0
    
    def test_volume_stats_custom_values(self):
        """Test VolumeStats with custom values."""
        stats = VolumeStats(positions_rejected_volume=5, positions_rejected_closure_volume=3, options_checked=10, skipped_closures=2)
        
        assert stats.positions_rejected_volume == 5
        assert stats.positions_rejected_closure_volume == 3
        assert stats.options_checked == 10
        assert stats.skipped_closures == 2
    
    def test_increment_rejected_positions(self):
        """Test incrementing rejected positions count."""
        stats = VolumeStats(positions_rejected_volume=3, options_checked=10)
        
        new_stats = stats.increment_rejected_positions()
        
        assert new_stats.positions_rejected_volume == 4
        assert new_stats.positions_rejected_closure_volume == 0
        assert new_stats.options_checked == 10
        assert new_stats.skipped_closures == 0
        assert stats.positions_rejected_volume == 3  # Original unchanged
    
    def test_increment_rejected_closures(self):
        """Test incrementing rejected closures count."""
        stats = VolumeStats(positions_rejected_volume=3, positions_rejected_closure_volume=1, options_checked=10, skipped_closures=1)
        
        new_stats = stats.increment_rejected_closures()
        
        assert new_stats.positions_rejected_volume == 3
        assert new_stats.positions_rejected_closure_volume == 2
        assert new_stats.options_checked == 10
        assert new_stats.skipped_closures == 2
        assert stats.positions_rejected_closure_volume == 1  # Original unchanged
    
    def test_increment_skipped_closures(self):
        """Test incrementing skipped closures count."""
        stats = VolumeStats(positions_rejected_volume=3, positions_rejected_closure_volume=1, options_checked=10, skipped_closures=1)
        
        new_stats = stats.increment_skipped_closures()
        
        assert new_stats.positions_rejected_volume == 3
        assert new_stats.positions_rejected_closure_volume == 1
        assert new_stats.options_checked == 10
        assert new_stats.skipped_closures == 2
        assert stats.skipped_closures == 1  # Original unchanged
    
    def test_increment_options_checked(self):
        """Test incrementing options checked count."""
        stats = VolumeStats(positions_rejected_volume=3, options_checked=10)
        
        new_stats = stats.increment_options_checked()
        
        assert new_stats.positions_rejected_volume == 3
        assert new_stats.options_checked == 11
        assert stats.options_checked == 10  # Original unchanged
    
    def test_get_summary(self):
        """Test getting summary statistics."""
        stats = VolumeStats(positions_rejected_volume=5, positions_rejected_closure_volume=3, options_checked=20, skipped_closures=2)
        
        summary = stats.get_summary()
        
        assert summary['positions_rejected_volume'] == 5
        assert summary['positions_rejected_closure_volume'] == 3
        assert summary['options_checked'] == 20
        assert summary['skipped_closures'] == 2
        assert summary['total_rejections'] == 8  # 5 + 3
        assert summary['rejection_rate'] == 40.0  # 8/20 * 100
    
    def test_get_summary_with_zero_options_checked(self):
        """Test getting summary with zero options checked."""
        stats = VolumeStats(positions_rejected_volume=0, positions_rejected_closure_volume=0, options_checked=0, skipped_closures=0)
        
        summary = stats.get_summary()
        
        assert summary['positions_rejected_volume'] == 0
        assert summary['positions_rejected_closure_volume'] == 0
        assert summary['options_checked'] == 0
        assert summary['skipped_closures'] == 0
        assert summary['total_rejections'] == 0
        assert summary['rejection_rate'] == 0.0  # 0/1 * 100 (max(0, 1) = 1)
    
    def test_volume_stats_immutability(self):
        """Test that VolumeStats is immutable."""
        stats = VolumeStats()
        
        with pytest.raises(dataclasses.FrozenInstanceError):
            stats.positions_rejected_volume = 5


class TestVolumeValidationLogic:
    """Test cases for volume validation logic with real Option objects."""
    
    def test_volume_validation_with_sufficient_volume(self):
        """Test volume validation with sufficient volume."""
        high_volume_option = Option(
            ticker="SPY",
            symbol="SPY240119C00100000",
            strike=100.0,
            expiration="2024-01-19",
            option_type=OptionType.CALL,
            last_price=1.50,
            volume=15,  # Sufficient volume
            open_interest=100,
            bid=1.45,
            ask=1.55
        )
        
        config = VolumeConfig(min_volume=10)
        
        # High volume should pass
        assert high_volume_option.volume >= config.min_volume
        assert VolumeValidator.validate_option_volume(high_volume_option, config.min_volume) is True
    
    def test_volume_validation_with_insufficient_volume(self):
        """Test volume validation with insufficient volume."""
        low_volume_option = Option(
            ticker="SPY",
            symbol="SPY240119C00100000",
            strike=100.0,
            expiration="2024-01-19",
            option_type=OptionType.CALL,
            last_price=1.50,
            volume=5,  # Insufficient volume
            open_interest=100,
            bid=1.45,
            ask=1.55
        )
        
        config = VolumeConfig(min_volume=10)
        
        # Low volume should fail
        assert low_volume_option.volume < config.min_volume
        assert VolumeValidator.validate_option_volume(low_volume_option, config.min_volume) is False
    
    def test_volume_validation_with_none_volume(self):
        """Test volume validation with None volume."""
        none_volume_option = Option(
            ticker="SPY",
            symbol="SPY240119C00100000",
            strike=100.0,
            expiration="2024-01-19",
            option_type=OptionType.CALL,
            last_price=1.50,
            volume=None,  # No volume data
            open_interest=100,
            bid=1.45,
            ask=1.55
        )
        
        config = VolumeConfig(min_volume=10)
        
        assert none_volume_option.volume is None
        assert VolumeValidator.validate_option_volume(none_volume_option, config.min_volume) is False
    
    def test_spread_volume_validation_with_mixed_volume(self):
        """Test spread volume validation with mixed volume levels."""
        high_volume_option = Option(
            ticker="SPY",
            symbol="SPY240119C00100000",
            strike=100.0,
            expiration="2024-01-19",
            option_type=OptionType.CALL,
            last_price=1.50,
            volume=15,  # Sufficient volume
            open_interest=100,
            bid=1.45,
            ask=1.55
        )
        
        low_volume_option = Option(
            ticker="SPY",
            symbol="SPY240119C00110000",
            strike=110.0,
            expiration="2024-01-19",
            option_type=OptionType.CALL,
            last_price=0.50,
            volume=5,  # Insufficient volume
            open_interest=50,
            bid=0.45,
            ask=0.55
        )
        
        spread_options = [high_volume_option, low_volume_option]
        
        # Spread should fail because one option has insufficient volume
        assert VolumeValidator.validate_spread_volume(spread_options, min_volume=10) is False
    
    def test_spread_volume_validation_with_all_sufficient_volume(self):
        """Test spread volume validation with all options having sufficient volume."""
        option1 = Option(
            ticker="SPY",
            symbol="SPY240119C00100000",
            strike=100.0,
            expiration="2024-01-19",
            option_type=OptionType.CALL,
            last_price=1.50,
            volume=15,  # Sufficient volume
            open_interest=100,
            bid=1.45,
            ask=1.55
        )
        
        option2 = Option(
            ticker="SPY",
            symbol="SPY240119C00110000",
            strike=110.0,
            expiration="2024-01-19",
            option_type=OptionType.CALL,
            last_price=0.50,
            volume=12,  # Sufficient volume
            open_interest=50,
            bid=0.45,
            ask=0.55
        )
        
        spread_options = [option1, option2]
        
        # Spread should pass because all options have sufficient volume
        assert VolumeValidator.validate_spread_volume(spread_options, min_volume=10) is True


if __name__ == "__main__":
    pytest.main([__file__]) 