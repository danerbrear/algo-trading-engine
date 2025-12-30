"""
Volume validation module for backtesting system.

This module provides volume validation functionality to ensure options have sufficient
liquidity for trading during backtests.
"""

from typing import List
from algo_trading_engine.common.models import Option


class VolumeValidator:
    """
    Static utility class for volume validation logic.
    
    Provides methods to validate option volume requirements and check
    liquidity thresholds for trading strategies.
    """
    
    @staticmethod
    def validate_option_volume(option: Option, min_volume: int = 10) -> bool:
        """
        Validate if an option has sufficient volume for trading.
        
        Args:
            option: The option to validate
            min_volume: Minimum required volume (default: 10)
            
        Returns:
            bool: True if volume is sufficient, False otherwise
        """
        if option.volume is None:
            return False
        return option.volume >= min_volume
    
    @staticmethod
    def validate_spread_volume(spread_options: List[Option], min_volume: int = 10) -> bool:
        """
        Validate if all options in a spread have sufficient volume.
        
        Args:
            spread_options: List of options that make up the spread
            min_volume: Minimum required volume for each option (default: 10)
            
        Returns:
            bool: True if all options have sufficient volume, False otherwise
        """
        if not spread_options:
            return False
            
        for option in spread_options:
            if not VolumeValidator.validate_option_volume(option, min_volume):
                return False
        return True
    
    @staticmethod
    def get_volume_status(option: Option) -> dict:
        """
        Get detailed volume information for an option.
        
        Args:
            option: The option to get volume status for
            
        Returns:
            dict: Dictionary containing volume status information
        """
        return {
            'symbol': option.symbol,
            'volume': option.volume,
            'has_volume_data': option.volume is not None,
            'is_liquid': VolumeValidator.validate_option_volume(option),
            'strike': option.strike,
            'expiration': option.expiration,
            'option_type': option.option_type.value
        }
    
    @staticmethod
    def get_spread_volume_status(spread_options: List[Option]) -> dict:
        """
        Get detailed volume status for all options in a spread.
        
        Args:
            spread_options: List of options that make up the spread
            
        Returns:
            dict: Dictionary containing volume status for the entire spread
        """
        if not spread_options:
            return {
                'is_valid': False,
                'reason': 'No options provided',
                'options_status': [],
                'total_options': 0,
                'valid_options': 0
            }
        
        options_status = []
        all_valid = True
        
        for option in spread_options:
            option_status = VolumeValidator.get_volume_status(option)
            options_status.append(option_status)
            
            if not option_status['is_liquid']:
                all_valid = False
        
        return {
            'is_valid': all_valid,
            'options_status': options_status,
            'total_options': len(spread_options),
            'valid_options': sum(1 for status in options_status if status['is_liquid'])
        } 