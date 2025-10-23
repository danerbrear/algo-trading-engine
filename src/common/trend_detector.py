"""
Trend Detection Utilities

This module provides common trend detection functionality for trading strategies.
"""

import pandas as pd
from dataclasses import dataclass
from typing import Tuple, List
from datetime import datetime


@dataclass
class TrendInfo:
    """Information about a detected trend."""
    start_date: datetime
    end_date: datetime
    duration: int
    start_price: float
    end_price: float
    net_return: float
    reversal_drawdown: float = 0.0
    reversal_date: datetime = None


class TrendDetector:
    """
    Utility class for detecting price trends in market data.
    
    Provides methods for both forward-looking trend detection (finding completed trends)
    and backward-looking trend validation (checking if a signal is part of a trend).
    """
    
    @staticmethod
    def detect_forward_trends(
        data: pd.DataFrame,
        min_duration: int = 3,
        max_duration: int = 10,
        reversal_threshold: float = 0.02,
        require_reversal: bool = True
    ) -> List[TrendInfo]:
        """
        Detect completed upward trends by scanning forward through data.
        
        An upward trend is defined as consecutive days of positive returns.
        Optionally requires a negative day at the end to signal reversal.
        
        Args:
            data: DataFrame with 'Close' column
            min_duration: Minimum number of consecutive positive days
            max_duration: Maximum trend duration to consider
            reversal_threshold: Maximum drawdown allowed within trend (default: 2%)
            require_reversal: If True, only include trends ending with a negative day
            
        Returns:
            List of TrendInfo objects for detected trends
        """
        trends = []
        
        # Calculate daily returns
        returns = data['Close'].pct_change()
        
        i = 1  # Start from second day (need previous day for return)
        while i < len(data) - 1:
            # Look for start of positive returns
            if returns.iloc[i] > 0:
                trend_start_idx = i
                trend_duration = 1
                
                # Count consecutive positive days
                j = i + 1
                while j < len(data) and returns.iloc[j] > 0:
                    trend_duration += 1
                    j += 1
                
                # Check if we found a trend of appropriate duration
                valid_duration = min_duration <= trend_duration <= max_duration
                has_reversal = j < len(data) and returns.iloc[j] < 0
                
                if valid_duration and (not require_reversal or has_reversal):
                    trend_end_idx = j - 1  # Last positive day
                    
                    start_date = data.index[trend_start_idx]
                    end_date = data.index[trend_end_idx]
                    start_price = data['Close'].iloc[trend_start_idx]
                    end_price = data['Close'].iloc[trend_end_idx]
                    net_return = (end_price - start_price) / start_price
                    
                    # Check for sustained trend (no significant reversals)
                    trend_sustained = TrendDetector._is_trend_sustained(
                        data, trend_start_idx, trend_end_idx, reversal_threshold
                    )
                    
                    # Only include trends with net positive returns and no reversals
                    if net_return > 0 and trend_sustained:
                        # Calculate reversal info if there's a negative day after
                        reversal_info = {}
                        if has_reversal:
                            reversal_idx = j
                            reversal_date = data.index[reversal_idx]
                            reversal_price = data['Close'].iloc[reversal_idx]
                            reversal_drawdown = (reversal_price - end_price) / end_price
                            reversal_info = {
                                'reversal_drawdown': reversal_drawdown,
                                'reversal_date': reversal_date
                            }
                        
                        trend_info = TrendInfo(
                            start_date=start_date,
                            end_date=end_date,
                            duration=trend_duration,
                            start_price=start_price,
                            end_price=end_price,
                            net_return=net_return,
                            **reversal_info
                        )
                        trends.append(trend_info)
                
                # Move past this trend
                i = j
            else:
                i += 1
        
        return trends
    
    @staticmethod
    def check_backward_trend(
        data: pd.DataFrame,
        signal_index: int,
        min_duration: int = 3,
        max_duration: int = 60,
        reversal_threshold: float = 0.02
    ) -> Tuple[bool, int, float]:
        """
        Check for upward trend by looking backward from a signal point.
        
        This is useful for validating if a signal (e.g., velocity increase) 
        occurred during a sustained upward trend.
        
        Args:
            data: DataFrame with 'Close' column
            signal_index: Index of the signal to validate
            min_duration: Minimum trend duration in days
            max_duration: Maximum trend duration in days
            reversal_threshold: Maximum drawdown allowed within trend (default: 2%)
            
        Returns:
            Tuple of (success, duration, return)
        """
        current_price = data['Close'].iloc[signal_index]
        
        # Look backward for trend continuation
        for duration in range(min_duration, min(max_duration + 1, signal_index + 1)):
            start_index = signal_index - duration
            if start_index < 0:
                break
            
            start_price = data['Close'].iloc[start_index]
            total_return = (current_price - start_price) / start_price
            
            # Check for upward trend
            if total_return > 0:
                # Verify sustained trend (no significant reversals)
                trend_sustained = TrendDetector._is_trend_sustained(
                    data, start_index, signal_index, reversal_threshold
                )
                
                if trend_sustained:
                    return True, duration, total_return
        
        return False, 0, 0.0
    
    @staticmethod
    def check_reversal_at_index(
        data: pd.DataFrame,
        date_index: int,
        min_trend_duration: int = 3,
        max_trend_duration: int = 10,
        reversal_threshold: float = 0.02
    ) -> Tuple[bool, TrendInfo]:
        """
        Check if a specific date represents a trend reversal.
        
        This is more efficient than scanning all trends when you only care about
        whether TODAY is a reversal day. Used for real-time signal detection.
        
        Args:
            data: DataFrame with 'Close' column
            date_index: Index to check for reversal
            min_trend_duration: Minimum upward trend duration
            max_trend_duration: Maximum upward trend duration
            reversal_threshold: Maximum drawdown allowed within trend
            
        Returns:
            Tuple of (is_reversal, TrendInfo or None)
        """
        # Need at least min_trend_duration days before current date
        if date_index < min_trend_duration:
            return False, None
        
        # Calculate returns
        returns = data['Close'].pct_change()
        
        # Check if current day is negative (reversal candidate)
        if date_index >= len(returns) or returns.iloc[date_index] >= 0:
            return False, None
        
        # Look backward to find the upward trend
        # Count consecutive positive days before current index
        consecutive_positive = 0
        lookback_idx = date_index - 1
        
        while (lookback_idx >= 0 and 
               returns.iloc[lookback_idx] > 0 and 
               consecutive_positive < max_trend_duration):
            consecutive_positive += 1
            lookback_idx -= 1
        
        # Check if we found a valid trend duration
        if consecutive_positive < min_trend_duration:
            return False, None
        
        if consecutive_positive > max_trend_duration:
            # Trim to max_trend_duration
            consecutive_positive = max_trend_duration
            lookback_idx = date_index - max_trend_duration - 1
        
        # Found a potential trend - validate it's sustained
        trend_start_idx = lookback_idx + 1
        trend_end_idx = date_index - 1
        
        # Check for sustained trend
        trend_sustained = TrendDetector._is_trend_sustained(
            data, trend_start_idx, trend_end_idx, reversal_threshold
        )
        
        if not trend_sustained:
            return False, None
        
        # Calculate trend info
        start_date = data.index[trend_start_idx]
        end_date = data.index[trend_end_idx]
        reversal_date = data.index[date_index]
        
        start_price = data['Close'].iloc[trend_start_idx]
        end_price = data['Close'].iloc[trend_end_idx]
        reversal_price = data['Close'].iloc[date_index]
        
        net_return = (end_price - start_price) / start_price
        reversal_drawdown = (reversal_price - end_price) / end_price
        
        # Only return if net return is positive
        if net_return <= 0:
            return False, None
        
        trend_info = TrendInfo(
            start_date=start_date,
            end_date=end_date,
            duration=consecutive_positive,
            start_price=start_price,
            end_price=end_price,
            net_return=net_return,
            reversal_drawdown=reversal_drawdown,
            reversal_date=reversal_date
        )
        
        return True, trend_info
    
    @staticmethod
    def _is_trend_sustained(
        data: pd.DataFrame,
        start_idx: int,
        end_idx: int,
        reversal_threshold: float
    ) -> bool:
        """
        Check if a trend is sustained without significant reversals.
        
        Args:
            data: DataFrame with 'Close' column
            start_idx: Start index of trend
            end_idx: End index of trend
            reversal_threshold: Maximum allowed drawdown from start price
            
        Returns:
            True if trend is sustained, False otherwise
        """
        start_price = data['Close'].iloc[start_idx]
        
        # Check each day within the trend
        for j in range(start_idx + 1, end_idx):
            current_price = data['Close'].iloc[j]
            current_return = (current_price - start_price) / start_price
            
            # If we see a reversal greater than threshold, trend is not sustained
            if current_return < -reversal_threshold:
                return False
        
        return True

