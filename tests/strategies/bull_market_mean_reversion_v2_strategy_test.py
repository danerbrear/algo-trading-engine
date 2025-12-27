import pytest
import pandas as pd
import numpy as np
from datetime import datetime
from unittest.mock import Mock, MagicMock, patch
from decimal import Decimal

from src.strategies.bull_market_mean_reversion_v2_strategy import BullMarketMeanReversionV2Strategy
from src.backtest.models import Position, StrategyType
from src.common.options_dtos import OptionContractDTO, OptionBarDTO
from src.common.models import OptionType


class TestBullMarketMeanReversionV2Strategy:
    """Test cases for BullMarketMeanReversionV2Strategy"""

    def setup_method(self):
        """Set up test data"""
        # Create a mock options handler
        self.mock_options_handler = Mock()
        self.mock_options_handler.symbol = "SPY"
        
        # Create strategy with default threshold of 1.5
        self.strategy = BullMarketMeanReversionV2Strategy(
            options_handler=self.mock_options_handler,
            z_score_entry_threshold=1.5
        )
        
        # Create sample market data with enough history for Z-Score calculation (60+ days)
        dates = pd.date_range('2024-01-01', '2024-03-31', freq='D')
        n_days = len(dates)
        
        # Create price data with upward trend and high z-scores
        base_price = 500.0
        prices = []
        for i in range(n_days):
            # Create upward trend with some volatility
            price = base_price + (i * 0.5) + np.sin(i * 0.1) * 2
            prices.append(price)
        
        self.market_data = pd.DataFrame({
            'Open': prices,
            'High': [p * 1.01 for p in prices],
            'Low': [p * 0.99 for p in prices],
            'Close': prices,
            'Volume': [1000000] * n_days
        }, index=dates)
        
        # Set the data
        self.strategy.set_data(self.market_data)
    
    def _create_mock_contract(self, strike: float, expiration: str, option_type: OptionType = OptionType.PUT) -> OptionContractDTO:
        """Helper to create mock option contracts"""
        return OptionContractDTO(
            ticker=f'O:SPY{expiration.replace("-", "")[:6]}P{int(strike * 1000):08d}',
            underlying_ticker='SPY',
            contract_type=option_type,
            strike_price=Decimal(str(strike)),
            expiration_date=expiration,
            exercise_style='american',
            shares_per_contract=100,
            primary_exchange='BATO',
            cfi='OCASPS',
            additional_underlyings=None
        )
    
    def _create_mock_bar(self, close_price: float, volume: int = 100) -> OptionBarDTO:
        """Helper to create mock option bar data"""
        return OptionBarDTO(
            ticker="SPY",
            timestamp=datetime(2024, 3, 15),
            open_price=Decimal(str(close_price * 0.99)),
            high_price=Decimal(str(close_price * 1.01)),
            low_price=Decimal(str(close_price * 0.98)),
            close_price=Decimal(str(close_price)),
            volume=volume,
            volume_weighted_avg_price=Decimal(str(close_price)),
            number_of_transactions=100
        )
    
    def test_z_score_entry_threshold_default_value(self):
        """Test that default z-score entry threshold is 1.0"""
        strategy = BullMarketMeanReversionV2Strategy(options_handler=self.mock_options_handler)
        assert strategy.z_score_entry_threshold == 1.0
    
    def test_z_score_entry_threshold_custom_value(self):
        """Test that custom z-score entry threshold can be set"""
        strategy = BullMarketMeanReversionV2Strategy(
            options_handler=self.mock_options_handler,
            z_score_entry_threshold=2.0
        )
        assert strategy.z_score_entry_threshold == 2.0
    
    def test_entry_signal_rejected_when_z_score_below_threshold(self):
        """Test that entry signal is NOT triggered when z-score is below threshold"""
        # Find a date with enough history
        test_date = self.market_data.index[70]  # After 60 days for Z-Score calculation
        
        # Manually set a low z-score (below 1.5 threshold)
        self.strategy.data.loc[test_date, 'Z_Score'] = 1.2
        
        # Ensure upward trend conditions are met
        self.strategy.data.loc[test_date, 'SMA_15'] = 520.0
        self.strategy.data.loc[test_date, 'SMA_30'] = 510.0
        self.strategy.data.loc[test_date, 'SMA_Width'] = 10.0
        self.strategy.data.loc[test_date, 'SMA_Width_Change'] = 0.5  # Increasing
        
        # Check entry signal
        has_signal = self.strategy._has_entry_signal(test_date)
        
        assert has_signal == False, "Entry signal should be rejected when z-score (1.2) is below threshold (1.5)"
    
    def test_entry_signal_rejected_when_z_score_equals_threshold(self):
        """Test that entry signal is NOT triggered when z-score equals threshold (boundary condition)"""
        test_date = self.market_data.index[70]
        
        # Set z-score exactly at threshold
        self.strategy.data.loc[test_date, 'Z_Score'] = 1.5
        
        # Ensure upward trend conditions are met
        self.strategy.data.loc[test_date, 'SMA_15'] = 520.0
        self.strategy.data.loc[test_date, 'SMA_30'] = 510.0
        self.strategy.data.loc[test_date, 'SMA_Width'] = 10.0
        self.strategy.data.loc[test_date, 'SMA_Width_Change'] = 0.5
        
        # Check entry signal
        has_signal = self.strategy._has_entry_signal(test_date)
        
        assert has_signal == False, "Entry signal should be rejected when z-score (1.5) equals threshold (1.5)"
    
    def test_entry_signal_accepted_when_z_score_above_threshold(self):
        """Test that entry signal IS triggered when z-score is above threshold"""
        test_date = self.market_data.index[70]
        
        # Set z-score above threshold
        self.strategy.data.loc[test_date, 'Z_Score'] = 1.6
        
        # Ensure upward trend conditions are met
        self.strategy.data.loc[test_date, 'SMA_15'] = 520.0
        self.strategy.data.loc[test_date, 'SMA_30'] = 510.0
        self.strategy.data.loc[test_date, 'SMA_Width'] = 10.0
        self.strategy.data.loc[test_date, 'SMA_Width_Change'] = 0.5
        
        # Check entry signal
        has_signal = self.strategy._has_entry_signal(test_date)
        
        assert has_signal == True, "Entry signal should be accepted when z-score (1.6) is above threshold (1.5)"
    
    def test_entry_signal_with_different_threshold_values(self):
        """Test entry signal behavior with different threshold values"""
        test_date = self.market_data.index[70]
        z_score = 1.8
        
        # Test with threshold 1.5 (should accept)
        strategy_low = BullMarketMeanReversionV2Strategy(
            options_handler=self.mock_options_handler,
            z_score_entry_threshold=1.5
        )
        strategy_low.set_data(self.market_data)
        strategy_low.data.loc[test_date, 'Z_Score'] = z_score
        strategy_low.data.loc[test_date, 'SMA_15'] = 520.0
        strategy_low.data.loc[test_date, 'SMA_30'] = 510.0
        strategy_low.data.loc[test_date, 'SMA_Width'] = 10.0
        strategy_low.data.loc[test_date, 'SMA_Width_Change'] = 0.5
        
        assert strategy_low._has_entry_signal(test_date) == True, \
            "Should accept z-score 1.8 with threshold 1.5"
        
        # Test with threshold 2.0 (should reject)
        strategy_high = BullMarketMeanReversionV2Strategy(
            options_handler=self.mock_options_handler,
            z_score_entry_threshold=2.0
        )
        strategy_high.set_data(self.market_data)
        strategy_high.data.loc[test_date, 'Z_Score'] = z_score
        strategy_high.data.loc[test_date, 'SMA_15'] = 520.0
        strategy_high.data.loc[test_date, 'SMA_30'] = 510.0
        strategy_high.data.loc[test_date, 'SMA_Width'] = 10.0
        strategy_high.data.loc[test_date, 'SMA_Width_Change'] = 0.5
        
        assert strategy_high._has_entry_signal(test_date) == False, \
            "Should reject z-score 1.8 with threshold 2.0"
    
    def test_entry_signal_rejected_when_z_score_is_nan(self):
        """Test that entry signal is rejected when z-score is NaN"""
        test_date = self.market_data.index[70]
        
        # Set z-score to NaN
        self.strategy.data.loc[test_date, 'Z_Score'] = np.nan
        
        # Ensure upward trend conditions are met
        self.strategy.data.loc[test_date, 'SMA_15'] = 520.0
        self.strategy.data.loc[test_date, 'SMA_30'] = 510.0
        self.strategy.data.loc[test_date, 'SMA_Width'] = 10.0
        self.strategy.data.loc[test_date, 'SMA_Width_Change'] = 0.5
        
        # Check entry signal
        has_signal = self.strategy._has_entry_signal(test_date)
        
        assert has_signal == False, "Entry signal should be rejected when z-score is NaN"
    
    def test_z_score_threshold_validation_in_full_entry_check(self):
        """Test that z-score threshold is properly checked in the full entry signal logic"""
        test_date = self.market_data.index[70]
        
        # Set up all conditions except z-score
        self.strategy.data.loc[test_date, 'SMA_15'] = 520.0
        self.strategy.data.loc[test_date, 'SMA_30'] = 510.0
        self.strategy.data.loc[test_date, 'SMA_Width'] = 10.0
        self.strategy.data.loc[test_date, 'SMA_Width_Change'] = 0.5
        
        # Test with z-score below threshold
        self.strategy.data.loc[test_date, 'Z_Score'] = 1.4
        assert self.strategy._has_entry_signal(test_date) == False
        
        # Test with z-score at threshold
        self.strategy.data.loc[test_date, 'Z_Score'] = 1.5
        assert self.strategy._has_entry_signal(test_date) == False
        
        # Test with z-score above threshold
        self.strategy.data.loc[test_date, 'Z_Score'] = 1.6
        assert self.strategy._has_entry_signal(test_date) == True
    
    def test_z_score_threshold_with_other_entry_conditions(self):
        """Test that z-score threshold works correctly with other entry conditions"""
        test_date = self.market_data.index[70]
        
        # Case 1: All conditions met including z-score above threshold
        self.strategy.data.loc[test_date, 'SMA_15'] = 520.0
        self.strategy.data.loc[test_date, 'SMA_30'] = 510.0  # SMA15 > SMA30 ✓
        self.strategy.data.loc[test_date, 'SMA_Width'] = 10.0
        self.strategy.data.loc[test_date, 'SMA_Width_Change'] = 0.5  # Width increasing ✓
        self.strategy.data.loc[test_date, 'Z_Score'] = 1.6  # Above threshold ✓
        
        assert self.strategy._has_entry_signal(test_date) == True
        
        # Case 2: All conditions met except z-score below threshold
        self.strategy.data.loc[test_date, 'Z_Score'] = 1.4  # Below threshold ✗
        
        assert self.strategy._has_entry_signal(test_date) == False
        
        # Case 3: Z-score above threshold but no upward trend
        self.strategy.data.loc[test_date, 'SMA_15'] = 510.0
        self.strategy.data.loc[test_date, 'SMA_30'] = 520.0  # SMA15 < SMA30 ✗
        self.strategy.data.loc[test_date, 'Z_Score'] = 1.6  # Above threshold ✓
        
        assert self.strategy._has_entry_signal(test_date) == False
    
    def test_long_put_exit_price_calculation(self):
        """Test that exit price is correctly calculated for long put positions"""
        from src.common.models import Option
        
        # Create a mock position with long put
        test_date = datetime(2024, 3, 15)
        expiration_date = datetime(2024, 3, 25)
        
        # Create mock option for long put
        put_option = Option(
            ticker="O:SPY240325P00500000",
            symbol="SPY",
            expiration="2024-03-25",
            strike=500.0,
            option_type=OptionType.PUT,
            last_price=2.50,
            bid=2.45,
            ask=2.55,
            volume=100,
            open_interest=1000
        )
        
        # Create position (long put)
        # Entry: Buy put and pay premium (entry_price = premium_paid, positive)
        position = Position(
            symbol="SPY",
            expiration_date=expiration_date,
            strategy_type=StrategyType.LONG_PUT,
            strike_price=500.0,
            entry_date=datetime(2024, 3, 10),
            entry_price=2.50,  # Premium paid (positive)
            spread_options=[put_option]
        )
        position.set_quantity(1)
        
        # Mock option bar data for exit
        # Exit: Sell put and receive premium (exit_price = premium_received, positive)
        put_bar = OptionBarDTO(
            ticker="O:SPY240325P00500000",
            timestamp=test_date,
            open_price=Decimal("1.25"),
            high_price=Decimal("1.30"),
            low_price=Decimal("1.15"),
            close_price=Decimal("1.20"),
            volume=150,
            volume_weighted_avg_price=Decimal("1.20"),
            number_of_transactions=50
        )
        
        # Mock the options_handler to return bar data
        self.mock_options_handler.get_option_bar = Mock(return_value=put_bar)
        
        # Calculate exit price
        exit_price, has_error = self.strategy._compute_exit_price(test_date, position)
        
        # Should not have error
        assert has_error == False, "Exit price calculation should not have error"
        assert exit_price is not None, "Exit price should not be None"
        
        # Expected: exit_price = premium received when selling = 1.20
        expected_exit_price = 1.20
        assert abs(exit_price - expected_exit_price) < 0.01, \
            f"Exit price should be {expected_exit_price}, got {exit_price}"
    
    def test_long_put_capital_calculation_on_close(self):
        """Test that capital is correctly updated when closing a long put position"""
        from src.backtest.main import BacktestEngine
        from src.common.models import Option
        
        # Create a mock position with long put
        test_date = datetime(2024, 3, 15)
        expiration_date = datetime(2024, 3, 25)
        
        # Create mock option
        put_option = Option(
            ticker="O:SPY240325P00500000",
            symbol="SPY",
            expiration="2024-03-25",
            strike=500.0,
            option_type=OptionType.PUT,
            last_price=2.50,
            bid=2.45,
            ask=2.55,
            volume=100,
            open_interest=1000
        )
        
        # Create position with entry premium of $2.50 per contract
        position = Position(
            symbol="SPY",
            expiration_date=expiration_date,
            strategy_type=StrategyType.LONG_PUT,
            strike_price=500.0,
            entry_date=datetime(2024, 3, 10),
            entry_price=2.50,  # Premium paid (positive)
            spread_options=[put_option]
        )
        position.set_quantity(7)  # 7 contracts
        
        # Create minimal strategy and data for backtest engine
        mock_options_handler = Mock()
        strategy = BullMarketMeanReversionV2Strategy(
            options_handler=mock_options_handler,
            max_risk_per_trade=0.35  # 35% to allow 7 contracts: (7 * 250) / 3000 = 0.58, but we'll use 0.35
        )
        
        dates = pd.date_range('2024-01-01', '2024-03-31', freq='D')
        data = pd.DataFrame({
            'Open': [500.0] * len(dates),
            'High': [501.0] * len(dates),
            'Low': [499.0] * len(dates),
            'Close': [500.0] * len(dates),
            'Volume': [1000000] * len(dates)
        }, index=dates)
        
        strategy.set_data(data)
        
        # Create backtest engine with initial capital
        initial_capital = 3000.0
        backtest_engine = BacktestEngine(
            data=data,
            strategy=strategy,
            initial_capital=initial_capital,
            start_date=test_date,
            end_date=test_date,
            quiet_mode=True
        )
        
        # Add position (should subtract premium paid)
        # Note: _add_position will recalculate quantity based on capital and max_risk_per_trade
        backtest_engine._add_position(position)
        
        # Get the actual quantity that was calculated (max_risk_allowed = 3000 * 0.35 = 1050, max_risk_per_contract = 2.50 * 100 = 250, quantity = 1050 / 250 = 4)
        actual_quantity = position.quantity
        
        # Expected capital after entry: 3000 - (2.50 * actual_quantity * 100)
        expected_capital_after_entry = initial_capital - (2.50 * actual_quantity * 100)
        assert abs(backtest_engine.capital - expected_capital_after_entry) < 0.01, \
            f"Capital after entry should be {expected_capital_after_entry}, got {backtest_engine.capital}"
        
        # Close position with exit price (premium received when selling = 1.20)
        exit_price = 1.20
        backtest_engine._remove_position(test_date, position, exit_price)
        
        # Expected capital after exit: expected_capital_after_entry + (1.20 * actual_quantity * 100)
        expected_capital_after_exit = expected_capital_after_entry + (1.20 * actual_quantity * 100)
        assert abs(backtest_engine.capital - expected_capital_after_exit) < 0.01, \
            f"Capital after exit should be {expected_capital_after_exit}, got {backtest_engine.capital}"
        
        # Verify exit price is not zero when it should have a value
        assert exit_price != 0.0, "Exit price should not be 0.0 when closing a position with value"
    
    def test_long_put_exit_price_not_zero_at_2_dte(self):
        """Test that exit price is calculated (not 0.0) when closing at 2 DTE"""
        from src.common.models import Option
        
        test_date = datetime(2024, 3, 15)
        expiration_date = datetime(2024, 3, 17)  # 2 days to expiration
        
        # Create mock option
        put_option = Option(
            ticker="O:SPY240317P00500000",
            symbol="SPY",
            expiration="2024-03-17",
            strike=500.0,
            option_type=OptionType.PUT,
            last_price=1.50,
            bid=1.45,
            ask=1.55,
            volume=100,
            open_interest=1000
        )
        
        position = Position(
            symbol="SPY",
            expiration_date=expiration_date,
            strategy_type=StrategyType.LONG_PUT,
            strike_price=500.0,
            entry_date=datetime(2024, 3, 10),
            entry_price=1.50,  # Premium paid (positive)
            spread_options=[put_option]
        )
        position.set_quantity(1)
        
        # Mock bar data for 2 DTE
        put_bar = OptionBarDTO(
            ticker="O:SPY240317P00500000",
            timestamp=test_date,
            open_price=Decimal("1.50"),
            high_price=Decimal("1.55"),
            low_price=Decimal("1.45"),
            close_price=Decimal("1.20"),
            volume=150,
            volume_weighted_avg_price=Decimal("1.20"),
            number_of_transactions=50
        )
        
        self.mock_options_handler.get_option_bar = Mock(return_value=put_bar)
        
        # Calculate exit price at 2 DTE
        exit_price, has_error = self.strategy._compute_exit_price(test_date, position)
        
        # Should calculate actual exit price, not return 0.0
        assert has_error == False, "Should not have error calculating exit price at 2 DTE"
        assert exit_price is not None, "Exit price should not be None"
        assert exit_price != 0.0, "Exit price should not be 0.0 when option prices are available"
        
        # Expected: exit_price = premium received when selling long put = 1.20
        expected_exit_price = 1.20
        assert abs(exit_price - expected_exit_price) < 0.01, \
            f"Exit price at 2 DTE should be {expected_exit_price}, got {exit_price}"

