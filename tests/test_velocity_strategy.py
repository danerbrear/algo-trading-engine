import pytest
import pandas as pd
from datetime import datetime, timedelta
from decimal import Decimal
from unittest.mock import Mock, MagicMock

from src.strategies.velocity_signal_momentum_strategy import VelocitySignalMomentumStrategy
from src.backtest.models import Position, StrategyType
from src.common.models import TreasuryRates, Option, OptionChain, OptionType


class TestVelocitySignalMomentumStrategy:
    """Test cases for VelocitySignalMomentumStrategy"""

    def setup_method(self):
        """Set up test data"""
        self.strategy = VelocitySignalMomentumStrategy()
        
        # Create sample market data
        dates = pd.date_range('2021-01-01', '2021-01-30', freq='D')
        self.market_data = pd.DataFrame({
            'Close': [100 + i * 0.5 for i in range(len(dates))],  # Upward trend
            'Volume': [1000000] * len(dates)
        }, index=dates)
        
        # Create sample treasury rates
        treasury_dates = pd.date_range('2021-01-01', '2021-01-30', freq='D')
        treasury_data = pd.DataFrame({
            'IRX_1Y': [0.001] * len(treasury_dates),  # 0.1% risk-free rate
            'TNX_10Y': [0.015] * len(treasury_dates)  # 1.5% 10-year rate
        }, index=treasury_dates)
        self.treasury_rates = TreasuryRates(treasury_data)
        
        # Create sample options
        self.atm_option = Option(
            ticker="SPY",
            symbol="SPY",
            expiration="2021-02-19",
            strike=100.0,
            option_type=OptionType.PUT,
            last_price=2.50,
            bid=2.45,
            ask=2.55,
            volume=100,
            open_interest=1000
        )
        
        self.otm_option = Option(
            ticker="SPY",
            symbol="SPY",
            expiration="2021-02-19",
            strike=95.0,
            option_type=OptionType.PUT,
            last_price=1.00,
            bid=0.95,
            ask=1.05,
            volume=50,
            open_interest=500
        )
        
        # Create sample position
        self.position = Position(
            symbol="SPY",
            expiration_date=datetime(2021, 2, 19),
            strategy_type=StrategyType.PUT_CREDIT_SPREAD,
            strike_price=100.0,
            entry_date=datetime(2021, 1, 15),
            entry_price=1.50,  # Net credit received
            spread_options=[self.atm_option, self.otm_option]
        )
        self.position.set_quantity(1)

    def test_calculate_sharpe_ratio_with_valid_data(self):
        """Test Sharpe ratio calculation with valid position and market data"""
        # Set up strategy data
        self.strategy.set_data(self.market_data, {}, self.treasury_rates)
        
        # Mock option chain with current prices
        mock_option_chain = Mock()
        mock_atm_data = Mock()
        mock_atm_data.last_price = 2.00  # Current ATM price
        mock_otm_data = Mock()
        mock_otm_data.last_price = 0.75  # Current OTM price
        
        mock_option_chain.get_option_data_for_option.side_effect = lambda opt: {
            self.atm_option: mock_atm_data,
            self.otm_option: mock_otm_data
        }.get(opt)
        
        current_date = datetime(2021, 1, 20)
        sharpe_ratio = self.strategy._calculate_sharpe_ratio(
            self.position, current_date, mock_option_chain
        )
        
        # Should return a valid Sharpe ratio (not 0.0)
        assert isinstance(sharpe_ratio, float)
        assert sharpe_ratio != 0.0

    def test_calculate_sharpe_ratio_without_treasury_data(self):
        """Test Sharpe ratio calculation when treasury data is not available"""
        # Set up strategy data without treasury rates
        self.strategy.set_data(self.market_data, {}, None)
        
        mock_option_chain = Mock()
        mock_atm_data = Mock()
        mock_atm_data.last_price = 2.00
        mock_otm_data = Mock()
        mock_otm_data.last_price = 0.75
        
        mock_option_chain.get_option_data_for_option.side_effect = lambda opt: {
            self.atm_option: mock_atm_data,
            self.otm_option: mock_otm_data
        }.get(opt)
        
        current_date = datetime(2021, 1, 20)
        sharpe_ratio = self.strategy._calculate_sharpe_ratio(
            self.position, current_date, mock_option_chain
        )
        
        # Should still calculate Sharpe ratio (risk-free rate defaults to 0.0)
        assert isinstance(sharpe_ratio, float)

    def test_calculate_sharpe_ratio_without_option_chain(self):
        """Test Sharpe ratio calculation when option chain is not provided"""
        self.strategy.set_data(self.market_data, {}, self.treasury_rates)
        
        current_date = datetime(2021, 1, 20)
        sharpe_ratio = self.strategy._calculate_sharpe_ratio(
            self.position, current_date, None
        )
        
        # Should return 2.0 when option chain is not available (realistic fallback)
        assert sharpe_ratio == 2.0

    def test_calculate_sharpe_ratio_without_spread_options(self):
        """Test Sharpe ratio calculation when position has no spread options"""
        position_without_options = Position(
            symbol="SPY",
            expiration_date=datetime(2021, 2, 19),
            strategy_type=StrategyType.PUT_CREDIT_SPREAD,
            strike_price=100.0,
            entry_date=datetime(2021, 1, 15),
            entry_price=1.50,
            spread_options=[]  # No spread options
        )
        position_without_options.set_quantity(1)
        
        self.strategy.set_data(self.market_data, {}, self.treasury_rates)
        
        mock_option_chain = Mock()
        current_date = datetime(2021, 1, 20)
        sharpe_ratio = self.strategy._calculate_sharpe_ratio(
            position_without_options, current_date, mock_option_chain
        )
        
        # Should return 2.0 when no spread options (realistic fallback)
        assert sharpe_ratio == 2.0

    def test_calculate_sharpe_ratio_with_missing_option_data(self):
        """Test Sharpe ratio calculation when option data is missing"""
        self.strategy.set_data(self.market_data, {}, self.treasury_rates)
        
        # Mock option chain that returns None for option data
        mock_option_chain = Mock()
        mock_option_chain.get_option_data_for_option.return_value = None
        
        current_date = datetime(2021, 1, 20)
        sharpe_ratio = self.strategy._calculate_sharpe_ratio(
            self.position, current_date, mock_option_chain
        )
        
        # Should return 2.0 when option data is missing (realistic fallback)
        assert sharpe_ratio == 2.0

    def test_calculate_sharpe_ratio_with_zero_max_risk(self):
        """Test Sharpe ratio calculation when max risk is zero"""
        # Create position with zero max risk (edge case)
        position_zero_risk = Position(
            symbol="SPY",
            expiration_date=datetime(2021, 2, 19),
            strategy_type=StrategyType.PUT_CREDIT_SPREAD,
            strike_price=100.0,
            entry_date=datetime(2021, 1, 15),
            entry_price=5.0,  # Credit equals spread width
            spread_options=[self.atm_option, self.otm_option]
        )
        position_zero_risk.set_quantity(1)
        
        self.strategy.set_data(self.market_data, {}, self.treasury_rates)
        
        mock_option_chain = Mock()
        mock_atm_data = Mock()
        mock_atm_data.last_price = 2.00
        mock_otm_data = Mock()
        mock_otm_data.last_price = 0.75
        
        mock_option_chain.get_option_data_for_option.side_effect = lambda opt: {
            self.atm_option: mock_atm_data,
            self.otm_option: mock_otm_data
        }.get(opt)
        
        current_date = datetime(2021, 1, 20)
        sharpe_ratio = self.strategy._calculate_sharpe_ratio(
            position_zero_risk, current_date, mock_option_chain
        )
        
        # Should handle zero max risk gracefully
        assert isinstance(sharpe_ratio, float)

    def test_calculate_sharpe_ratio_without_market_data(self):
        """Test Sharpe ratio calculation when market data is not available"""
        self.strategy.set_data(None, {}, self.treasury_rates)
        
        mock_option_chain = Mock()
        mock_atm_data = Mock()
        mock_atm_data.last_price = 2.00
        mock_otm_data = Mock()
        mock_otm_data.last_price = 0.75
        
        mock_option_chain.get_option_data_for_option.side_effect = lambda opt: {
            self.atm_option: mock_atm_data,
            self.otm_option: mock_otm_data
        }.get(opt)
        
        current_date = datetime(2021, 1, 20)
        sharpe_ratio = self.strategy._calculate_sharpe_ratio(
            self.position, current_date, mock_option_chain
        )
        
        # Should return 2.0 when market data is not available (realistic fallback)
        assert sharpe_ratio == 2.0

    def test_calculate_sharpe_ratio_with_empty_market_data(self):
        """Test Sharpe ratio calculation when market data is empty"""
        empty_market_data = pd.DataFrame(columns=['Close', 'Volume'])
        self.strategy.set_data(empty_market_data, {}, self.treasury_rates)
        
        mock_option_chain = Mock()
        mock_atm_data = Mock()
        mock_atm_data.last_price = 2.00
        mock_otm_data = Mock()
        mock_otm_data.last_price = 0.75
        
        mock_option_chain.get_option_data_for_option.side_effect = lambda opt: {
            self.atm_option: mock_atm_data,
            self.otm_option: mock_otm_data
        }.get(opt)
        
        current_date = datetime(2021, 1, 20)
        sharpe_ratio = self.strategy._calculate_sharpe_ratio(
            self.position, current_date, mock_option_chain
        )
        
        # Should return 2.0 when market data is empty (realistic fallback)
        assert sharpe_ratio == 2.0

    def test_calculate_sharpe_ratio_positive_return(self):
        """Test Sharpe ratio calculation with positive return scenario"""
        self.strategy.set_data(self.market_data, {}, self.treasury_rates)
        
        # Mock option chain with prices that result in positive return
        mock_option_chain = Mock()
        mock_atm_data = Mock()
        mock_atm_data.last_price = 1.00  # Lower than entry, positive return
        mock_otm_data = Mock()
        mock_otm_data.last_price = 0.25  # Lower than entry
        
        mock_option_chain.get_option_data_for_option.side_effect = lambda opt: {
            self.atm_option: mock_atm_data,
            self.otm_option: mock_otm_data
        }.get(opt)
        
        current_date = datetime(2021, 1, 20)
        sharpe_ratio = self.strategy._calculate_sharpe_ratio(
            self.position, current_date, mock_option_chain
        )
        
        # Should calculate a valid Sharpe ratio for positive return
        assert isinstance(sharpe_ratio, float)

    def test_calculate_sharpe_ratio_negative_return(self):
        """Test Sharpe ratio calculation with negative return scenario"""
        self.strategy.set_data(self.market_data, {}, self.treasury_rates)
        
        # Mock option chain with prices that result in negative return
        mock_option_chain = Mock()
        mock_atm_data = Mock()
        mock_atm_data.last_price = 4.00  # Higher than entry, negative return
        mock_otm_data = Mock()
        mock_otm_data.last_price = 2.00  # Higher than entry
        
        mock_option_chain.get_option_data_for_option.side_effect = lambda opt: {
            self.atm_option: mock_atm_data,
            self.otm_option: mock_otm_data
        }.get(opt)
        
        current_date = datetime(2021, 1, 20)
        sharpe_ratio = self.strategy._calculate_sharpe_ratio(
            self.position, current_date, mock_option_chain
        )
        
        # Should calculate a valid Sharpe ratio for negative return
        assert isinstance(sharpe_ratio, float)

    def test_get_risk_free_rate_with_treasury_data(self):
        """Test getting risk-free rate when treasury data is available"""
        self.strategy.set_data(self.market_data, {}, self.treasury_rates)
        
        current_date = datetime(2021, 1, 15)
        risk_free_rate = self.strategy._get_risk_free_rate(current_date)
        
        assert isinstance(risk_free_rate, float)
        assert risk_free_rate == 0.001  # Should match treasury data

    def test_get_risk_free_rate_without_treasury_data(self):
        """Test getting risk-free rate when treasury data is not available"""
        self.strategy.set_data(self.market_data, {}, None)
        
        current_date = datetime(2021, 1, 15)
        risk_free_rate = self.strategy._get_risk_free_rate(current_date)
        
        assert risk_free_rate == 0.0  # Should default to 0.0
