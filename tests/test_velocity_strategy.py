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
        # Create a mock options handler for testing
        mock_options_handler = Mock()
        self.strategy = VelocitySignalMomentumStrategy(options_handler=mock_options_handler)
        
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
            self.position, current_date
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
            self.position, current_date
        )
        
        # Should still calculate Sharpe ratio (risk-free rate defaults to 0.0)
        assert isinstance(sharpe_ratio, float)

    def test_calculate_sharpe_ratio_without_option_chain(self):
        """Test Sharpe ratio calculation when option chain is not provided"""
        self.strategy.set_data(self.market_data, {}, self.treasury_rates)
        
        current_date = datetime(2021, 1, 20)
        sharpe_ratio = self.strategy._calculate_sharpe_ratio(
            self.position, current_date
        )
        
        # Should calculate a valid Sharpe ratio using max profit (no longer needs option chain)
        assert isinstance(sharpe_ratio, float)
        assert sharpe_ratio > 0

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
        with pytest.raises(ValueError, match="Unable to calculate Sharpe ratio for position"):
            self.strategy._calculate_sharpe_ratio(
                position_without_options, current_date
            )

    def test_calculate_sharpe_ratio_with_missing_option_data(self):
        """Test Sharpe ratio calculation when option data is missing"""
        self.strategy.set_data(self.market_data, {}, self.treasury_rates)
        
        # Mock option chain that returns None for option data
        mock_option_chain = Mock()
        mock_option_chain.get_option_data_for_option.return_value = None
        
        current_date = datetime(2021, 1, 20)
        sharpe_ratio = self.strategy._calculate_sharpe_ratio(
            self.position, current_date
        )
        
        # Should calculate a valid Sharpe ratio using max profit (no longer needs option data)
        assert isinstance(sharpe_ratio, float)
        assert sharpe_ratio > 0

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
            position_zero_risk, current_date
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
        with pytest.raises(ValueError, match="Unable to calculate Sharpe ratio for position"):
            self.strategy._calculate_sharpe_ratio(
                self.position, current_date
            )

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
        with pytest.raises(ValueError, match="Unable to calculate Sharpe ratio for position"):
            self.strategy._calculate_sharpe_ratio(
                self.position, current_date
            )

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
            self.position, current_date
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
            self.position, current_date
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
        """Test getting risk-free rate when treasury data is not available."""
        mock_options_handler = Mock()
        strategy = VelocitySignalMomentumStrategy(options_handler=mock_options_handler)
        strategy.set_data(pd.DataFrame({'Close': [100, 101, 102]}), {})
        
        rate = strategy._get_risk_free_rate(datetime.now())
        assert rate == 2.0  # Default fallback when no treasury data

    def test_determine_expiration_date_with_valid_data(self):
        """Test determining expiration date with valid options data."""
        mock_options_handler = Mock()
        # Configure mock to return empty list for contracts
        mock_options_handler._fetch_filtered_option_contracts.return_value = []
        strategy = VelocitySignalMomentumStrategy(options_handler=mock_options_handler)
        
        # Create mock data
        data = pd.DataFrame({
            'Close': [100, 101, 102]
        }, index=pd.date_range('2024-01-01', periods=3))
        
        # Create mock options data with multiple expiration dates
        atm_put_1 = Option(
            ticker='SPY',
            symbol='SPY240115P100',
            strike=100.0,
            expiration='2024-01-15',
            option_type=OptionType.PUT,
            last_price=2.50,
            volume=100
        )
        
        otm_put_1 = Option(
            ticker='SPY',
            symbol='SPY240115P90',
            strike=90.0,
            expiration='2024-01-15',
            option_type=OptionType.PUT,
            last_price=0.50,
            volume=100
        )
        
        atm_put_2 = Option(
            ticker='SPY',
            symbol='SPY240130P100',
            strike=100.0,
            expiration='2024-01-30',
            option_type=OptionType.PUT,
            last_price=3.00,
            volume=100
        )
        
        otm_put_2 = Option(
            ticker='SPY',
            symbol='SPY240130P90',
            strike=90.0,
            expiration='2024-01-30',
            option_type=OptionType.PUT,
            last_price=0.75,
            volume=100
        )
        
        option_chain = OptionChain(
            calls=[],
            puts=[atm_put_1, otm_put_1, atm_put_2, otm_put_2]
        )
        
        options_data = {'2024-01-01': option_chain}
        
        # Create mock treasury data
        treasury_data = TreasuryRates(pd.DataFrame({
            'IRX_1Y': [0.05, 0.06, 0.07],
            'TNX_10Y': [0.04, 0.05, 0.06]
        }, index=pd.date_range('2024-01-01', periods=3)))
        
        strategy.set_data(data, options_data, treasury_data)
        
        # Test expiration date determination
        expiration_date = strategy._determine_expiration_date(datetime(2024, 1, 1))
        
        # Should return default 30-day expiration when options handler returns empty list
        expected_date = datetime(2024, 1, 1) + pd.Timedelta(days=30)
        assert expiration_date == expected_date

    def test_determine_expiration_date_without_options_data(self):
        """Test determining expiration date when no options data is available."""
        mock_options_handler = Mock()
        strategy = VelocitySignalMomentumStrategy(options_handler=mock_options_handler)
        strategy.set_data(pd.DataFrame({'Close': [100, 101, 102]}), {})
        
        with pytest.raises(ValueError, match="Options data is required for expiration date determination but is not available"):
            strategy._determine_expiration_date(datetime(2024, 1, 1))

    def test_determine_expiration_date_without_market_data(self):
        """Test determining expiration date when no market data is available."""
        mock_options_handler = Mock()
        strategy = VelocitySignalMomentumStrategy(options_handler=mock_options_handler)
        
        # Create mock options data but with None market data
        atm_put = Option(
            ticker='SPY',
            symbol='SPY240115P100',
            strike=100.0,
            expiration='2024-01-15',
            option_type=OptionType.PUT,
            last_price=2.50,
            volume=100
        )
        
        option_chain = OptionChain(
            calls=[],
            puts=[atm_put]
        )
        
        options_data = {'2024-01-01': option_chain}
        
        strategy.set_data(None, options_data)
        
        with pytest.raises(ValueError, match="Market data is required for expiration date determination but is not available"):
            strategy._determine_expiration_date(datetime(2024, 1, 1))

    def test_determine_expiration_date_with_empty_market_data(self):
        """Test determining expiration date when market data is empty."""
        mock_options_handler = Mock()
        strategy = VelocitySignalMomentumStrategy(options_handler=mock_options_handler)
        
        # Create mock options data but with empty market data
        atm_put = Option(
            ticker='SPY',
            symbol='SPY240115P100',
            strike=100.0,
            expiration='2024-01-15',
            option_type=OptionType.PUT,
            last_price=2.50,
            volume=100
        )
        
        option_chain = OptionChain(
            calls=[],
            puts=[atm_put]
        )
        
        options_data = {'2024-01-01': option_chain}
        
        strategy.set_data(pd.DataFrame(), options_data)
        
        with pytest.raises(ValueError, match="Market data is required for expiration date determination but is not available"):
            strategy._determine_expiration_date(datetime(2024, 1, 1))

    def test_determine_expiration_date_without_treasury_data(self):
        """Test determining expiration date when no treasury data is available."""
        mock_options_handler = Mock()
        strategy = VelocitySignalMomentumStrategy(options_handler=mock_options_handler)
        
        # Create mock data
        data = pd.DataFrame({
            'Close': [100, 101, 102]
        }, index=pd.date_range('2024-01-01', periods=3))
        
        # Create mock options data
        atm_put = Option(
            ticker='SPY',
            symbol='SPY240115P100',
            strike=100.0,
            expiration='2024-01-15',
            option_type=OptionType.PUT,
            last_price=2.50,
            volume=100
        )
        
        option_chain = OptionChain(
            calls=[],
            puts=[atm_put]
        )
        
        options_data = {'2024-01-01': option_chain}
        
        strategy.set_data(data, options_data)  # No treasury data
        
        with pytest.raises(ValueError, match="Treasury data is required for Sharpe ratio calculation but is not available"):
            strategy._determine_expiration_date(datetime(2024, 1, 1))

    def test_create_test_put_credit_spread_success(self):
        """Test creating a test put credit spread successfully."""
        mock_options_handler = Mock()
        strategy = VelocitySignalMomentumStrategy(options_handler=mock_options_handler)
        
        # Create mock data
        data = pd.DataFrame({
            'Close': [100, 101, 102]
        }, index=pd.date_range('2024-01-01', periods=3))
        
        # Create mock options data
        atm_put = Option(
            ticker='SPY',
            symbol='SPY240115P100',
            strike=100.0,
            expiration='2024-01-15',
            option_type=OptionType.PUT,
            last_price=2.50,
            volume=100
        )
        
        otm_put = Option(
            ticker='SPY',
            symbol='SPY240115P90',
            strike=90.0,
            expiration='2024-01-15',
            option_type=OptionType.PUT,
            last_price=0.50,
            volume=100
        )
        
        option_chain = OptionChain(
            calls=[],
            puts=[atm_put, otm_put]
        )
        
        options_data = {'2024-01-01': option_chain}
        
        # Create mock treasury data
        treasury_data = TreasuryRates(pd.DataFrame({
            'IRX_1Y': [0.05, 0.06, 0.07],
            'TNX_10Y': [0.04, 0.05, 0.06]
        }, index=pd.date_range('2024-01-01', periods=3)))
        
        strategy.set_data(data, options_data, treasury_data)
        
        # Test creating test put credit spread
        position = strategy._create_test_put_credit_spread(
            datetime(2024, 1, 1), 100.0, '2024-01-15'
        )
        
        assert position is not None
        assert position.strategy_type == StrategyType.PUT_CREDIT_SPREAD
        assert position.strike_price == 100.0
        assert position.entry_price == 2.0  # 2.50 - 0.50
        assert len(position.spread_options) == 2

    def test_create_test_put_credit_spread_no_credit(self):
        """Test creating a test put credit spread when no credit is received."""
        mock_options_handler = Mock()
        strategy = VelocitySignalMomentumStrategy(options_handler=mock_options_handler)
        
        # Create mock data
        data = pd.DataFrame({
            'Close': [100, 101, 102]
        }, index=pd.date_range('2024-01-01', periods=3))
        
        # Create mock options data where ATM put is cheaper than OTM put (no credit)
        atm_put = Option(
            ticker='SPY',
            symbol='SPY240115P100',
            strike=100.0,
            expiration='2024-01-15',
            option_type=OptionType.PUT,
            last_price=0.50,
            volume=100
        )
        
        otm_put = Option(
            ticker='SPY',
            symbol='SPY240115P90',
            strike=90.0,
            expiration='2024-01-15',
            option_type=OptionType.PUT,
            last_price=2.50,
            volume=100
        )
        
        option_chain = OptionChain(
            calls=[],
            puts=[atm_put, otm_put]
        )
        
        options_data = {'2024-01-01': option_chain}
        
        # Create mock treasury data
        treasury_data = TreasuryRates(pd.DataFrame({
            'IRX_1Y': [0.05, 0.06, 0.07],
            'TNX_10Y': [0.04, 0.05, 0.06]
        }, index=pd.date_range('2024-01-01', periods=3)))
        
        strategy.set_data(data, options_data, treasury_data)
        
        # Test creating test put credit spread
        position = strategy._create_test_put_credit_spread(
            datetime(2024, 1, 1), 100.0, '2024-01-15'
        )
        
        # Should return None because no credit is received
        assert position is None
