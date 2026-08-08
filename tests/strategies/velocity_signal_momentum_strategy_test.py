import pytest
import pandas as pd
from datetime import datetime
from unittest.mock import Mock
from algo_trading_engine.strategies.velocity_signal_momentum_strategy import VelocitySignalMomentumStrategy
from algo_trading_engine.vo import Position, create_position
from algo_trading_engine.common.models import StrategyType
from algo_trading_engine.common.models import TreasuryRates, Option, OptionChain, OptionType

class TestVelocitySignalMomentumStrategy:
    """Test cases for VelocitySignalMomentumStrategy"""

    def setup_method(self):
        """Set up test data"""
        mock_options_handler = Mock()
        get_contract_list_for_date = mock_options_handler.get_contract_list_for_date
        get_option_bar = mock_options_handler.get_option_bar
        get_options_chain = mock_options_handler.get_options_chain
        self.strategy = VelocitySignalMomentumStrategy(get_contract_list_for_date=get_contract_list_for_date, get_option_bar=get_option_bar, get_options_chain=get_options_chain)
        dates = pd.date_range('2021-01-01', '2021-01-30', freq='D')
        self.market_data = pd.DataFrame({'Close': [100 + i * 0.5 for i in range(len(dates))], 'Volume': [1000000] * len(dates)}, index=dates)
        treasury_dates = pd.date_range('2021-01-01', '2021-01-30', freq='D')
        treasury_data = pd.DataFrame({'IRX_1Y': [0.001] * len(treasury_dates), 'TNX_10Y': [0.015] * len(treasury_dates)}, index=treasury_dates)
        self.treasury_rates = TreasuryRates(treasury_data)
        self.atm_option = Option(ticker='SPY', symbol='SPY', expiration='2021-02-19', strike=100.0, option_type=OptionType.PUT, last_price=2.5, bid=2.45, ask=2.55, volume=100, open_interest=1000)
        self.otm_option = Option(ticker='SPY', symbol='SPY', expiration='2021-02-19', strike=95.0, option_type=OptionType.PUT, last_price=1.0, bid=0.95, ask=1.05, volume=50, open_interest=500)
        self.position = create_position(symbol='SPY', expiration_date=datetime(2021, 2, 19), strategy_type=StrategyType.PUT_CREDIT_SPREAD, strike_price=100.0, entry_date=datetime(2021, 1, 15), entry_price=1.5, spread_options=[self.atm_option, self.otm_option])
        self.position.set_quantity(1)

    def test_profit_target_and_stop_loss_initialization(self):
        """Test that profit_target and stop_loss are properly initialized"""
        mock_options_handler = Mock()
        get_contract_list_for_date = mock_options_handler.get_contract_list_for_date
        get_option_bar = mock_options_handler.get_option_bar
        get_options_chain = mock_options_handler.get_options_chain
        strategy_with_params = VelocitySignalMomentumStrategy(get_contract_list_for_date=get_contract_list_for_date, get_option_bar=get_option_bar, get_options_chain=get_options_chain, profit_target=0.25, stop_loss=0.6)
        assert strategy_with_params.profit_target == 0.25
        assert strategy_with_params.stop_loss == 0.6
        strategy_without_params = VelocitySignalMomentumStrategy(get_contract_list_for_date=get_contract_list_for_date, get_option_bar=get_option_bar, get_options_chain=get_options_chain)
        assert strategy_without_params.profit_target is None
        assert strategy_without_params.stop_loss is None

    def test_should_close_due_to_profit_target(self):
        """Test that profit target check works correctly"""
        mock_options_handler = Mock()
        get_contract_list_for_date = mock_options_handler.get_contract_list_for_date
        get_option_bar = mock_options_handler.get_option_bar
        get_options_chain = mock_options_handler.get_options_chain
        strategy = VelocitySignalMomentumStrategy(get_contract_list_for_date=get_contract_list_for_date, get_option_bar=get_option_bar, get_options_chain=get_options_chain, profit_target=0.2)
        position = Mock()
        position.profit_target_hit = Mock(side_effect=lambda target, price: price is not None and True)
        assert strategy._profit_target_hit(position, 1.5) == True
        position.profit_target_hit = Mock(side_effect=lambda target, price: price is not None and True)
        assert strategy._profit_target_hit(position, None) == False
        strategy_no_target = VelocitySignalMomentumStrategy(get_contract_list_for_date=get_contract_list_for_date, get_option_bar=get_option_bar, get_options_chain=get_options_chain, profit_target=None)
        assert strategy_no_target._profit_target_hit(position, 1.5) == False

    def test_should_close_due_to_stop_loss(self):
        """Test that stop loss check works correctly"""
        mock_options_handler = Mock()
        get_contract_list_for_date = mock_options_handler.get_contract_list_for_date
        get_option_bar = mock_options_handler.get_option_bar
        get_options_chain = mock_options_handler.get_options_chain
        strategy = VelocitySignalMomentumStrategy(get_contract_list_for_date=get_contract_list_for_date, get_option_bar=get_option_bar, get_options_chain=get_options_chain, stop_loss=0.6)
        position = Mock()
        position.stop_loss_hit = Mock(side_effect=lambda stop, price: price is not None and True)
        assert strategy._stop_loss_hit(position, 1.5) == True
        position.stop_loss_hit = Mock(side_effect=lambda stop, price: price is not None and True)
        assert strategy._stop_loss_hit(position, None) == False
        strategy_no_stop = VelocitySignalMomentumStrategy(get_contract_list_for_date=get_contract_list_for_date, get_option_bar=get_option_bar, get_options_chain=get_options_chain, stop_loss=None)
        assert strategy_no_stop._stop_loss_hit(position, 1.5) == False

    def test_select_week_expiration_prefers_5_to_10_days(self):
        mock_options_handler = Mock()
        get_contract_list_for_date = mock_options_handler.get_contract_list_for_date
        get_option_bar = mock_options_handler.get_option_bar
        get_options_chain = mock_options_handler.get_options_chain
        strategy = VelocitySignalMomentumStrategy(get_contract_list_for_date=get_contract_list_for_date, get_option_bar=get_option_bar, get_options_chain=get_options_chain)
        from algo_trading_engine.dto import OptionContractDTO
        from algo_trading_engine.vo import StrikePrice, ExpirationDate
        from algo_trading_engine.common.models import OptionType as CommonOptionType
        from decimal import Decimal
        contracts = [OptionContractDTO(ticker='O:SPY240103P100', underlying_ticker='SPY', contract_type=CommonOptionType.PUT, strike_price=StrikePrice(Decimal('100.0')), expiration_date=ExpirationDate(datetime(2024, 1, 3).date()), exercise_style='american', shares_per_contract=100), OptionContractDTO(ticker='O:SPY240108P100', underlying_ticker='SPY', contract_type=CommonOptionType.PUT, strike_price=StrikePrice(Decimal('100.0')), expiration_date=ExpirationDate(datetime(2024, 1, 8).date()), exercise_style='american', shares_per_contract=100), OptionContractDTO(ticker='O:SPY240120P100', underlying_ticker='SPY', contract_type=CommonOptionType.PUT, strike_price=StrikePrice(Decimal('100.0')), expiration_date=ExpirationDate(datetime(2024, 1, 20).date()), exercise_style='american', shares_per_contract=100)]
        strategy.get_contract_list_for_date = Mock(return_value=contracts)
        date = datetime(2024, 1, 1)
        picked = strategy._select_week_expiration(date)
        assert picked == '2024-01-08'

    def test_get_current_underlying_price(self):
        mock_options_handler = Mock()
        get_contract_list_for_date = mock_options_handler.get_contract_list_for_date
        get_option_bar = mock_options_handler.get_option_bar
        get_options_chain = mock_options_handler.get_options_chain
        strategy = VelocitySignalMomentumStrategy(get_contract_list_for_date=get_contract_list_for_date, get_option_bar=get_option_bar, get_options_chain=get_options_chain)
        dates = pd.date_range('2024-01-01', periods=3)
        data = pd.DataFrame({'Close': [100.0, 101.0, 102.0]}, index=dates)
        data.index.name = 'SPY'
        strategy.set_data(data)

        def mock_get_price(date, _symbol):
            if date == datetime(2024, 1, 2):
                return 101.0
            else:
                raise KeyError(f'Date {date} not found')
        strategy.get_current_underlying_price = mock_get_price
        assert strategy.get_current_underlying_price(datetime(2024, 1, 2), 'SPY') == 101.0
        import pytest
        with pytest.raises(KeyError):
            strategy.get_current_underlying_price(datetime(2023, 12, 31), 'SPY')

    def test_sanitize_exit_price(self):
        mock_options_handler = Mock()
        get_contract_list_for_date = mock_options_handler.get_contract_list_for_date
        get_option_bar = mock_options_handler.get_option_bar
        get_options_chain = mock_options_handler.get_options_chain
        strategy = VelocitySignalMomentumStrategy(get_contract_list_for_date=get_contract_list_for_date, get_option_bar=get_option_bar, get_options_chain=get_options_chain)
        assert strategy._sanitize_exit_price(-1.234) == 0.0
        assert strategy._sanitize_exit_price(0.004) == 0.0
        assert strategy._sanitize_exit_price(1.235) == 1.24
        assert strategy._sanitize_exit_price(None) is None

    def test_should_close_predicates(self):
        mock_options_handler = Mock()
        get_contract_list_for_date = mock_options_handler.get_contract_list_for_date
        get_option_bar = mock_options_handler.get_option_bar
        get_options_chain = mock_options_handler.get_options_chain
        strategy = VelocitySignalMomentumStrategy(get_contract_list_for_date=get_contract_list_for_date, get_option_bar=get_option_bar, get_options_chain=get_options_chain)
        pos = self.position
        assert strategy._should_close_due_to_holding(pos, datetime(2021, 1, 20), holding_period=5) is True
        assert strategy._stop_loss_hit(pos, exit_price=0.5) is False

    def test_compute_exit_price_with_chain_and_missing_contracts(self):
        """Test that strategy uses the injected compute_exit_price callable"""
        mock_options_handler = Mock()
        get_contract_list_for_date = mock_options_handler.get_contract_list_for_date
        get_option_bar = mock_options_handler.get_option_bar
        get_options_chain = mock_options_handler.get_options_chain
        strategy = VelocitySignalMomentumStrategy(get_contract_list_for_date=get_contract_list_for_date, get_option_bar=get_option_bar, get_options_chain=get_options_chain)
        date = datetime(2024, 1, 1)
        atm = Option(ticker='O:SPY240115P100', symbol='A', expiration='2024-01-15', strike=100.0, option_type=OptionType.PUT, last_price=2.0)
        otm = Option(ticker='O:SPY240115P90', symbol='B', expiration='2024-01-15', strike=90.0, option_type=OptionType.PUT, last_price=1.0)
        pos = create_position(symbol='SPY', expiration_date=datetime(2024, 1, 15), strategy_type=StrategyType.PUT_CREDIT_SPREAD, strike_price=100.0, entry_date=date, entry_price=1.0, spread_options=[atm, otm])
        pos.set_quantity(1)
        strategy.compute_exit_price = Mock(return_value=0.8)
        exit_price = strategy.compute_exit_price(pos, date)
        assert exit_price == 0.8
        strategy.compute_exit_price.assert_called_once_with(pos, date)

    def test_get_risk_free_rate_with_treasury_data(self):
        """Test getting risk-free rate when treasury data is available"""
        self.strategy.set_data(self.market_data, self.treasury_rates)
        current_date = datetime(2021, 1, 15)
        risk_free_rate = self.strategy._get_risk_free_rate(current_date)
        assert isinstance(risk_free_rate, float)
        assert risk_free_rate == 0.001

    def test_get_risk_free_rate_without_treasury_data(self):
        """Test getting risk-free rate when treasury data is not available."""
        mock_options_handler = Mock()
        get_contract_list_for_date = mock_options_handler.get_contract_list_for_date
        get_option_bar = mock_options_handler.get_option_bar
        get_options_chain = mock_options_handler.get_options_chain
        strategy = VelocitySignalMomentumStrategy(get_contract_list_for_date=get_contract_list_for_date, get_option_bar=get_option_bar, get_options_chain=get_options_chain)
        strategy.set_data(pd.DataFrame({'Close': [100, 101, 102]}))
        rate = strategy._get_risk_free_rate(datetime.now())
        assert rate == 2.0

    def test_create_test_put_credit_spread_success(self):
        """Test creating a test put credit spread successfully."""
        mock_options_handler = Mock()
        get_contract_list_for_date = mock_options_handler.get_contract_list_for_date
        get_option_bar = mock_options_handler.get_option_bar
        get_options_chain = mock_options_handler.get_options_chain
        strategy = VelocitySignalMomentumStrategy(get_contract_list_for_date=get_contract_list_for_date, get_option_bar=get_option_bar, get_options_chain=get_options_chain)
        from algo_trading_engine.dto import OptionContractDTO
        from algo_trading_engine.vo import StrikePrice, ExpirationDate
        from algo_trading_engine.common.models import OptionType
        from datetime import date as date_type
        from decimal import Decimal
        atm_contract = OptionContractDTO(ticker='O:SPY240115P100', underlying_ticker='SPY', contract_type=OptionType.PUT, strike_price=StrikePrice(Decimal('100.0')), expiration_date=ExpirationDate(date_type(2024, 1, 15)), exercise_style='american', shares_per_contract=100)
        otm_contract = OptionContractDTO(ticker='O:SPY240115P94', underlying_ticker='SPY', contract_type=OptionType.PUT, strike_price=StrikePrice(Decimal('94.0')), expiration_date=ExpirationDate(date_type(2024, 1, 15)), exercise_style='american', shares_per_contract=100)

        def mock_get_contract_list_for_date(date, strike_range=None, expiration_range=None):
            _ = date, strike_range, expiration_range
            return [atm_contract, otm_contract]
        strategy.get_contract_list_for_date = Mock(side_effect=mock_get_contract_list_for_date)
        from algo_trading_engine.dto import OptionBarDTO
        from decimal import Decimal
        atm_bar = OptionBarDTO(ticker='O:SPY240115P100', timestamp=datetime(2024, 1, 1), open_price=Decimal('2.50'), high_price=Decimal('2.60'), low_price=Decimal('2.40'), close_price=Decimal('2.50'), volume=1000, volume_weighted_avg_price=Decimal('2.50'), number_of_transactions=100)
        otm_bar = OptionBarDTO(ticker='O:SPY240115P94', timestamp=datetime(2024, 1, 1), open_price=Decimal('0.50'), high_price=Decimal('0.60'), low_price=Decimal('0.40'), close_price=Decimal('0.50'), volume=800, volume_weighted_avg_price=Decimal('0.50'), number_of_transactions=80)
        strategy.get_option_bar = Mock(side_effect=lambda opt, date, multiplier=1, timespan=None: atm_bar if '100' in opt.ticker else otm_bar)
        data = pd.DataFrame({'Close': [100, 101, 102]}, index=pd.date_range('2024-01-01', periods=3))
        treasury_data = TreasuryRates(pd.DataFrame({'IRX_1Y': [0.05, 0.06, 0.07], 'TNX_10Y': [0.04, 0.05, 0.06]}, index=pd.date_range('2024-01-01', periods=3)))
        strategy.set_data(data, treasury_data)
        position = strategy._create_put_credit_spread(datetime(2024, 1, 1), 100.0, '2024-01-15')
        assert position is not None
        assert position.strategy_type == StrategyType.PUT_CREDIT_SPREAD
        assert position.strike_price == 100.0
        assert position.entry_price == 2.0
        assert len(position.spread_options) == 2

    def test_create_test_put_credit_spread_no_credit(self):
        """Test creating a test put credit spread when no credit is received."""
        mock_options_handler = Mock()
        get_contract_list_for_date = mock_options_handler.get_contract_list_for_date
        get_option_bar = mock_options_handler.get_option_bar
        get_options_chain = mock_options_handler.get_options_chain
        strategy = VelocitySignalMomentumStrategy(get_contract_list_for_date=get_contract_list_for_date, get_option_bar=get_option_bar, get_options_chain=get_options_chain)
        data = pd.DataFrame({'Close': [100, 101, 102]}, index=pd.date_range('2024-01-01', periods=3))
        atm_put = Option(ticker='SPY', symbol='SPY240115P100', strike=100.0, expiration='2024-01-15', option_type=OptionType.PUT, last_price=0.5, volume=100)
        otm_put = Option(ticker='SPY', symbol='SPY240115P90', strike=90.0, expiration='2024-01-15', option_type=OptionType.PUT, last_price=2.5, volume=100)
        option_chain = OptionChain(calls=[], puts=[atm_put, otm_put])
        treasury_data = TreasuryRates(pd.DataFrame({'IRX_1Y': [0.05, 0.06, 0.07], 'TNX_10Y': [0.04, 0.05, 0.06]}, index=pd.date_range('2024-01-01', periods=3)))
        strategy.set_data(data, treasury_data)
        position = strategy._create_put_credit_spread(datetime(2024, 1, 1), 100.0, '2024-01-15')
        assert position is None

    def test_has_buy_signal_valid_uptrend(self):
        """Test _has_buy_signal with a valid upward trend using MA velocity."""
        mock_options_handler = Mock()
        get_contract_list_for_date = mock_options_handler.get_contract_list_for_date
        get_option_bar = mock_options_handler.get_option_bar
        get_options_chain = mock_options_handler.get_options_chain
        strategy = VelocitySignalMomentumStrategy(get_contract_list_for_date=get_contract_list_for_date, get_option_bar=get_option_bar, get_options_chain=get_options_chain)
        dates = pd.date_range('2024-01-01', '2024-03-10', freq='D')
        base_prices = [100] * 30 + [95] * 5 + [98, 102, 105, 107, 108, 109, 110] + [110] * (len(dates) - 42)
        remaining_days = len(dates) - len(base_prices)
        prices = base_prices + [110] * remaining_days
        market_data = pd.DataFrame({'Close': prices}, index=dates)
        strategy.set_data(market_data)
        result = strategy._has_buy_signal(datetime(2024, 2, 8))
        assert result is True

    def test_has_buy_signal_no_data(self):
        """Test _has_buy_signal when no data is available."""
        mock_options_handler = Mock()
        get_contract_list_for_date = mock_options_handler.get_contract_list_for_date
        get_option_bar = mock_options_handler.get_option_bar
        get_options_chain = mock_options_handler.get_options_chain
        strategy = VelocitySignalMomentumStrategy(get_contract_list_for_date=get_contract_list_for_date, get_option_bar=get_option_bar, get_options_chain=get_options_chain)
        result = strategy._has_buy_signal(datetime(2024, 1, 20))
        assert result is False

    def test_has_buy_signal_insufficient_history(self):
        """Test _has_buy_signal when there's insufficient historical data."""
        mock_options_handler = Mock()
        get_contract_list_for_date = mock_options_handler.get_contract_list_for_date
        get_option_bar = mock_options_handler.get_option_bar
        get_options_chain = mock_options_handler.get_options_chain
        strategy = VelocitySignalMomentumStrategy(get_contract_list_for_date=get_contract_list_for_date, get_option_bar=get_option_bar, get_options_chain=get_options_chain)
        dates = pd.date_range('2024-01-01', '2024-01-20', freq='D')
        market_data = pd.DataFrame({'Close': [100 + i * 0.5 for i in range(len(dates))]}, index=dates)
        strategy.set_data(market_data)
        result = strategy._has_buy_signal(datetime(2024, 1, 20))
        assert result is False

    def test_has_buy_signal_trend_too_short(self):
        """Test _has_buy_signal when trend duration is less than 3 days."""
        mock_options_handler = Mock()
        get_contract_list_for_date = mock_options_handler.get_contract_list_for_date
        get_option_bar = mock_options_handler.get_option_bar
        get_options_chain = mock_options_handler.get_options_chain
        strategy = VelocitySignalMomentumStrategy(get_contract_list_for_date=get_contract_list_for_date, get_option_bar=get_option_bar, get_options_chain=get_options_chain)
        dates = pd.date_range('2024-01-01', '2024-03-10', freq='D')
        base_prices = [100] * 65 + [95, 96, 97]
        remaining_days = len(dates) - len(base_prices)
        prices = base_prices + [97] * remaining_days
        market_data = pd.DataFrame({'Close': prices}, index=dates)
        strategy.set_data(market_data)
        result = strategy._has_buy_signal(datetime(2024, 3, 8))
        assert result is False

    def test_has_buy_signal_trend_too_long(self):
        """Test _has_buy_signal when trend duration exceeds 60 days."""
        mock_options_handler = Mock()
        get_contract_list_for_date = mock_options_handler.get_contract_list_for_date
        get_option_bar = mock_options_handler.get_option_bar
        get_options_chain = mock_options_handler.get_options_chain
        strategy = VelocitySignalMomentumStrategy(get_contract_list_for_date=get_contract_list_for_date, get_option_bar=get_option_bar, get_options_chain=get_options_chain)
        dates = pd.date_range('2024-01-01', '2024-03-10', freq='D')
        base_prices = [100] * 30 + [95] + [95 + i * 0.1 for i in range(39)]
        remaining_days = len(dates) - len(base_prices)
        prices = base_prices + [base_prices[-1]] * remaining_days
        market_data = pd.DataFrame({'Close': prices}, index=dates)
        strategy.set_data(market_data)
        result = strategy._has_buy_signal(datetime(2024, 2, 5))
        assert result is False

    def test_has_buy_signal_no_velocity_increase(self):
        """Test _has_buy_signal when there's no MA velocity increase."""
        mock_options_handler = Mock()
        get_contract_list_for_date = mock_options_handler.get_contract_list_for_date
        get_option_bar = mock_options_handler.get_option_bar
        get_options_chain = mock_options_handler.get_options_chain
        strategy = VelocitySignalMomentumStrategy(get_contract_list_for_date=get_contract_list_for_date, get_option_bar=get_option_bar, get_options_chain=get_options_chain)
        dates = pd.date_range('2024-01-01', '2024-03-10', freq='D')
        base_prices = [100] * 30 + [95] * (len(dates) - 30)
        remaining_days = len(dates) - len(base_prices)
        prices = base_prices + [95] * remaining_days
        market_data = pd.DataFrame({'Close': prices}, index=dates)
        strategy.set_data(market_data)
        result = strategy._has_buy_signal(datetime(2024, 2, 15))
        assert result is False

    def test_has_buy_signal_significant_reversal(self):
        """Test _has_buy_signal when there's a significant reversal (>2% drop)."""
        mock_options_handler = Mock()
        get_contract_list_for_date = mock_options_handler.get_contract_list_for_date
        get_option_bar = mock_options_handler.get_option_bar
        get_options_chain = mock_options_handler.get_options_chain
        strategy = VelocitySignalMomentumStrategy(get_contract_list_for_date=get_contract_list_for_date, get_option_bar=get_option_bar, get_options_chain=get_options_chain)
        dates = pd.date_range('2024-01-01', '2024-03-10', freq='D')
        base_prices = [100] * 5 + [95] * 2 + [98, 102, 105, 107, 108, 109, 110] + [107] * 55
        remaining_days = len(dates) - len(base_prices)
        prices = base_prices + [107] * remaining_days
        market_data = pd.DataFrame({'Close': prices}, index=dates)
        strategy.set_data(market_data)
        result = strategy._has_buy_signal(datetime(2024, 3, 10))
        assert result is False

    def test_has_buy_signal_invalid_date(self):
        """Test _has_buy_signal with an invalid date."""
        mock_options_handler = Mock()
        get_contract_list_for_date = mock_options_handler.get_contract_list_for_date
        get_option_bar = mock_options_handler.get_option_bar
        get_options_chain = mock_options_handler.get_options_chain
        strategy = VelocitySignalMomentumStrategy(get_contract_list_for_date=get_contract_list_for_date, get_option_bar=get_option_bar, get_options_chain=get_options_chain)
        dates = pd.date_range('2024-01-01', '2024-03-10', freq='D')
        market_data = pd.DataFrame({'Close': [100 + i * 0.5 for i in range(len(dates))]}, index=dates)
        strategy.set_data(market_data)
        result = strategy._has_buy_signal(datetime(2025, 1, 1))
        assert result is False

    def test_has_buy_signal_edge_case_minimal_reversal(self):
        """Test _has_buy_signal with a minimal reversal that doesn't exceed 2%."""
        mock_options_handler = Mock()
        get_contract_list_for_date = mock_options_handler.get_contract_list_for_date
        get_option_bar = mock_options_handler.get_option_bar
        get_options_chain = mock_options_handler.get_options_chain
        strategy = VelocitySignalMomentumStrategy(get_contract_list_for_date=get_contract_list_for_date, get_option_bar=get_option_bar, get_options_chain=get_options_chain)
        dates = pd.date_range('2024-01-01', '2024-03-10', freq='D')
        base_prices = [100] * 30 + [95] * 2 + [98, 102, 105, 107, 108, 109, 110] + [108.35] * 30
        remaining_days = len(dates) - len(base_prices)
        prices = base_prices + [108.35] * remaining_days
        market_data = pd.DataFrame({'Close': prices}, index=dates)
        strategy.set_data(market_data)
        result = strategy._has_buy_signal(datetime(2024, 2, 2))
        assert isinstance(result, bool)

    def test_set_data_pre_calculates_moving_averages(self):
        """Test that set_data pre-calculates moving averages and velocity."""
        mock_options_handler = Mock()
        get_contract_list_for_date = mock_options_handler.get_contract_list_for_date
        get_option_bar = mock_options_handler.get_option_bar
        get_options_chain = mock_options_handler.get_options_chain
        strategy = VelocitySignalMomentumStrategy(get_contract_list_for_date=get_contract_list_for_date, get_option_bar=get_option_bar, get_options_chain=get_options_chain)
        dates = pd.date_range('2024-01-01', '2024-03-10', freq='D')
        prices = [100 + i * 0.1 for i in range(len(dates))]
        market_data = pd.DataFrame({'Close': prices}, index=dates)
        strategy.set_data(market_data)
        assert 'SMA_15' in strategy.data.columns
        assert 'SMA_30' in strategy.data.columns
        assert 'MA_Velocity_15_30' in strategy.data.columns
        assert 'Velocity_Changes' in strategy.data.columns
        assert not strategy.data['SMA_15'].iloc[20:].isna().all()
        assert not strategy.data['SMA_30'].iloc[35:].isna().all()
        velocity = strategy.data['MA_Velocity_15_30'].iloc[35:]
        assert not velocity.isna().all()
        assert (velocity > 0).all()

    def test_position_entries_tracking(self):
        """Test that position entries are tracked correctly for plotting."""
        mock_options_handler = Mock()
        get_contract_list_for_date = mock_options_handler.get_contract_list_for_date
        get_option_bar = mock_options_handler.get_option_bar
        get_options_chain = mock_options_handler.get_options_chain
        strategy = VelocitySignalMomentumStrategy(get_contract_list_for_date=get_contract_list_for_date, get_option_bar=get_option_bar, get_options_chain=get_options_chain)
        dates = pd.date_range('2024-01-01', '2024-03-10', freq='D')
        prices = [100 + i * 0.1 for i in range(len(dates))]
        market_data = pd.DataFrame({'Close': prices}, index=dates)
        strategy.set_data(market_data)
        assert len(strategy._position_entries) == 0
        entry_date1 = datetime(2024, 1, 15)
        entry_date2 = datetime(2024, 2, 1)
        strategy._position_entries.append(entry_date1)
        strategy._position_entries.append(entry_date2)
        assert len(strategy._position_entries) == 2
        assert entry_date1 in strategy._position_entries
        assert entry_date2 in strategy._position_entries
        strategy.set_data(market_data)
        assert len(strategy._position_entries) == 0

    def test_on_end_plotting_with_no_data(self):
        """Test on_end method when no data is available."""
        mock_options_handler = Mock()
        get_contract_list_for_date = mock_options_handler.get_contract_list_for_date
        get_option_bar = mock_options_handler.get_option_bar
        get_options_chain = mock_options_handler.get_options_chain
        strategy = VelocitySignalMomentumStrategy(get_contract_list_for_date=get_contract_list_for_date, get_option_bar=get_option_bar, get_options_chain=get_options_chain)
        mock_logger = Mock()
        with pytest.MonkeyPatch().context() as m:
            m.setattr('algo_trading_engine.strategies.velocity_signal_momentum_strategy.get_logger', lambda: mock_logger)
            strategy.on_end((), Mock(), datetime.now())
            mock_logger.warning.assert_called_with('⚠️  No data available for plotting')

    def test_on_end_plotting_with_data(self):
        """Test on_end method with valid data (without actually showing plot)."""
        mock_options_handler = Mock()
        get_contract_list_for_date = mock_options_handler.get_contract_list_for_date
        get_option_bar = mock_options_handler.get_option_bar
        get_options_chain = mock_options_handler.get_options_chain
        strategy = VelocitySignalMomentumStrategy(get_contract_list_for_date=get_contract_list_for_date, get_option_bar=get_option_bar, get_options_chain=get_options_chain)
        dates = pd.date_range('2024-01-01', '2024-01-10', freq='D')
        prices = [100 + i * 0.1 for i in range(len(dates))]
        market_data = pd.DataFrame({'Close': prices}, index=dates)
        strategy.set_data(market_data)
        strategy._position_entries = [datetime(2024, 1, 5), datetime(2024, 1, 8)]
        mock_fig = Mock()
        mock_ax1 = Mock()
        mock_ax2 = Mock()
        mock_ax1.get_legend_handles_labels = Mock(return_value=([], []))
        mock_ax2.get_legend_handles_labels = Mock(return_value=([], []))
        mock_ax1.twinx = Mock(return_value=mock_ax2)
        mock_subplots = Mock(return_value=(mock_fig, mock_ax1))
        mock_show = Mock()
        mock_savefig = Mock()
        mock_tight_layout = Mock()
        mock_logger = Mock()
        with pytest.MonkeyPatch().context() as m:
            m.setattr('matplotlib.pyplot.subplots', mock_subplots)
            m.setattr('matplotlib.pyplot.show', mock_show)
            m.setattr('matplotlib.pyplot.savefig', mock_savefig)
            m.setattr('matplotlib.pyplot.tight_layout', mock_tight_layout)
            m.setattr('algo_trading_engine.strategies.velocity_signal_momentum_strategy.get_logger', lambda: mock_logger)
            strategy.on_end((), Mock(), datetime.now())
            mock_subplots.assert_called_once()
            mock_show.assert_called_once()
            mock_tight_layout.assert_called_once()
            mock_ax1.twinx.assert_called_once()

class TestVelocityStrategyFactory:
    """Test cases for VelocitySignalMomentumStrategy via StrategyFactory"""

    def test_factory_passes_profit_target_and_stop_loss(self):
        """Test that StrategyFactory properly passes profit_target and stop_loss to the strategy"""
        from algo_trading_engine.backtest.strategy_builder import StrategyFactory
        mock_options_handler = Mock()
        get_contract_list_for_date = mock_options_handler.get_contract_list_for_date
        get_option_bar = mock_options_handler.get_option_bar
        get_options_chain = mock_options_handler.get_options_chain
        strategy = StrategyFactory.create_strategy('velocity_momentum', get_contract_list_for_date=get_contract_list_for_date, get_option_bar=get_option_bar, get_options_chain=get_options_chain, profit_target=0.2, stop_loss=0.6)
        assert strategy.profit_target == 0.2
        assert strategy.stop_loss == 0.6

    def test_factory_handles_none_parameters(self):
        """Test that StrategyFactory handles None parameters correctly"""
        from algo_trading_engine.backtest.strategy_builder import StrategyFactory
        mock_options_handler = Mock()
        get_contract_list_for_date = mock_options_handler.get_contract_list_for_date
        get_option_bar = mock_options_handler.get_option_bar
        get_options_chain = mock_options_handler.get_options_chain
        strategy = StrategyFactory.create_strategy('velocity_momentum', get_contract_list_for_date=get_contract_list_for_date, get_option_bar=get_option_bar, get_options_chain=get_options_chain)
        assert strategy.profit_target is None
        assert strategy.stop_loss is None

class TestVelocityStrategyMethodSignatures:
    """Test cases to verify method signatures match the base Strategy class"""

    def test_on_new_date_signature_matches_base_class(self):
        """Test that on_new_date can be called with the correct signature from base class"""
        from typing import Callable, Optional
        from algo_trading_engine.core.strategy import Strategy as BaseStrategy
        mock_options_handler = Mock()
        get_contract_list_for_date = mock_options_handler.get_contract_list_for_date
        get_option_bar = mock_options_handler.get_option_bar
        get_options_chain = mock_options_handler.get_options_chain
        strategy = VelocitySignalMomentumStrategy(get_contract_list_for_date=get_contract_list_for_date, get_option_bar=get_option_bar, get_options_chain=get_options_chain)
        dates = pd.date_range('2024-01-01', periods=50, freq='D')
        data = pd.DataFrame({'Close': [100 + i * 0.1 for i in range(50)], 'Open': [100 + i * 0.1 for i in range(50)], 'High': [101 + i * 0.1 for i in range(50)], 'Low': [99 + i * 0.1 for i in range(50)], 'Volume': [1000000] * 50}, index=dates)
        strategy.set_data(data)
        import inspect
        base_sig = inspect.signature(BaseStrategy.on_new_date)
        strategy_sig = inspect.signature(strategy.on_new_date)
        base_params = [p for p in base_sig.parameters.keys() if p != 'self']
        strategy_params = list(strategy_sig.parameters.keys())
        assert strategy_params == base_params, f"Parameter names don't match: {strategy_params} vs {base_params}"
        base_remove_position_type = base_sig.parameters['remove_position'].annotation
        strategy_remove_position_type = strategy_sig.parameters['remove_position'].annotation
        assert 'Callable' in str(base_remove_position_type), f'remove_position should be Callable, got: {base_remove_position_type}'
        assert 'Callable' in str(strategy_remove_position_type), f'remove_position should be Callable, got: {strategy_remove_position_type}'
        date = datetime(2024, 1, 25)
        positions = tuple()

        def mock_add_position(_position: Position):
            pass

        def mock_remove_position(_date: datetime, _position: Position, _exit_price: float, _underlying_price: Optional[float]=None, _current_volumes: Optional[list[int]]=None):
            pass

        def mock_get_price(d, _s):
            return float(data.loc[d, 'Close'])
        strategy.get_current_underlying_price = mock_get_price
        try:
            strategy.on_new_date(date, positions, mock_add_position, mock_remove_position)
        except TypeError as e:
            pytest.fail(f'on_new_date signature mismatch: {e}')

    def test_on_end_signature_matches_base_class(self):
        """Test that on_end can be called with the correct signature from base class"""
        from typing import Callable, Optional
        from algo_trading_engine.core.strategy import Strategy as BaseStrategy
        mock_options_handler = Mock()
        get_contract_list_for_date = mock_options_handler.get_contract_list_for_date
        get_option_bar = mock_options_handler.get_option_bar
        get_options_chain = mock_options_handler.get_options_chain
        strategy = VelocitySignalMomentumStrategy(get_contract_list_for_date=get_contract_list_for_date, get_option_bar=get_option_bar, get_options_chain=get_options_chain)
        dates = pd.date_range('2024-01-01', periods=50, freq='D')
        data = pd.DataFrame({'Close': [100 + i * 0.1 for i in range(50)], 'Open': [100 + i * 0.1 for i in range(50)], 'High': [101 + i * 0.1 for i in range(50)], 'Low': [99 + i * 0.1 for i in range(50)], 'Volume': [1000000] * 50}, index=dates)
        strategy.set_data(data)
        import inspect
        base_sig = inspect.signature(BaseStrategy.on_end)
        strategy_sig = inspect.signature(strategy.on_end)
        base_param_names = {p for p in base_sig.parameters.keys() if p != 'self'}
        strategy_param_names = set(strategy_sig.parameters.keys())
        assert strategy_param_names == base_param_names, f"Parameter names don't match: {strategy_param_names} vs {base_param_names}"
        base_remove_position_type = base_sig.parameters['remove_position'].annotation
        strategy_remove_position_type = strategy_sig.parameters['remove_position'].annotation
        assert 'Callable' in str(base_remove_position_type), f'remove_position should be Callable, got: {base_remove_position_type}'
        assert 'Callable' in str(strategy_remove_position_type), f'remove_position should be Callable, got: {strategy_remove_position_type}'
        positions = tuple()
        date = datetime(2024, 1, 25)

        def mock_remove_position(_date: datetime, _position: Position, _exit_price: float, _underlying_price: Optional[float]=None, _current_volumes: Optional[list[int]]=None):
            pass
        mock_fig = Mock()
        mock_ax1 = Mock()
        mock_ax2 = Mock()
        mock_ax1.get_legend_handles_labels = Mock(return_value=([], []))
        mock_ax2.get_legend_handles_labels = Mock(return_value=([], []))
        mock_ax1.twinx = Mock(return_value=mock_ax2)
        mock_logger = Mock()
        with pytest.MonkeyPatch().context() as m:
            m.setattr('matplotlib.pyplot.subplots', Mock(return_value=(mock_fig, mock_ax1)))
            m.setattr('matplotlib.pyplot.show', Mock())
            m.setattr('matplotlib.pyplot.tight_layout', Mock())
            m.setattr('algo_trading_engine.strategies.velocity_signal_momentum_strategy.get_logger', lambda: mock_logger)
            try:
                strategy.on_end(positions, mock_remove_position, date)
            except TypeError as e:
                pytest.fail(f'on_end signature mismatch: {e}')

    def test_try_close_positions_signature(self):
        """Test that _try_close_positions uses the correct remove_position signature"""
        from typing import Callable, Optional
        mock_options_handler = Mock()
        get_contract_list_for_date = mock_options_handler.get_contract_list_for_date
        get_option_bar = mock_options_handler.get_option_bar
        get_options_chain = mock_options_handler.get_options_chain
        strategy = VelocitySignalMomentumStrategy(get_contract_list_for_date=get_contract_list_for_date, get_option_bar=get_option_bar, get_options_chain=get_options_chain)
        dates = pd.date_range('2024-01-01', periods=50, freq='D')
        data = pd.DataFrame({'Close': [100 + i * 0.1 for i in range(50)], 'Open': [100 + i * 0.1 for i in range(50)], 'High': [101 + i * 0.1 for i in range(50)], 'Low': [99 + i * 0.1 for i in range(50)], 'Volume': [1000000] * 50}, index=dates)
        strategy.set_data(data)
        import inspect
        sig = inspect.signature(strategy._try_close_positions)
        remove_position_param = sig.parameters['remove_position']
        expected_type = "Callable[[datetime, 'Position', float, Optional[float], Optional[list[int]]], None]"
        assert 'datetime' in str(remove_position_param.annotation), f'remove_position should accept datetime as first parameter, got: {remove_position_param.annotation}'
        assert 'Position' in str(remove_position_param.annotation), f'remove_position should accept Position as second parameter, got: {remove_position_param.annotation}'
        assert 'float' in str(remove_position_param.annotation), f'remove_position should accept float as third parameter, got: {remove_position_param.annotation}'
        date = datetime(2024, 1, 25)
        positions = tuple()

        def mock_remove_position(_date: datetime, _position: Position, _exit_price: float, _underlying_price: Optional[float]=None, _current_volumes: Optional[list[int]]=None):
            pass
        strategy.get_option_bar = Mock(return_value=None)

        def mock_get_price(d, _s):
            return float(data.loc[d, 'Close'])
        strategy.get_current_underlying_price = mock_get_price
        try:
            strategy._try_close_positions(date, positions, mock_remove_position)
        except TypeError as e:
            pytest.fail(f'_try_close_positions signature mismatch: {e}')

    def test_integration_with_backtest_engine_signature(self):
        """Integration test: verify strategy can be called by backtest engine with correct signatures"""
        from algo_trading_engine.backtest.main import BacktestEngine
        from typing import Callable, Optional
        mock_options_handler = Mock()
        get_contract_list_for_date = mock_options_handler.get_contract_list_for_date
        get_option_bar = mock_options_handler.get_option_bar
        get_options_chain = mock_options_handler.get_options_chain
        strategy = VelocitySignalMomentumStrategy(get_contract_list_for_date=get_contract_list_for_date, get_option_bar=get_option_bar, get_options_chain=get_options_chain)
        dates = pd.date_range('2024-01-01', periods=50, freq='D')
        data = pd.DataFrame({'Close': [100 + i * 0.1 for i in range(50)], 'Open': [100 + i * 0.1 for i in range(50)], 'High': [101 + i * 0.1 for i in range(50)], 'Low': [99 + i * 0.1 for i in range(50)], 'Volume': [1000000] * 50}, index=dates)
        strategy.set_data(data)
        engine = BacktestEngine(data=data, strategy=strategy, initial_capital=10000, start_date=datetime(2024, 1, 1), end_date=datetime(2024, 1, 10))
        import inspect
        add_pos_sig = inspect.signature(engine._add_position)
        assert len(add_pos_sig.parameters) == 1, f'_add_position should take 1 parameter (Position), got: {list(add_pos_sig.parameters.keys())}'
        remove_pos_sig = inspect.signature(engine._remove_position)
        remove_pos_params = list(remove_pos_sig.parameters.keys())
        expected_params = ['date', 'position', 'exit_price', 'underlying_price', 'current_volumes']
        assert remove_pos_params == expected_params or len(remove_pos_params) >= 3, f"_remove_position parameters don't match: {remove_pos_params} vs {expected_params}"
        test_date = datetime(2024, 1, 5)
        positions_tuple = tuple(engine.positions)
        try:
            strategy.on_new_date(test_date, positions_tuple, engine._add_position, engine._remove_position)
        except TypeError as e:
            pytest.fail(f'Strategy.on_new_date cannot be called by BacktestEngine: {e}')
        mock_fig = Mock()
        mock_ax1 = Mock()
        mock_ax2 = Mock()
        mock_ax1.get_legend_handles_labels = Mock(return_value=([], []))
        mock_ax2.get_legend_handles_labels = Mock(return_value=([], []))
        mock_ax1.twinx = Mock(return_value=mock_ax2)
        mock_logger = Mock()
        with pytest.MonkeyPatch().context() as m:
            m.setattr('matplotlib.pyplot.subplots', Mock(return_value=(mock_fig, mock_ax1)))
            m.setattr('matplotlib.pyplot.show', Mock())
            m.setattr('matplotlib.pyplot.tight_layout', Mock())
            m.setattr('algo_trading_engine.strategies.velocity_signal_momentum_strategy.get_logger', lambda: mock_logger)
            try:
                strategy.on_end(positions_tuple, engine._remove_position, test_date)
            except TypeError as e:
                pytest.fail(f'Strategy.on_end cannot be called by BacktestEngine: {e}')
