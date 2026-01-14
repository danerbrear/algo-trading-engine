"""
Tests for equity curve plotting functionality with overlay features.
"""

from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock
import pandas as pd

from algo_trading_engine.prediction.plot_equity_curve import (
    ClosedPosition,
    calculate_equity_curve,
    calculate_drawdowns,
    fetch_spy_data,
    fetch_treasury_rates,
    plot_equity_curve,
)


class TestClosedPosition:
    """Test cases for ClosedPosition dataclass"""

    def test_from_decision_dict_credit_spread(self):
        """Test creating ClosedPosition from credit spread decision"""
        decision = {
            'closed_at': '2025-11-01T15:30:00+00:00',
            'outcome': 'accepted',
            'entry_price': 2.50,
            'exit_price': 1.00,
            'quantity': 1,
            'proposal': {
                'strategy_name': 'velocity_signal_momentum',
                'strategy_type': 'put_credit_spread'
            }
        }

        position = ClosedPosition.from_decision_dict(decision)

        assert position is not None
        assert position.entry_price == 2.50
        assert position.exit_price == 1.00
        assert position.quantity == 1
        assert position.strategy_name == 'velocity_signal_momentum'
        assert position.strategy_type == 'put_credit_spread'
        # Credit spread: profit when exit < entry
        assert position.pnl == (2.50 - 1.00) * 1 * 100
        assert position.pnl == 150.0

    def test_from_decision_dict_debit_spread(self):
        """Test creating ClosedPosition from debit spread decision"""
        decision = {
            'closed_at': '2025-11-01T15:30:00+00:00',
            'outcome': 'accepted',
            'entry_price': 1.00,
            'exit_price': 2.00,
            'quantity': 2,
            'proposal': {
                'strategy_name': 'test_strategy',
                'strategy_type': 'put_debit_spread'
            }
        }

        position = ClosedPosition.from_decision_dict(decision)

        assert position is not None
        # Debit spread: profit when exit > entry
        assert position.pnl == (2.00 - 1.00) * 2 * 100
        assert position.pnl == 200.0

    def test_from_decision_dict_not_closed(self):
        """Test that open positions return None"""
        decision = {
            'closed_at': None,
            'outcome': 'accepted',
            'entry_price': 2.50,
            'exit_price': None,
            'quantity': 1,
            'proposal': {
                'strategy_name': 'test',
                'strategy_type': 'put_credit_spread'
            }
        }

        position = ClosedPosition.from_decision_dict(decision)
        assert position is None

    def test_from_decision_dict_rejected(self):
        """Test that rejected positions return None"""
        decision = {
            'closed_at': '2025-11-01T15:30:00+00:00',
            'outcome': 'rejected',
            'entry_price': 2.50,
            'exit_price': 1.00,
            'quantity': 1,
            'proposal': {
                'strategy_name': 'test',
                'strategy_type': 'put_credit_spread'
            }
        }

        position = ClosedPosition.from_decision_dict(decision)
        assert position is None


class TestCalculateEquityCurve:
    """Test cases for equity curve calculation"""

    def test_calculate_equity_curve_empty(self):
        """Test equity curve with no positions"""
        dates, capital = calculate_equity_curve([], initial_capital=10000.0)
        assert dates == []
        assert capital == []

    def test_calculate_equity_curve_single_position(self):
        """Test equity curve with single position"""
        closed_at = datetime(2025, 11, 1, 15, 0, 0)
        position = ClosedPosition(
            closed_at=closed_at,
            entry_price=2.50,
            exit_price=1.00,
            quantity=1,
            strategy_name='test',
            strategy_type='put_credit_spread',
            pnl=150.0
        )

        dates, capital = calculate_equity_curve([position], initial_capital=10000.0)

        assert len(dates) == 2
        assert len(capital) == 2
        # First point: start capital
        assert capital[0] == 10000.0
        assert dates[0] == closed_at - timedelta(days=1)
        # Second point: after trade
        assert capital[1] == 10150.0
        assert dates[1] == closed_at

    def test_calculate_equity_curve_multiple_positions(self):
        """Test equity curve with multiple positions"""
        positions = [
            ClosedPosition(
                closed_at=datetime(2025, 11, 1, 15, 0, 0),
                entry_price=2.50,
                exit_price=1.00,
                quantity=1,
                strategy_name='test',
                strategy_type='put_credit_spread',
                pnl=150.0
            ),
            ClosedPosition(
                closed_at=datetime(2025, 11, 5, 15, 0, 0),
                entry_price=3.00,
                exit_price=1.50,
                quantity=1,
                strategy_name='test',
                strategy_type='put_credit_spread',
                pnl=150.0
            ),
            ClosedPosition(
                closed_at=datetime(2025, 11, 10, 15, 0, 0),
                entry_price=2.00,
                exit_price=3.00,
                quantity=1,
                strategy_name='test',
                strategy_type='put_credit_spread',
                pnl=-100.0  # Loss
            )
        ]

        dates, capital = calculate_equity_curve(positions, initial_capital=10000.0)

        assert len(dates) == 4  # Start + 3 positions
        assert len(capital) == 4
        assert capital[0] == 10000.0
        assert capital[1] == 10150.0
        assert capital[2] == 10300.0
        assert capital[3] == 10200.0


class TestCalculateDrawdowns:
    """Test cases for drawdown calculation"""

    def test_calculate_drawdowns_empty(self):
        """Test drawdown calculation with empty list"""
        drawdowns = calculate_drawdowns([])
        assert drawdowns == []

    def test_calculate_drawdowns_single_value(self):
        """Test drawdown calculation with single value"""
        drawdowns = calculate_drawdowns([10000.0])
        assert drawdowns == []

    def test_calculate_drawdowns_no_drawdown(self):
        """Test drawdown calculation with no drawdowns (always increasing)"""
        capital = [10000.0, 10100.0, 10200.0, 10300.0]
        drawdowns = calculate_drawdowns(capital)
        assert drawdowns == []

    def test_calculate_drawdowns_single_drawdown(self):
        """Test drawdown calculation with single drawdown"""
        # Peak at 10200, drops to 9500 (6.86% drawdown), then 9600 (5.88% drawdown)
        capital = [10000.0, 10100.0, 10200.0, 9500.0, 9600.0]
        drawdowns = calculate_drawdowns(capital)
        assert len(drawdowns) == 2  # Two points below peak
        # First drawdown: (10200 - 9500) / 10200 * 100 = 6.86%
        assert abs(drawdowns[0] - 6.86) < 0.1
        # Second drawdown: (10200 - 9600) / 10200 * 100 = 5.88%
        assert abs(drawdowns[1] - 5.88) < 0.1

    def test_calculate_drawdowns_multiple_drawdowns(self):
        """Test drawdown calculation with multiple drawdown periods"""
        # First peak at 10000, drops to 9000 (10% drawdown)
        # Then recovers to 11000 (new peak), drops to 9900 (10% drawdown)
        capital = [10000.0, 9000.0, 9500.0, 11000.0, 9900.0, 10500.0]
        drawdowns = calculate_drawdowns(capital)
        assert len(drawdowns) > 0
        # Should have drawdowns from both periods
        assert max(drawdowns) >= 10.0  # At least 10% max drawdown

    def test_calculate_drawdowns_with_zero_peak(self):
        """Test drawdown calculation when peak is zero (edge case)"""
        capital = [0.0, 1000.0, 900.0]
        drawdowns = calculate_drawdowns(capital)
        # Should handle zero peak gracefully
        assert isinstance(drawdowns, list)

    def test_calculate_drawdowns_recovery_after_drawdown(self):
        """Test that recovery after drawdown doesn't create negative drawdowns"""
        # Peak at 10000, drops to 8000 (20% drawdown), recovers to 12000
        capital = [10000.0, 8000.0, 12000.0]
        drawdowns = calculate_drawdowns(capital)
        # Should only have drawdowns when below peak
        assert all(dd >= 0 for dd in drawdowns)
        assert max(drawdowns) >= 20.0  # At least 20% max drawdown


class TestFetchSpyData:
    """Test cases for SPY data fetching"""

    @patch('algo_trading_engine.prediction.plot_equity_curve.yf.Ticker')
    def test_fetch_spy_data_success(self, mock_ticker):
        """Test successful SPY data fetch"""
        # Mock yfinance data
        mock_data = pd.DataFrame({
            'Close': [450.0, 452.0, 455.0]
        }, index=pd.date_range('2025-11-01', periods=3))

        mock_ticker_instance = Mock()
        mock_ticker_instance.history.return_value = mock_data
        mock_ticker.return_value = mock_ticker_instance

        start_date = datetime(2025, 11, 1)
        end_date = datetime(2025, 11, 3)

        result = fetch_spy_data(start_date, end_date)

        assert result is not None
        assert len(result) == 3
        assert 'Close' in result.columns
        mock_ticker.assert_called_once_with("SPY")

    @patch('algo_trading_engine.prediction.plot_equity_curve.yf.Ticker')
    def test_fetch_spy_data_empty(self, mock_ticker):
        """Test SPY data fetch with empty result"""
        mock_ticker_instance = Mock()
        mock_ticker_instance.history.return_value = pd.DataFrame()
        mock_ticker.return_value = mock_ticker_instance

        start_date = datetime(2025, 11, 1)
        end_date = datetime(2025, 11, 3)

        result = fetch_spy_data(start_date, end_date)

        assert result is None

    @patch('algo_trading_engine.prediction.plot_equity_curve.yf.Ticker')
    def test_fetch_spy_data_exception(self, mock_ticker):
        """Test SPY data fetch with exception"""
        mock_ticker.side_effect = Exception("API error")

        start_date = datetime(2025, 11, 1)
        end_date = datetime(2025, 11, 3)

        result = fetch_spy_data(start_date, end_date)

        assert result is None


class TestFetchTreasuryRates:
    """Test cases for treasury rates fetching"""

    @patch('algo_trading_engine.prediction.plot_equity_curve.yf.Ticker')
    def test_fetch_treasury_rates_success(self, mock_ticker):
        """Test successful treasury rates fetch"""
        # Mock yfinance data for ^TNX
        mock_data = pd.DataFrame({
            'Close': [4.25, 4.30, 4.28]
        }, index=pd.date_range('2025-11-01', periods=3))

        mock_ticker_instance = Mock()
        mock_ticker_instance.history.return_value = mock_data
        mock_ticker.return_value = mock_ticker_instance

        start_date = datetime(2025, 11, 1)
        end_date = datetime(2025, 11, 3)

        result = fetch_treasury_rates(start_date, end_date)

        assert result is not None
        assert len(result) == 3
        assert '10Y_Rate' in result.columns
        assert result['10Y_Rate'].iloc[0] == 4.25
        mock_ticker.assert_called_once_with("^TNX")

    @patch('algo_trading_engine.prediction.plot_equity_curve.yf.Ticker')
    def test_fetch_treasury_rates_empty(self, mock_ticker):
        """Test treasury rates fetch with empty result"""
        mock_ticker_instance = Mock()
        mock_ticker_instance.history.return_value = pd.DataFrame()
        mock_ticker.return_value = mock_ticker_instance

        start_date = datetime(2025, 11, 1)
        end_date = datetime(2025, 11, 3)

        result = fetch_treasury_rates(start_date, end_date)

        assert result is None

    @patch('algo_trading_engine.prediction.plot_equity_curve.yf.Ticker')
    def test_fetch_treasury_rates_exception(self, mock_ticker):
        """Test treasury rates fetch with exception"""
        mock_ticker.side_effect = Exception("API error")

        start_date = datetime(2025, 11, 1)
        end_date = datetime(2025, 11, 3)

        result = fetch_treasury_rates(start_date, end_date)

        assert result is None


class TestPlotEquityCurve:
    """Test cases for plot_equity_curve function"""

    @patch('algo_trading_engine.prediction.plot_equity_curve.plt')
    def test_plot_equity_curve_no_positions(self, mock_plt):
        """Test plotting with no positions"""
        plot_equity_curve([], show_plot=False)
        # Should not create any plot
        mock_plt.subplots.assert_not_called()

    @patch('algo_trading_engine.prediction.plot_equity_curve.plt')
    @patch('algo_trading_engine.prediction.plot_equity_curve.fetch_spy_data')
    @patch('algo_trading_engine.prediction.plot_equity_curve.fetch_treasury_rates')
    def test_plot_equity_curve_with_spy_overlay(
        self, mock_fetch_rates, mock_fetch_spy, mock_plt
    ):
        """Test plotting with SPY overlay"""
        # Create test positions
        positions = [
            ClosedPosition(
                closed_at=datetime(2025, 11, 1, 15, 0, 0),
                entry_price=2.50,
                exit_price=1.00,
                quantity=1,
                strategy_name='test',
                strategy_type='put_credit_spread',
                pnl=150.0
            )
        ]

        # Mock SPY data
        spy_data = pd.DataFrame({
            'Close': [450.0, 452.0]
        }, index=pd.date_range('2025-11-01', periods=2))
        mock_fetch_spy.return_value = spy_data

        # Mock matplotlib
        fig, ax1 = MagicMock(), MagicMock()
        ax2 = MagicMock()
        mock_plt.subplots.return_value = (fig, ax1)
        ax1.twinx.return_value = ax2
        # Mock get_legend_handles_labels to return empty lists
        ax1.get_legend_handles_labels.return_value = ([], [])
        ax2.get_legend_handles_labels.return_value = ([], [])

        plot_equity_curve(
            positions,
            show_plot=False,
            overlay_spy=True
        )

        # Verify SPY data was fetched
        mock_fetch_spy.assert_called_once()
        # Verify secondary axis was created
        ax1.twinx.assert_called()
        # Verify plot was created
        mock_plt.subplots.assert_called_once()

    @patch('algo_trading_engine.prediction.plot_equity_curve.plt')
    @patch('algo_trading_engine.prediction.plot_equity_curve.fetch_spy_data')
    @patch('algo_trading_engine.prediction.plot_equity_curve.fetch_treasury_rates')
    def test_plot_equity_curve_with_rates_overlay(
        self, mock_fetch_rates, mock_fetch_spy, mock_plt
    ):
        """Test plotting with treasury rates overlay"""
        # Create test positions
        positions = [
            ClosedPosition(
                closed_at=datetime(2025, 11, 1, 15, 0, 0),
                entry_price=2.50,
                exit_price=1.00,
                quantity=1,
                strategy_name='test',
                strategy_type='put_credit_spread',
                pnl=150.0
            )
        ]

        # Mock treasury rates data
        rates_data = pd.DataFrame({
            '10Y_Rate': [4.25, 4.30]
        }, index=pd.date_range('2025-11-01', periods=2))
        mock_fetch_rates.return_value = rates_data

        # Mock matplotlib
        fig, ax1 = MagicMock(), MagicMock()
        ax3 = MagicMock()
        mock_plt.subplots.return_value = (fig, ax1)
        ax1.twinx.return_value = ax3
        # Mock get_legend_handles_labels to return empty lists
        ax1.get_legend_handles_labels.return_value = ([], [])
        ax3.get_legend_handles_labels.return_value = ([], [])

        plot_equity_curve(
            positions,
            show_plot=False,
            overlay_rates=True
        )

        # Verify rates data was fetched
        mock_fetch_rates.assert_called_once()
        # Verify secondary axis was created
        ax1.twinx.assert_called()

    @patch('algo_trading_engine.prediction.plot_equity_curve.plt')
    @patch('algo_trading_engine.prediction.plot_equity_curve.fetch_spy_data')
    @patch('algo_trading_engine.prediction.plot_equity_curve.fetch_treasury_rates')
    def test_plot_equity_curve_with_both_overlays(
        self, mock_fetch_rates, mock_fetch_spy, mock_plt
    ):
        """Test plotting with both SPY and rates overlays"""
        # Create test positions
        positions = [
            ClosedPosition(
                closed_at=datetime(2025, 11, 1, 15, 0, 0),
                entry_price=2.50,
                exit_price=1.00,
                quantity=1,
                strategy_name='test',
                strategy_type='put_credit_spread',
                pnl=150.0
            )
        ]

        # Mock data
        spy_data = pd.DataFrame({
            'Close': [450.0, 452.0]
        }, index=pd.date_range('2025-11-01', periods=2))
        rates_data = pd.DataFrame({
            '10Y_Rate': [4.25, 4.30]
        }, index=pd.date_range('2025-11-01', periods=2))
        mock_fetch_spy.return_value = spy_data
        mock_fetch_rates.return_value = rates_data

        # Mock matplotlib
        fig, ax1 = MagicMock(), MagicMock()
        ax2, ax3 = MagicMock(), MagicMock()
        mock_plt.subplots.return_value = (fig, ax1)
        # First twinx call for SPY, second for rates
        ax1.twinx.side_effect = [ax2, ax3]
        # Mock get_legend_handles_labels to return empty lists
        ax1.get_legend_handles_labels.return_value = ([], [])
        ax2.get_legend_handles_labels.return_value = ([], [])
        ax3.get_legend_handles_labels.return_value = ([], [])

        plot_equity_curve(
            positions,
            show_plot=False,
            overlay_spy=True,
            overlay_rates=True
        )

        # Verify both data sources were fetched
        mock_fetch_spy.assert_called_once()
        mock_fetch_rates.assert_called_once()
        # Verify two secondary axes were created
        assert ax1.twinx.call_count == 2

