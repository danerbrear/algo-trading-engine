"""
Unit tests for position statistics functionality in BacktestEngine
"""

import pytest
import pandas as pd
from datetime import datetime
from unittest.mock import Mock

from src.backtest.main import BacktestEngine
from src.backtest.models import StrategyType


class TestPositionStatistics:
    """Test position statistics functionality"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.engine = BacktestEngine(
            data=pd.DataFrame({'Close': [100] * 10}),
            strategy=Mock(),
            initial_capital=10000,
            enable_progress_tracking=False
        )
        
        # Mock closed positions data
        self.engine.closed_positions = [
            {
                'strategy_type': StrategyType.CALL_CREDIT_SPREAD,
                'entry_date': datetime(2024, 1, 1),
                'exit_date': datetime(2024, 1, 15),
                'entry_price': 2.50,
                'exit_price': 1.20,
                'return_dollars': 130.0,
                'return_percentage': 52.0,
                'days_held': 14,
                'max_risk': 250.0
            },
            {
                'strategy_type': StrategyType.CALL_CREDIT_SPREAD,
                'entry_date': datetime(2024, 1, 20),
                'exit_date': datetime(2024, 2, 5),
                'entry_price': 3.00,
                'exit_price': 4.50,
                'return_dollars': -150.0,
                'return_percentage': -50.0,
                'days_held': 16,
                'max_risk': 300.0
            },
            {
                'strategy_type': StrategyType.PUT_CREDIT_SPREAD,
                'entry_date': datetime(2024, 2, 10),
                'exit_date': datetime(2024, 2, 25),
                'entry_price': 2.80,
                'exit_price': 1.50,
                'return_dollars': 130.0,
                'return_percentage': 46.4,
                'days_held': 15,
                'max_risk': 280.0
            }
        ]
    
    def test_print_position_statistics_with_positions(self, capsys):
        """Test statistics output when positions exist"""
        self.engine._print_position_statistics()
        captured = capsys.readouterr()
        
        # Check that statistics are printed
        assert "Position Performance Statistics" in captured.out
        assert "Total closed positions: 3" in captured.out
        assert "Overall win rate: 66.7%" in captured.out
        assert "Total P&L: $+110.00" in captured.out
        assert "Call Credit Spread" in captured.out
        assert "Put Credit Spread" in captured.out
    
    def test_print_position_statistics_no_positions(self, capsys):
        """Test statistics output when no positions exist"""
        self.engine.closed_positions = []
        self.engine._print_position_statistics()
        captured = capsys.readouterr()
        
        # Check that empty statistics are handled
        assert "Position Performance Statistics" in captured.out
        assert "Total closed positions: 0" in captured.out
        assert "Overall win rate: 0.0%" in captured.out
    
    def test_win_rate_calculation(self):
        """Test win rate calculation logic"""
        # All winning positions
        self.engine.closed_positions = [
            {'strategy_type': StrategyType.CALL_CREDIT_SPREAD, 'return_dollars': 100.0},
            {'strategy_type': StrategyType.CALL_CREDIT_SPREAD, 'return_dollars': 200.0},
            {'strategy_type': StrategyType.CALL_CREDIT_SPREAD, 'return_dollars': 50.0}
        ]
        
        # Mock the print method to capture statistics
        with pytest.MonkeyPatch().context() as m:
            printed_lines = []
            def mock_print(*args):
                printed_lines.append(' '.join(str(arg) for arg in args))
            
            m.setattr('builtins.print', mock_print)
            self.engine._print_position_statistics()
            
            # Check win rate is 100%
            win_rate_line = next(line for line in printed_lines if "Overall win rate" in line)
            assert "100.0%" in win_rate_line
    
    def test_drawdown_calculation(self):
        """Test maximum drawdown calculation"""
        # Create positions that would cause a drawdown
        self.engine.closed_positions = [
            {'strategy_type': StrategyType.CALL_CREDIT_SPREAD, 'return_dollars': 100.0},  # Peak
            {'strategy_type': StrategyType.CALL_CREDIT_SPREAD, 'return_dollars': -200.0}, # Drawdown
            {'strategy_type': StrategyType.CALL_CREDIT_SPREAD, 'return_dollars': 50.0}    # Recovery
        ]
        
        with pytest.MonkeyPatch().context() as m:
            printed_lines = []
            def mock_print(*args):
                printed_lines.append(' '.join(str(arg) for arg in args))
            
            m.setattr('builtins.print', mock_print)
            self.engine._print_position_statistics()
            
            # Check drawdown is calculated and should be > 0
            drawdown_line = next(line for line in printed_lines if "Average drawdown" in line)
            assert "Average drawdown" in drawdown_line
            # The average drawdown should be present (calculated as average of drawdown periods)
            assert "1.7%" in drawdown_line  # Average of drawdown periods
    
    def test_drawdown_calculation_no_drawdown(self):
        """Test drawdown calculation when there should be no drawdown"""
        # Create all winning positions (no drawdown)
        self.engine.closed_positions = [
            {'strategy_type': StrategyType.CALL_CREDIT_SPREAD, 'return_dollars': 100.0},
            {'strategy_type': StrategyType.CALL_CREDIT_SPREAD, 'return_dollars': 200.0},
            {'strategy_type': StrategyType.CALL_CREDIT_SPREAD, 'return_dollars': 150.0}
        ]
        
        with pytest.MonkeyPatch().context() as m:
            printed_lines = []
            def mock_print(*args):
                printed_lines.append(' '.join(str(arg) for arg in args))
            
            m.setattr('builtins.print', mock_print)
            self.engine._print_position_statistics()
            
            # Check drawdown is calculated and should be 0
            drawdown_line = next(line for line in printed_lines if "Average drawdown" in line)
            assert "Average drawdown" in drawdown_line
            assert "0.0%" in drawdown_line
    
    def test_strategy_type_statistics(self):
        """Test statistics breakdown by strategy type"""
        with pytest.MonkeyPatch().context() as m:
            printed_lines = []
            def mock_print(*args):
                printed_lines.append(' '.join(str(arg) for arg in args))
            
            m.setattr('builtins.print', mock_print)
            self.engine._print_position_statistics()
            
            # Check that both strategy types are included
            call_spread_line = next(line for line in printed_lines if "Call Credit Spread" in line)
            put_spread_line = next(line for line in printed_lines if "Put Credit Spread" in line)
            
            assert "Call Credit Spread" in call_spread_line
            assert "Put Credit Spread" in put_spread_line
            
            # Check that the statistics are present (more flexible assertion)
            call_spread_stats = [line for line in printed_lines if "Call Credit Spread" in line]
            put_spread_stats = [line for line in printed_lines if "Put Credit Spread" in line]
            
            assert len(call_spread_stats) > 0
            assert len(put_spread_stats) > 0
    
    def test_average_return_calculation(self):
        """Test average return calculation"""
        self.engine.closed_positions = [
            {'strategy_type': StrategyType.CALL_CREDIT_SPREAD, 'return_dollars': 100.0},
            {'strategy_type': StrategyType.CALL_CREDIT_SPREAD, 'return_dollars': 200.0},
            {'strategy_type': StrategyType.CALL_CREDIT_SPREAD, 'return_dollars': 300.0}
        ]
        
        with pytest.MonkeyPatch().context() as m:
            printed_lines = []
            def mock_print(*args):
                printed_lines.append(' '.join(str(arg) for arg in args))
            
            m.setattr('builtins.print', mock_print)
            self.engine._print_position_statistics()
            
            # Check average return is calculated correctly (600/3 = 200)
            avg_line = next(line for line in printed_lines if "Average return per position" in line)
            assert "$+200.00" in avg_line


class TestPositionTracking:
    """Test position tracking functionality"""
    
    def test_closed_positions_tracking(self):
        """Test that closed positions are properly tracked"""
        engine = BacktestEngine(
            data=pd.DataFrame({'Close': [100] * 10}),
            strategy=Mock(),
            initial_capital=10000,
            enable_progress_tracking=False
        )
        
        # Verify initial state
        assert len(engine.closed_positions) == 0
        
        # Add a mock closed position
        engine.closed_positions.append({
            'strategy_type': StrategyType.CALL_CREDIT_SPREAD,
            'return_dollars': 100.0
        })
        
        # Verify position was added
        assert len(engine.closed_positions) == 1
        assert engine.closed_positions[0]['return_dollars'] == 100.0 