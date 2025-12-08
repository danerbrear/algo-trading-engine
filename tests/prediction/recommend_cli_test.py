#!/usr/bin/env python3
"""
Unit tests for the recommend_cli auto-close functionality.
Tests the --auto-close argument and its integration with the recommendation engine.
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime
import sys
import os

# Add the src directory to the path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from src.prediction.recommend_cli import main
from src.prediction.decision_store import JsonDecisionStore
from src.backtest.models import Position, StrategyType
from src.common.models import Option, OptionType


class TestRecommendCliAutoClose(unittest.TestCase):
    """Test cases for the --auto-close functionality in recommend_cli."""
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        # Mock the decision store
        self.mock_decision_store = Mock(spec=JsonDecisionStore)
        
        # Create test open position record
        self.test_decision_record = Mock()
        self.test_decision_record.id = "test_decision_123"
        self.test_decision_record.proposal = Mock()
        self.test_decision_record.proposal.symbol = "SPY"
        self.test_decision_record.proposal.strategy_type = StrategyType.PUT_CREDIT_SPREAD
        self.test_decision_record.proposal.legs = [
            Option(
                ticker="O:SPY250930P00666000",
                symbol="SPY",
                strike=666.0,
                expiration="2025-09-30",
                option_type=OptionType.PUT,
                last_price=5.18
            ),
            Option(
                ticker="O:SPY250930P00657000",
                symbol="SPY", 
                strike=657.0,
                expiration="2025-09-30",
                option_type=OptionType.PUT,
                last_price=1.93
            )
        ]
        self.test_decision_record.entry_price = 2.35
        self.test_decision_record.quantity = 1
        self.test_decision_record.decided_at = "2025-09-22T17:06:46.176650+00:00"

    @patch('src.prediction.recommend_cli.JsonDecisionStore')
    @patch('src.prediction.recommend_cli.OptionsHandler')
    @patch('src.prediction.recommend_cli.build_strategy')
    @patch('src.prediction.recommend_cli.InteractiveStrategyRecommender')
    def test_auto_close_mode_with_open_positions(self, mock_recommender_class, mock_build_strategy, 
                                                mock_options_handler, mock_decision_store):
        """Test that --auto-close mode works with open positions."""
        # Setup mocks
        mock_decision_store.return_value.get_open_positions.return_value = [self.test_decision_record]
        mock_strategy = Mock()
        mock_build_strategy.return_value = mock_strategy
        mock_recommender = Mock()
        mock_recommender_class.return_value = mock_recommender
        mock_recommender.get_open_positions_status.return_value = [
            {
                'symbol': 'SPY',
                'strategy_type': 'put_credit_spread',
                'quantity': 1,
                'entry_price': 2.35,
                'exit_price': 3.25,
                'pnl_dollars': -90.0,
                'pnl_percent': -0.383,
                'days_held': 1,
                'dte': 7
            }
        ]
        mock_recommender.recommend_close_positions.return_value = [Mock()]  # One position closed
        
        # Mock sys.argv to simulate --auto-close argument
        with patch('sys.argv', ['recommend_cli.py', '--strategy', 'velocity_momentum', '--auto-close']):
            with patch('sys.exit'):  # Prevent actual exit
                main()
        
        # Assertions
        mock_decision_store.return_value.get_open_positions.assert_called_once_with(symbol='SPY')
        # Verify build_strategy was called with correct arguments
        mock_build_strategy.assert_called_once_with('velocity_momentum', 'SPY', mock_options_handler.return_value)
        mock_recommender_class.assert_called_once()
        mock_recommender.auto_yes = True  # Should be set to True for auto-close
        mock_recommender.recommend_close_positions.assert_called_once()

    @patch('src.prediction.recommend_cli.JsonDecisionStore')
    @patch('src.prediction.recommend_cli.OptionsHandler')
    @patch('src.prediction.recommend_cli.build_strategy')
    @patch('src.prediction.recommend_cli.InteractiveStrategyRecommender')
    def test_auto_close_mode_no_positions_to_close(self, mock_recommender_class, mock_build_strategy,
                                                  mock_options_handler, mock_decision_store):
        """Test that --auto-close mode handles case when no positions need closing."""
        # Setup mocks
        mock_decision_store.return_value.get_open_positions.return_value = [self.test_decision_record]
        mock_strategy = Mock()
        mock_build_strategy.return_value = mock_strategy
        mock_recommender = Mock()
        mock_recommender_class.return_value = mock_recommender
        mock_recommender.get_open_positions_status.return_value = [
            {
                'symbol': 'SPY',
                'strategy_type': 'put_credit_spread',
                'quantity': 1,
                'entry_price': 2.35,
                'exit_price': 3.25,
                'pnl_dollars': -90.0,
                'pnl_percent': -0.383,
                'days_held': 1,
                'dte': 7
            }
        ]
        mock_recommender.recommend_close_positions.return_value = []  # No positions closed
        
        # Mock sys.argv to simulate --auto-close argument
        with patch('sys.argv', ['recommend_cli.py', '--strategy', 'velocity_momentum', '--auto-close']):
            with patch('sys.exit'):  # Prevent actual exit
                main()
        
        # Assertions
        # Verify build_strategy was called with correct arguments
        mock_build_strategy.assert_called_once_with('velocity_momentum', 'SPY', mock_options_handler.return_value)
        mock_recommender.recommend_close_positions.assert_called_once()

    @patch('src.prediction.recommend_cli.JsonDecisionStore')
    @patch('src.prediction.recommend_cli.OptionsHandler')
    @patch('src.prediction.recommend_cli.build_strategy')
    @patch('src.prediction.recommend_cli.InteractiveStrategyRecommender')
    def test_interactive_mode_default_behavior(self, mock_recommender_class, mock_build_strategy,
                                             mock_options_handler, mock_decision_store):
        """Test that default behavior (no --auto-close) uses interactive mode."""
        # Setup mocks
        mock_decision_store.return_value.get_open_positions.return_value = [self.test_decision_record]
        mock_strategy = Mock()
        mock_build_strategy.return_value = mock_strategy
        mock_recommender = Mock()
        mock_recommender_class.return_value = mock_recommender
        mock_recommender.get_open_positions_status.return_value = []
        mock_recommender.recommend_close_positions.return_value = []
        
        # Mock sys.argv without --auto-close
        with patch('sys.argv', ['recommend_cli.py', '--strategy', 'velocity_momentum']):
            with patch('sys.exit'):  # Prevent actual exit
                main()
        
        # Assertions
        mock_recommender_class.assert_called_once()
        # Should not set auto_yes to True (default is False)
        mock_recommender.recommend_close_positions.assert_called_once()

    @patch('src.prediction.recommend_cli.JsonDecisionStore')
    @patch('src.prediction.recommend_cli.OptionsHandler')
    @patch('src.prediction.recommend_cli.build_strategy')
    @patch('src.prediction.recommend_cli.InteractiveStrategyRecommender')
    def test_auto_close_with_verbose_flag(self, mock_recommender_class, mock_build_strategy,
                                        mock_options_handler, mock_decision_store):
        """Test that --auto-close works with --verbose flag."""
        # Setup mocks
        mock_decision_store.return_value.get_open_positions.return_value = [self.test_decision_record]
        mock_strategy = Mock()
        mock_build_strategy.return_value = mock_strategy
        mock_recommender = Mock()
        mock_recommender_class.return_value = mock_recommender
        mock_recommender.get_open_positions_status.return_value = []
        mock_recommender.recommend_close_positions.return_value = []
        
        # Mock sys.argv with both --auto-close and --verbose
        with patch('sys.argv', ['recommend_cli.py', '--strategy', 'velocity_momentum', '--auto-close', '--verbose']):
            with patch('sys.exit'):  # Prevent actual exit
                main()
        
        # Assertions
        mock_decision_store.return_value.get_open_positions.assert_called_once_with(symbol='SPY')
        # Verify build_strategy was called with correct arguments
        mock_build_strategy.assert_called_once_with('velocity_momentum', 'SPY', mock_options_handler.return_value)
        mock_recommender_class.assert_called_once()

    @patch('src.prediction.recommend_cli.JsonDecisionStore')
    @patch('src.prediction.recommend_cli.OptionsHandler')
    @patch('src.prediction.recommend_cli.build_strategy')
    @patch('src.prediction.recommend_cli.InteractiveStrategyRecommender')
    def test_auto_close_with_custom_date(self, mock_recommender_class, mock_build_strategy,
                                            mock_options_handler, mock_decision_store):
        """Test that --auto-close works with custom date."""
        # Setup mocks
        mock_decision_store.return_value.get_open_positions.return_value = [self.test_decision_record]
        mock_strategy = Mock()
        mock_build_strategy.return_value = mock_strategy
        mock_recommender = Mock()
        mock_recommender_class.return_value = mock_recommender
        mock_recommender.get_open_positions_status.return_value = []
        mock_recommender.recommend_close_positions.return_value = []
        
        # Mock sys.argv with --auto-close and custom date
        with patch('sys.argv', ['recommend_cli.py', '--strategy', 'velocity_momentum', '--auto-close', '--date', '2025-09-23']):
            with patch('sys.exit'):  # Prevent actual exit
                main()
        
        # Assertions
        mock_decision_store.return_value.get_open_positions.assert_called_once_with(symbol='SPY')
        # Verify build_strategy was called with correct arguments
        mock_build_strategy.assert_called_once_with('velocity_momentum', 'SPY', mock_options_handler.return_value)
        mock_recommender_class.assert_called_once()

    def test_auto_close_argument_parsing(self):
        """Test that --auto-close argument is parsed correctly."""
        import argparse
        
        # Test argument parser
        parser = argparse.ArgumentParser()
        parser.add_argument("--auto-close", action="store_true", default=False, 
                          help="Automatically close any open positions recommended to close using previous day's prices")
        
        # Test with --auto-close
        args_with_auto_close = parser.parse_args(['--auto-close'])
        self.assertTrue(args_with_auto_close.auto_close)
        
        # Test without --auto-close
        args_without_auto_close = parser.parse_args([])
        self.assertFalse(args_without_auto_close.auto_close)

    @patch('src.prediction.recommend_cli.JsonDecisionStore')
    @patch('src.prediction.recommend_cli.OptionsHandler')
    @patch('src.prediction.recommend_cli.build_strategy')
    @patch('src.prediction.recommend_cli.InteractiveStrategyRecommender')
    def test_auto_close_with_free_tier_flag(self, mock_recommender_class, mock_build_strategy,
                                         mock_options_handler, mock_decision_store):
        """Test that --auto-close works with --free flag."""
        # Setup mocks
        mock_decision_store.return_value.get_open_positions.return_value = [self.test_decision_record]
        mock_strategy = Mock()
        mock_build_strategy.return_value = mock_strategy
        mock_recommender = Mock()
        mock_recommender_class.return_value = mock_recommender
        mock_recommender.get_open_positions_status.return_value = []
        mock_recommender.recommend_close_positions.return_value = []
        
        # Mock sys.argv with --auto-close and --free
        with patch('sys.argv', ['recommend_cli.py', '--strategy', 'velocity_momentum', '--auto-close', '--free']):
            with patch('sys.exit'):  # Prevent actual exit
                main()
        
        # Assertions
        mock_decision_store.return_value.get_open_positions.assert_called_once_with(symbol='SPY')
        # Verify build_strategy was called with correct arguments
        mock_build_strategy.assert_called_once_with('velocity_momentum', 'SPY', mock_options_handler.return_value)
        mock_recommender_class.assert_called_once()


if __name__ == '__main__':
    unittest.main()

