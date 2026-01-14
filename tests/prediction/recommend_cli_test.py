#!/usr/bin/env python3
"""
Unit tests for the recommend_cli with new public API.
Tests the paper trading CLI using PaperTradingEngine.from_config().
"""

import unittest
from unittest.mock import Mock, patch
import os

from algo_trading_engine.prediction.recommend_cli import main


class TestRecommendCli(unittest.TestCase):
    """Test cases for the recommend CLI using the new public API."""
    
    @patch('algo_trading_engine.prediction.recommend_cli.PaperTradingEngine')
    @patch('os.getenv')
    def test_paper_trading_cli_creates_config_and_engine(self, mock_getenv, mock_engine_class):
        """Test that CLI creates correct config and engine."""
        # Setup mocks
        mock_getenv.return_value = 'test_api_key'
        mock_engine = Mock()
        mock_engine.run.return_value = True
        mock_engine_class.from_config.return_value = mock_engine
        
        # Mock sys.argv
        with patch('sys.argv', ['recommend_cli.py', '--strategy', 'velocity_momentum', '--symbol', 'SPY']):
            with patch('sys.exit'):  # Prevent actual exit
                main()
        
        # Verify engine was created from config
        mock_engine_class.from_config.assert_called_once()
        config = mock_engine_class.from_config.call_args[0][0]
        
        # Verify config parameters
        self.assertEqual(config.symbol, 'SPY')
        self.assertEqual(config.strategy_type, 'velocity_momentum')
        self.assertEqual(config.api_key, 'test_api_key')
        
        # Verify engine.run was called
        mock_engine.run.assert_called_once()

    @patch('algo_trading_engine.prediction.recommend_cli.PaperTradingEngine')
    @patch('os.getenv')
    def test_paper_trading_cli_with_all_parameters(self, mock_getenv, mock_engine_class):
        """Test CLI with all parameters."""
        # Setup mocks
        mock_getenv.return_value = 'test_api_key'
        mock_engine = Mock()
        mock_engine.run.return_value = True
        mock_engine_class.from_config.return_value = mock_engine
        
        # Mock sys.argv with all parameters
        # Note: initial_capital removed - managed via config/strategies/capital_allocations.json
        with patch('sys.argv', [
            'recommend_cli.py',
            '--strategy', 'credit_spread',
            '--symbol', 'DIA',
            '--max-position-size', '0.25',
            '--stop-loss', '0.6',
            '--profit-target', '0.5',
            '--free'
        ]):
            with patch('sys.exit'):
                main()
        
        # Verify config was created correctly
        mock_engine_class.from_config.assert_called_once()
        config = mock_engine_class.from_config.call_args[0][0]
        
        self.assertEqual(config.symbol, 'DIA')
        self.assertEqual(config.strategy_type, 'credit_spread')
        self.assertEqual(config.max_position_size, 0.25)
        self.assertEqual(config.stop_loss, 0.6)
        self.assertEqual(config.profit_target, 0.5)
        self.assertEqual(config.use_free_tier, True)

    @patch('algo_trading_engine.prediction.recommend_cli.PaperTradingEngine')
    @patch('os.getenv')
    def test_paper_trading_cli_with_defaults(self, mock_getenv, mock_engine_class):
        """Test CLI uses correct defaults."""
        # Setup mocks
        mock_getenv.return_value = 'test_api_key'
        mock_engine = Mock()
        mock_engine.run.return_value = True
        mock_engine_class.from_config.return_value = mock_engine
        
        # Mock sys.argv with minimal parameters
        with patch('sys.argv', ['recommend_cli.py']):
            with patch('sys.exit'):
                main()
        
        # Verify config defaults
        # Note: initial_capital removed - managed via config/strategies/capital_allocations.json
        mock_engine_class.from_config.assert_called_once()
        config = mock_engine_class.from_config.call_args[0][0]
        
        self.assertEqual(config.symbol, 'SPY')
        self.assertEqual(config.strategy_type, 'credit_spread')
        self.assertEqual(config.max_position_size, 0.40)
        self.assertIsNone(config.stop_loss)
        self.assertIsNone(config.profit_target)
        self.assertEqual(config.use_free_tier, False)

    @patch('algo_trading_engine.prediction.recommend_cli.PaperTradingEngine')
    @patch('os.getenv')
    def test_paper_trading_cli_engine_success(self, mock_getenv, mock_engine_class):
        """Test CLI handles successful engine execution."""
        # Setup mocks
        mock_getenv.return_value = 'test_api_key'
        mock_engine = Mock()
        mock_engine.run.return_value = True
        mock_engine_class.from_config.return_value = mock_engine
        
        # Mock sys.argv
        with patch('sys.argv', ['recommend_cli.py']):
            with patch('sys.exit') as mock_exit:
                main()
                
                # Should not exit or exit with 0
                if mock_exit.called:
                    self.assertEqual(mock_exit.call_args[0][0], 0)

    @patch('algo_trading_engine.prediction.recommend_cli.PaperTradingEngine')
    @patch('os.getenv')
    def test_paper_trading_cli_engine_failure(self, mock_getenv, mock_engine_class):
        """Test CLI handles engine execution failure."""
        # Setup mocks
        mock_getenv.return_value = 'test_api_key'
        mock_engine = Mock()
        mock_engine.run.return_value = False  # Failure
        mock_engine_class.from_config.return_value = mock_engine
        
        # Mock sys.argv
        with patch('sys.argv', ['recommend_cli.py']):
            with patch('sys.exit') as mock_exit:
                main()
                
                # Should exit with 1
                mock_exit.assert_called_once_with(1)

    @patch('algo_trading_engine.prediction.recommend_cli.PaperTradingEngine')
    @patch('os.getenv')
    def test_paper_trading_cli_engine_exception(self, mock_getenv, mock_engine_class):
        """Test CLI handles engine exceptions."""
        # Setup mocks
        mock_getenv.return_value = 'test_api_key'
        mock_engine_class.from_config.side_effect = Exception("Test error")
        
        # Mock sys.argv
        with patch('sys.argv', ['recommend_cli.py']):
            with patch('sys.exit') as mock_exit:
                main()
                
                # Should exit with 1
                mock_exit.assert_called_once_with(1)

    @patch('algo_trading_engine.prediction.recommend_cli.PaperTradingEngine')
    @patch('os.getenv')
    def test_argument_parsing(self, mock_getenv, mock_engine_class):
        """Test that CLI parses arguments correctly."""
        # Setup mocks
        mock_getenv.return_value = 'test_api_key'
        mock_engine = Mock()
        mock_engine.run.return_value = True
        mock_engine_class.from_config.return_value = mock_engine
        
        # Test with custom stop-loss and profit-target
        with patch('sys.argv', [
            'recommend_cli.py',
            '--strategy', 'velocity_momentum',
            '--stop-loss', '0.7',
            '--profit-target', '0.3'
        ]):
            with patch('sys.exit'):
                main()
        
        # Verify arguments were parsed correctly
        config = mock_engine_class.from_config.call_args[0][0]
        self.assertEqual(config.stop_loss, 0.7)
        self.assertEqual(config.profit_target, 0.3)


if __name__ == '__main__':
    unittest.main()

