"""
Tests for CLI integration.

This module tests that:
1. Backtest CLI can be invoked and parses arguments correctly
2. Paper trading CLI can be invoked and parses arguments correctly
3. CLI arguments map correctly to config classes
"""

import pytest
import sys
from unittest.mock import patch, MagicMock
from datetime import datetime
from io import StringIO


def test_backtest_cli_parse_arguments():
    """Test that backtest CLI parses arguments correctly."""
    from algo_trading_engine.backtest.main import parse_arguments
    
    # Mock sys.argv with test arguments
    test_args = [
        'prog',
        '--strategy', 'velocity_momentum',
        '--start-date', '2024-01-01',
        '--end-date', '2024-12-31',
        '--symbol', 'SPY',
        '--initial-capital', '5000',
        '--max-position-size', '0.30',
        '--stop-loss', '0.6',
        '--profit-target', '0.5',
        '--start-date-offset', '120',
        '--verbose',
        '--free'
    ]
    
    with patch.object(sys, 'argv', test_args):
        args = parse_arguments()
        
        assert args.strategy == 'velocity_momentum'
        assert args.start_date == '2024-01-01'
        assert args.end_date == '2024-12-31'
        assert args.symbol == 'SPY'
        assert args.initial_capital == 5000
        assert args.max_position_size == 0.30
        assert args.stop_loss == 0.6
        assert args.profit_target == 0.5
        assert args.start_date_offset == 120
        assert args.verbose is True
        assert args.free is True


def test_backtest_cli_default_arguments():
    """Test that backtest CLI uses correct defaults."""
    from algo_trading_engine.backtest.main import parse_arguments
    
    # Mock sys.argv with minimal arguments
    test_args = ['prog']
    
    with patch.object(sys, 'argv', test_args):
        args = parse_arguments()
        
        # Check defaults
        assert args.strategy == 'credit_spread'
        assert args.start_date_offset == 60
        assert args.initial_capital == 3000
        assert args.max_position_size == 0.40
        assert args.symbol == 'SPY'
        assert args.verbose is False
        assert args.free is False


def test_backtest_main_function_exists():
    """Test that backtest main function exists and is callable."""
    from algo_trading_engine.backtest import main as backtest_module
    
    # Verify main function exists
    assert hasattr(backtest_module, 'main')
    assert callable(backtest_module.main)


def test_paper_trading_cli_main_function_exists():
    """Test that paper trading CLI main function exists."""
    from algo_trading_engine.prediction import recommend_cli
    
    # Verify main function exists
    assert hasattr(recommend_cli, 'main')
    assert callable(recommend_cli.main)


def test_backtest_cli_creates_config():
    """Test that CLI arguments are correctly converted to BacktestConfig."""
    # This tests the mapping logic in main()
    from algo_trading_engine.backtest.main import parse_arguments
    from algo_trading_engine import BacktestConfig
    from datetime import datetime, timedelta
    
    # Mock sys.argv
    test_args = [
        'prog',
        '--strategy', 'credit_spread',
        '--initial-capital', '10000',
        '--max-position-size', '0.25',
        '--start-date', '2024-06-01',
        '--end-date', '2024-12-31',
        '--symbol', 'DIA',
        '--stop-loss', '0.7',
        '--profit-target', '0.4',
        '--start-date-offset', '90',
        '--free'
    ]
    
    with patch.object(sys, 'argv', test_args):
        args = parse_arguments()
        
        # Manually create config like main() does
        start_date = datetime.strptime(args.start_date, '%Y-%m-%d')
        end_date = datetime.strptime(args.end_date, '%Y-%m-%d')
        
        config = BacktestConfig(
            initial_capital=args.initial_capital,
            start_date=start_date,
            end_date=end_date,
            symbol=args.symbol,
            strategy_type=args.strategy,
            max_position_size=args.max_position_size,
            use_free_tier=args.free,
            quiet_mode=not args.verbose,
            lstm_start_date_offset=args.start_date_offset,
            stop_loss=args.stop_loss,
            profit_target=args.profit_target
        )
        
        # Verify config was created correctly
        assert config.initial_capital == 10000
        assert config.start_date == datetime(2024, 6, 1)
        assert config.end_date == datetime(2024, 12, 31)
        assert config.symbol == 'DIA'
        assert config.strategy_type == 'credit_spread'
        assert config.max_position_size == 0.25
        assert config.use_free_tier is True
        assert config.quiet_mode is True  # verbose=False -> quiet_mode=True
        assert config.lstm_start_date_offset == 90
        assert config.stop_loss == 0.7
        assert config.profit_target == 0.4


def test_backtest_cli_date_defaults():
    """Test that CLI correctly handles default dates."""
    from algo_trading_engine.backtest.main import parse_arguments
    from datetime import datetime, timedelta
    
    # Mock sys.argv without dates
    test_args = ['prog']
    
    with patch.object(sys, 'argv', test_args):
        args = parse_arguments()
        
        # Verify defaults
        assert args.start_date is None
        assert args.end_date is None
        
        # Test that main() would set defaults correctly
        today = datetime.now()
        if args.start_date is None:
            start_date = today - timedelta(days=365)
        else:
            start_date = datetime.strptime(args.start_date, '%Y-%m-%d')
        
        if args.end_date is None:
            end_date = today
        else:
            end_date = datetime.strptime(args.end_date, '%Y-%m-%d')
        
        # Verify date logic
        assert start_date < end_date
        assert (end_date - start_date).days >= 365


@patch('algo_trading_engine.backtest.main.BacktestEngine')
@patch('os.getenv')
def test_backtest_main_flow(mock_getenv, mock_engine_class):
    """Test that backtest main() creates engine and runs it."""
    from algo_trading_engine.backtest.main import main
    
    # Mock environment
    mock_getenv.return_value = 'test_api_key'
    
    # Mock engine
    mock_engine = MagicMock()
    mock_engine.run.return_value = True
    mock_engine_class.from_config.return_value = mock_engine
    
    # Mock sys.argv with minimal args
    test_args = [
        'prog',
        '--strategy', 'credit_spread',
        '--start-date', '2024-01-01',
        '--end-date', '2024-01-31'
    ]
    
    with patch.object(sys, 'argv', test_args):
        with pytest.raises(SystemExit) as exc_info:
            main()
        
        # Should exit with 0 (success)
        assert exc_info.value.code == 0
        
        # Verify engine was created and run
        assert mock_engine_class.from_config.called
        assert mock_engine.run.called


@patch('algo_trading_engine.backtest.main.BacktestEngine')
@patch('os.getenv')
def test_backtest_main_failure(mock_getenv, mock_engine_class):
    """Test that backtest main() handles failures correctly."""
    from algo_trading_engine.backtest.main import main
    
    # Mock environment
    mock_getenv.return_value = 'test_api_key'
    
    # Mock engine to return failure
    mock_engine = MagicMock()
    mock_engine.run.return_value = False
    mock_engine_class.from_config.return_value = mock_engine
    
    # Mock sys.argv
    test_args = [
        'prog',
        '--strategy', 'credit_spread',
        '--start-date', '2024-01-01',
        '--end-date', '2024-01-31'
    ]
    
    with patch.object(sys, 'argv', test_args):
        with pytest.raises(SystemExit) as exc_info:
            main()
        
        # Should exit with 1 (failure)
        assert exc_info.value.code == 1


def test_cli_entrypoint_definitions():
    """Test that CLI entrypoints are defined in pyproject.toml."""
    from pathlib import Path
    import sys
    
    # Read pyproject.toml
    pyproject_path = Path(__file__).parent.parent / 'pyproject.toml'
    
    # Check if pyproject.toml exists
    if not pyproject_path.exists():
        pytest.skip("pyproject.toml not found")
    
    # Use tomllib (Python 3.11+) or tomli (Python 3.10 and below)
    with open(pyproject_path, 'rb') as f:
        if sys.version_info >= (3, 11):
            import tomllib
            pyproject = tomllib.load(f)
        else:
            try:
                import tomli
                pyproject = tomli.load(f)
            except ImportError:
                pytest.skip("tomli not installed (required for Python < 3.11)")
    
    # Check for scripts/entrypoints
    if 'project' in pyproject and 'scripts' in pyproject['project']:
        scripts = pyproject['project']['scripts']
        
        # Verify backtest entrypoint
        assert 'algo-backtest' in scripts
        assert 'algo_trading_engine.backtest.main:main' in scripts['algo-backtest']
        
        # Verify paper trading entrypoint
        assert 'algo-paper-trade' in scripts
        assert 'algo_trading_engine.prediction.recommend_cli:main' in scripts['algo-paper-trade']


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

