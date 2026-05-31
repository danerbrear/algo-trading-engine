"""
Tests for ML pipeline lazy loading and entry-point behavior.
"""

import subprocess
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]


def test_data_retriever_import_does_not_load_sklearn_subprocess():
    """Verify a fresh interpreter does not load sklearn with DataRetriever alone."""
    code = (
        "import sys\n"
        "import algo_trading_engine.common.data_retriever\n"
        "print('sklearn' in sys.modules)\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=_REPO_ROOT,
        capture_output=True,
        text=True,
        env={**__import__("os").environ, "PYTHONPATH": str(_REPO_ROOT / "src")},
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "False"


def test_is_credit_spread_strategy():
    from algo_trading_engine.common.ml_pipeline import is_credit_spread_strategy

    assert is_credit_spread_strategy("credit_spread") is True
    assert is_credit_spread_strategy("velocity_momentum") is False


def test_prepare_credit_spread_backtest_data_applies_market_state():
    from algo_trading_engine.common.ml_pipeline import prepare_credit_spread_backtest_data

    dates = pd.date_range("2024-01-01", periods=60, freq="B")
    data = pd.DataFrame(
        {
            "Open": 100.0,
            "High": 101.0,
            "Low": 99.0,
            "Close": 100.0,
            "Volume": 1_000_000,
        },
        index=dates,
    )

    retriever = MagicMock()
    retriever.calculate_features_for_data.side_effect = lambda df, window=20: None

    mock_classifier = MagicMock()
    mock_classifier.predict_states.return_value = pd.Series([1] * len(dates), index=dates)

    with patch(
        "algo_trading_engine.common.functions.get_model_directory",
        return_value="/tmp/models",
    ), patch(
        "algo_trading_engine.common.functions.load_hmm_model",
        return_value=mock_classifier,
    ), patch(
        "algo_trading_engine.ml_models.calendar_features.CalendarFeatureProcessor"
    ) as mock_calendar_cls:
        mock_calendar_cls.return_value.calculate_all_features.side_effect = lambda df: df

        result = prepare_credit_spread_backtest_data(data, retriever, "SPY")

    assert "Market_State" in result.columns
    retriever.calculate_features_for_data.assert_called_once()
    mock_classifier.predict_states.assert_called_once()
