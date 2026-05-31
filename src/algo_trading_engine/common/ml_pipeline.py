"""
ML feature pipeline for LSTM / HMM workflows.

Import this module only from explicit entry points (credit-spread backtest,
strategy builder, training scripts) so ``DataRetriever`` stays free of sklearn
and hmmlearn at import time.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional, Tuple

import pandas as pd

from algo_trading_engine.common.logger import get_logger

if TYPE_CHECKING:
    from algo_trading_engine.common.data_retriever import DataRetriever
    from algo_trading_engine.ml_models.market_state_classifier import MarketStateClassifier

CREDIT_SPREAD_STRATEGY_NAME = "credit_spread"


def is_credit_spread_strategy(strategy_type: Any) -> bool:
    """Return True when the backtest/paper config targets the credit spread strategy."""
    if isinstance(strategy_type, str):
        return strategy_type == CREDIT_SPREAD_STRATEGY_NAME
    return strategy_type.__class__.__name__ == "CreditSpreadStrategy"


def load_credit_spread_models(symbol: str) -> Tuple[Any, Any]:
    """
    Load trained LSTM model and scaler for credit spread inference.

    Used by ``CreditSpreadStrategyBuilder.build()``.
    """
    from algo_trading_engine.common.functions import get_model_directory, load_lstm_model

    model_dir = get_model_directory(symbol=symbol)
    return load_lstm_model(model_dir, return_lstm_instance=True)


def prepare_data_for_lstm(
    data_retriever: "DataRetriever",
    sequence_length: int = 60,
    state_classifier: Optional["MarketStateClassifier"] = None,
) -> pd.DataFrame:
    """
    Prepare LSTM training data with HMM market states and calendar features.

    Used by ``prepare_training_data()`` and training entry points.
    """
    from algo_trading_engine.ml_models.calendar_features import CalendarFeatureProcessor

    get_logger().info(f"Phase 1: Preparing LSTM training data from {data_retriever.lstm_start_date}")
    data_retriever.lstm_data = data_retriever.fetch_data_for_period(data_retriever.lstm_start_date)
    data_retriever.calculate_features_for_data(data_retriever.lstm_data)

    get_logger().info("Phase 2: Applying trained HMM to LSTM data")
    if state_classifier is not None:
        data_retriever.lstm_data["Market_State"] = state_classifier.predict_states(
            data_retriever.lstm_data
        )
    else:
        get_logger().warning("No state classifier provided, skipping market state prediction")
        data_retriever.lstm_data["Market_State"] = 0

    get_logger().info("Phase 3: Adding economic calendar features")
    calendar_processor = CalendarFeatureProcessor()
    data_retriever.lstm_data = calendar_processor.calculate_all_features(data_retriever.lstm_data)

    data_retriever.data = data_retriever.lstm_data
    return data_retriever.lstm_data


def prepare_training_data(data_retriever: "DataRetriever") -> "MarketStateClassifier":
    """
    Train HMM on historical market data and prepare LSTM feature dataset.

    Entry point for ``StockPredictor.prepare_data()`` and ML training scripts.
    """
    from algo_trading_engine.ml_models.market_state_classifier import MarketStateClassifier

    get_logger().info(f"Phase 1: Preparing HMM training data from {data_retriever.hmm_start_date}")
    hmm_data = data_retriever.fetch_data_for_period(data_retriever.hmm_start_date)
    data_retriever.calculate_features_for_data(hmm_data)

    get_logger().info(f"Phase 2: Training HMM on market data ({len(hmm_data)} samples)")
    state_classifier = MarketStateClassifier()
    state_classifier.train_hmm_model(hmm_data)

    prepare_data_for_lstm(data_retriever, state_classifier=state_classifier)
    return state_classifier


def prepare_credit_spread_backtest_data(
    data: pd.DataFrame,
    data_retriever: "DataRetriever",
    symbol: str,
) -> pd.DataFrame:
    """
    Enrich OHLCV backtest bars with technical, HMM, and calendar features.

    Used by ``BacktestEngine.from_config()`` when ``strategy_type`` is credit spread.
    """
    from algo_trading_engine.common.functions import get_model_directory, load_hmm_model
    from algo_trading_engine.ml_models.calendar_features import CalendarFeatureProcessor

    enriched = data.copy()
    data_retriever.calculate_features_for_data(enriched)

    model_dir = get_model_directory(symbol=symbol)
    try:
        state_classifier = load_hmm_model(model_dir)
        enriched["Market_State"] = state_classifier.predict_states(enriched)
        get_logger().info(f"Applied HMM market states for {symbol} backtest data")
    except (FileNotFoundError, Exception) as exc:
        get_logger().warning(
            f"HMM model unavailable for {symbol} ({exc}); using default Market_State=0"
        )
        enriched["Market_State"] = 0

    calendar_processor = CalendarFeatureProcessor()
    enriched = calendar_processor.calculate_all_features(enriched)
    return enriched
