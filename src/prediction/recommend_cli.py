from __future__ import annotations

import argparse
from datetime import datetime, timedelta
import sys

from src.common.data_retriever import DataRetriever
from src.common.functions import get_model_directory, load_lstm_model, load_hmm_model
from src.model.options_handler import OptionsHandler
from src.prediction.decision_store import JsonDecisionStore
from src.prediction.recommendation_engine import InteractiveStrategyRecommender
from src.strategies.credit_spread_minimal import CreditSpreadStrategy

LOOKBACK_DAYS = 120

STRATEGY_REGISTRY = {
    "credit_spread": CreditSpreadStrategy,
}


def build_strategy(name: str, options_handler: OptionsHandler, symbol: str):
    if name not in STRATEGY_REGISTRY:
        raise ValueError(f"Unknown strategy: {name}")

    # Load trained LSTM model and scaler for prediction use inside the strategy
    model_dir = get_model_directory(symbol=symbol)
    lstm_model, lstm_scaler = load_lstm_model(model_dir, return_lstm_instance=True)

    StrategyClass = STRATEGY_REGISTRY[name]
    strategy = StrategyClass(lstm_model, lstm_scaler, options_handler=options_handler)
    return strategy


def main():
    parser = argparse.ArgumentParser(description="Run interactive recommendation flow for a given date")
    parser.add_argument("--symbol", default="SPY")
    parser.add_argument("--date", help="Run date YYYY-MM-DD; defaults to today")
    parser.add_argument("--strategy", default="credit_spread", help="Strategy to run (default: credit_spread)")
    parser.add_argument("--yes", action="store_true", help="Auto-accept prompts (non-interactive)")
    args = parser.parse_args()

    # Resolve date
    run_date = datetime.now()
    if args.date:
        run_date = datetime.fromisoformat(args.date)

    # If there are open positions, skip historical fetch and only run close flow
    store = JsonDecisionStore()
    open_records = store.get_open_positions(symbol=args.symbol)
    if open_records:
        options_handler = OptionsHandler(args.symbol, quiet_mode=True)
        strategy = build_strategy(args.strategy, options_handler, symbol=args.symbol)
        recommender = InteractiveStrategyRecommender(strategy, options_handler, store, auto_yes=args.yes)

        # Print current status for open positions before prompting to close
        statuses = recommender.get_open_positions_status(run_date)
        if statuses:
            print("\nOpen position status:")
            for s in statuses:
                pnl_dollars = f"${s['pnl_dollars']:.2f}" if s.get('pnl_dollars') is not None else "N/A"
                pnl_pct = f"{s['pnl_percent']:.1%}" if s.get('pnl_percent') is not None else "N/A"
                print(
                    f"- {s['symbol']} {s['strategy_type']} x{s['quantity']} | "
                    f"Entry ${s['entry_price']:.2f}  Exit ${s['exit_price']:.2f} | "
                    f"P&L {pnl_dollars} ({pnl_pct}) | Held {s['days_held']}d  DTE {s['dte']}d"
                )

        recommender.recommend_close_positions(run_date)
        return

    # Prepare data around the run date to ensure the LSTM sequence/features exist
    lstm_start_date = (run_date.date() - timedelta(days=LOOKBACK_DAYS)).strftime("%Y-%m-%d")
    retriever = DataRetriever(symbol=args.symbol, lstm_start_date=lstm_start_date, quiet_mode=True)

    # HMM is required for LSTM features; fail fast if unavailable
    try:
        model_dir = get_model_directory(symbol=args.symbol)
        hmm_model = load_hmm_model(model_dir)
    except Exception as e:
        print(f"❌ ERROR: HMM model is required but could not be loaded: {e}")
        print("   Ensure models are saved under the expected directory or set MODEL_SAVE_BASE_PATH.")
        sys.exit(1)

    data, options_data = retriever.prepare_data_for_lstm(state_classifier=hmm_model)

    # Wire options handler and strategy
    options_handler = retriever.options_handler
    strategy = build_strategy(args.strategy, options_handler, symbol=args.symbol)
    strategy.set_data(data, options_data)

    # Store and recommender
    recommender = InteractiveStrategyRecommender(strategy, options_handler, store, auto_yes=args.yes)
    recommender.run(run_date, auto_yes=args.yes)


if __name__ == "__main__":
    main()


