from __future__ import annotations

import argparse
from datetime import datetime, timedelta
import sys

from src.common.data_retriever import DataRetriever
from src.common.functions import get_model_directory, load_lstm_model, load_hmm_model
from src.common.options_handler import OptionsHandler
from src.prediction.decision_store import JsonDecisionStore
from src.prediction.recommendation_engine import InteractiveStrategyRecommender
from src.strategies.credit_spread_minimal import CreditSpreadStrategy
from src.strategies.velocity_signal_momentum_strategy import VelocitySignalMomentumStrategy

LOOKBACK_DAYS = 120

STRATEGY_REGISTRY = {
    "credit_spread": CreditSpreadStrategy,
    "velocity_momentum": VelocitySignalMomentumStrategy
}

def build_strategy(name: str, symbol: str, options_handler: OptionsHandler):
    """
    Build strategy and inject the options_handler.
    
    Args:
        name: Strategy name
        symbol: Symbol for the strategy
        options_handler: OptionsHandler instance to inject
    
    Returns:
        Strategy instance with options_handler injected
    """
    if name not in STRATEGY_REGISTRY:
        raise ValueError(f"Unknown strategy: {name}")

    StrategyClass = STRATEGY_REGISTRY[name]

    # Load trained LSTM model and scaler for strategies that need prediction support
    model_dir = get_model_directory(symbol=symbol)
    lstm_model, lstm_scaler = load_lstm_model(model_dir, return_lstm_instance=True)

    # VelocitySignalMomentumStrategy expects different constructor
    if StrategyClass is VelocitySignalMomentumStrategy:
        strategy = StrategyClass()
        # Attach LSTM artifacts for internal prediction usage
        strategy.lstm_model = lstm_model
        strategy.lstm_scaler = lstm_scaler
        # Inject options_handler
        strategy.set_options_handler(options_handler)
        return strategy

    # CreditSpreadStrategy
    strategy = StrategyClass(lstm_model, lstm_scaler, symbol=symbol)
    # Inject options_handler
    strategy.set_options_handler(options_handler)
    return strategy


def main():
    parser = argparse.ArgumentParser(description="Run interactive recommendation flow for a given date")
    parser.add_argument("--symbol", default="SPY")
    parser.add_argument("--date", help="Run date YYYY-MM-DD; defaults to today")
    parser.add_argument("--strategy", default="credit_spread", help="Strategy to run (default: credit_spread)")
    parser.add_argument("--yes", action="store_true", help="Auto-accept prompts (non-interactive)")
    parser.add_argument("--auto-close", action="store_true", default=False, help="Automatically close any open positions recommended to close using previous day's prices")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output (disable quiet mode)")
    parser.add_argument('-f', '--free', action='store_true', default=False,
                       help='Use free tier rate limiting (13 second timeout between API requests)')
    args = parser.parse_args()

    # Resolve date
    run_date = datetime.now()
    if args.date:
        run_date = datetime.fromisoformat(args.date)

    # If there are open positions, skip historical fetch and only run close flow
    store = JsonDecisionStore()
    open_records = store.get_open_positions(symbol=args.symbol)
    if open_records:
        print(f"Open positions found: {len(open_records)}")
        options_handler = OptionsHandler(args.symbol, use_free_tier=args.free)
        strategy = build_strategy(args.strategy, symbol=args.symbol, options_handler=options_handler)
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

        if args.auto_close:
            # Auto-close mode: use the existing logic but with auto_yes=True
            print("\n🔄 Auto-close mode: Checking for positions to close...")
            recommender.auto_yes = True  # Force auto-accept for all prompts
            closed_records = recommender.recommend_close_positions(run_date)
            
            if closed_records:
                print(f"✅ Auto-closed {len(closed_records)} position(s)")
            else:
                print("ℹ️  No open positions are recommended to be closed")
        else:
            # Normal interactive flow
            recommender.recommend_close_positions(run_date)
        return
    
    print(f"No open positions found, running open flow")

    # Create options_handler first
    options_handler = OptionsHandler(args.symbol, use_free_tier=args.free)
    
    # Prepare data around the run date to ensure the LSTM sequence/features exist
    lstm_start_date = (run_date.date() - timedelta(days=LOOKBACK_DAYS)).strftime("%Y-%m-%d")
    retriever = DataRetriever(symbol=args.symbol, lstm_start_date=lstm_start_date, quiet_mode=not args.verbose, use_free_tier=args.free)

    data = retriever.fetch_data_for_period(lstm_start_date, 'recommend')

    # Print date range information for the processed data
    if data is not None and len(data) > 0:
        start_date = data.index[0]
        end_date = data.index[-1]
        print(f"\nDate range: {start_date.date()} to {end_date.date()}\n")

    # Build strategy and inject options_handler
    strategy = build_strategy(args.strategy, symbol=args.symbol, options_handler=options_handler)
    
    # Prepare options data through the strategy
    strategy.set_data(data)

    # Store and recommender
    recommender = InteractiveStrategyRecommender(strategy, options_handler, store, auto_yes=args.yes)
    recommender.run(run_date, auto_yes=args.yes)

if __name__ == "__main__":
    main()
