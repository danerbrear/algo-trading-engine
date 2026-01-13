from __future__ import annotations

import argparse
import os
import sys

from algo_trading_engine.core.engine import PaperTradingEngine
from algo_trading_engine.models import PaperTradingConfig


def main():
    parser = argparse.ArgumentParser(description="Run interactive recommendation flow for a given date")
    parser.add_argument("--symbol", default="SPY", help="Symbol to trade")
    parser.add_argument("--strategy", default="credit_spread", help="Strategy to run (default: credit_spread)")
    parser.add_argument("--initial-capital", type=float, default=100000, help="Initial capital for paper trading")
    parser.add_argument("--max-position-size", type=float, default=0.40, help="Maximum position size as fraction of capital")
    parser.add_argument("--stop-loss", type=float, default=None, help="Stop loss percentage")
    parser.add_argument("--profit-target", type=float, default=None, help="Profit target percentage")
    parser.add_argument('-f', '--free', action='store_true', default=False,
                       help='Use free tier rate limiting (13 second timeout between API requests)')
    args = parser.parse_args()

    # Get API key from environment
    api_key = os.getenv("POLYGON_API_KEY")

    # Create configuration
    config = PaperTradingConfig(
        initial_capital=args.initial_capital,
        symbol=args.symbol,
        strategy_type=args.strategy,
        max_position_size=args.max_position_size,
        api_key=api_key,
        use_free_tier=args.free,
        stop_loss=args.stop_loss,
        profit_target=args.profit_target
    )

    # Create and run engine using from_config (same as examples)
    print(f"🚀 Starting paper trading with strategy: {args.strategy}")
    print(f"   Symbol: {args.symbol}")
    print(f"   Initial capital: ${args.initial_capital:,.2f}")
    print(f"   Max position size: {args.max_position_size * 100:.1f}%")
    if args.stop_loss:
        print(f"   Stop loss: {args.stop_loss * 100:.1f}%")
    if args.profit_target:
        print(f"   Profit target: {args.profit_target * 100:.1f}%")
    print()

    try:
        engine = PaperTradingEngine.from_config(config)
        success = engine.run()
        
        if success:
            print("✅ Paper trading completed successfully!")
        else:
            print("❌ Paper trading failed!")
            sys.exit(1)
    except Exception as e:
        print(f"❌ Error during paper trading: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
