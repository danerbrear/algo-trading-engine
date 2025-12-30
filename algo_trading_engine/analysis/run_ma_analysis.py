#!/usr/bin/env python3
"""
CLI script to run Moving Average Velocity Analysis

Usage:
    python run_ma_analysis.py [--symbol SYMBOL] [--save-plot PATH]
"""

import argparse
import sys
import os

# Import from the same directory
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))
from ma_velocity_analysis import MAVelocityAnalyzer


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Run Moving Average Velocity Analysis for trend signals',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python run_ma_analysis.py
    python run_ma_analysis.py --symbol QQQ
    python run_ma_analysis.py --symbol QQQ --save-plot qqq_ma_analysis.png
        """
    )
    
    parser.add_argument(
        '--symbol',
        type=str,
        default='SPY',
        help='Stock symbol to analyze (default: SPY)'
    )
    
    parser.add_argument(
        '--save-plot',
        type=str,
        help='Path to save the analysis plot (optional)'
    )
    
    parser.add_argument(
        '--short-periods',
        type=str,
        default='5,10,15,20,25',
        help='Comma-separated list of short MA periods (default: 5,10,15,20,25)'
    )
    
    parser.add_argument(
        '--long-periods',
        type=str,
        default='30,50,100,150,200',
        help='Comma-separated list of long MA periods (default: 30,50,100,150,200)'
    )
    
    parser.add_argument(
        '--min-duration',
        type=int,
        default=3,
        help='Minimum trend duration in days (default: 3)'
    )
    
    parser.add_argument(
        '--max-duration',
        type=int,
        default=60,
        help='Maximum trend duration in days (default: 60)'
    )
    
    return parser.parse_args()


def main():
    """Main function to run the analysis."""
    args = parse_arguments()
    
    print("🚀 Moving Average Velocity Analysis")
    print("=" * 50)
    print(f"Symbol: {args.symbol}")
    print(f"Analysis Period: Last 6 months")
    print(f"Short MA Periods: {args.short_periods}")
    print(f"Long MA Periods: {args.long_periods}")
    print(f"Trend Duration Range: {args.min_duration}-{args.max_duration} days")
    print()
    
    # Parse MA periods
    try:
        short_periods = [int(x.strip()) for x in args.short_periods.split(',')]
        long_periods = [int(x.strip()) for x in args.long_periods.split(',')]
    except ValueError as e:
        print(f"❌ Error parsing MA periods: {e}")
        sys.exit(1)
    
    # Initialize analyzer
    try:
        analyzer = MAVelocityAnalyzer(symbol=args.symbol)
        print(f"📅 Using last 6 months of data (from {analyzer.start_date})")
    except Exception as e:
        print(f"❌ Error initializing analyzer: {e}")
        sys.exit(1)
    
    # Run analysis
    try:
        print("📊 Running analysis...")
        optimal_combinations = analyzer.find_optimal_ma_combinations(short_periods, long_periods)
        
        # Generate and print report
        report = analyzer.generate_report(optimal_combinations)
        print(report)
        
        # Create plots
        if args.save_plot:
            analyzer.plot_results(optimal_combinations, save_path=args.save_plot)
        else:
            analyzer.plot_results(optimal_combinations)
        
        print("\n✅ Analysis complete!")
        
        # Print summary
        print("\n📋 SUMMARY:")
        for trend_type, result in optimal_combinations.items():
            trend_name = "UPWARD" if trend_type == 'up' else "DOWNWARD"
            print(f"{trend_name} TREND: SMA {result.short_ma}/{result.long_ma} "
                  f"(Success Rate: {result.success_rate:.1%})")
        
    except Exception as e:
        print(f"❌ Error during analysis: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
