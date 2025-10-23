"""
Entry point for running the Daily Drawdown Likelihood Analysis.

This script can be run from the command line to analyze the likelihood of drawdowns
occurring on each specific day of an upward trend.

Usage:
    python -m src.analysis.run_daily_likelihood_analysis
    python -m src.analysis.run_daily_likelihood_analysis --symbol SPY --months 12
"""

import argparse
import sys
from .daily_drawdown_likelihood_analysis import DailyDrawdownLikelihoodAnalyzer


def parse_arguments():
    """
    Parse command line arguments.
    
    Returns:
        Namespace with parsed arguments
    """
    parser = argparse.ArgumentParser(
        description='Analyze likelihood of drawdowns on each day of upward trends',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with defaults (SPY, 12 months)
  python -m src.analysis.run_daily_likelihood_analysis
  
  # Specify different analysis period
  python -m src.analysis.run_daily_likelihood_analysis --months 6
  
  # Specify custom output file
  python -m src.analysis.run_daily_likelihood_analysis --output my_analysis.png
        """
    )
    
    parser.add_argument(
        '--symbol',
        type=str,
        default='SPY',
        help='Stock symbol to analyze (default: SPY)'
    )
    
    parser.add_argument(
        '--months',
        type=int,
        default=12,
        help='Number of months to analyze (default: 12)'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default='daily_drawdown_likelihood_analysis.png',
        help='Output file path for plot (default: daily_drawdown_likelihood_analysis.png)'
    )
    
    parser.add_argument(
        '--no-plot',
        action='store_true',
        help='Skip plotting and only print report'
    )
    
    return parser.parse_args()


def main():
    """
    Main entry point for the analysis script.
    """
    # Parse command line arguments
    args = parse_arguments()
    
    try:
        print("=" * 80)
        print("DAILY DRAWDOWN LIKELIHOOD ANALYSIS")
        print("=" * 80)
        print(f"Symbol: {args.symbol}")
        print(f"Analysis Period: {args.months} months")
        print("=" * 80)
        print()
        
        # Initialize analyzer
        analyzer = DailyDrawdownLikelihoodAnalyzer(
            symbol=args.symbol,
            analysis_period_months=args.months
        )
        
        # Run analysis
        result = analyzer.run_analysis()
        
        # Generate and print report
        report = analyzer.generate_report(result)
        print("\n" + report)
        
        # Create plots unless --no-plot is specified
        if not args.no_plot:
            analyzer.plot_results(result, save_path=args.output)
        else:
            print("\n⏭️  Skipping plot generation (--no-plot specified)")
        
        print("\n✅ Analysis complete!")
        return 0
        
    except Exception as e:
        print(f"\n❌ Error during analysis: {str(e)}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())

