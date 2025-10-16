"""
Entry point for running the Post-Upward Trend Return Analysis.

This script executes the analysis that measures returns in the 1-3 days following
upward trends in SPY.

Usage:
    python -m src.analysis.run_post_trend_analysis
    python -m src.analysis.run_post_trend_analysis --symbol SPY --months 12
"""

import argparse
import sys
import os

# Add parent directory to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(current_dir, '..')
sys.path.insert(0, src_dir)

from analysis.post_upward_trend_return_analysis import PostUpwardTrendReturnAnalyzer


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Analyze returns in the 1-3 days following upward trends in SPY.'
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
        '--no-plot',
        action='store_true',
        help='Skip creating visualizations'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default='post_upward_trend_return_analysis.png',
        help='Output file path for the plot (default: post_upward_trend_return_analysis.png)'
    )
    
    return parser.parse_args()


def main():
    """Main entry point for the analysis."""
    args = parse_arguments()
    
    print("=" * 80)
    print("POST-UPWARD TREND RETURN ANALYSIS")
    print("=" * 80)
    print(f"Symbol: {args.symbol}")
    print(f"Analysis Period: {args.months} months")
    print("=" * 80)
    print()
    
    try:
        # Initialize analyzer
        analyzer = PostUpwardTrendReturnAnalyzer(
            symbol=args.symbol,
            analysis_period_months=args.months
        )
        
        # Run analysis
        result = analyzer.run_analysis()
        
        # Generate and print report
        report = analyzer.generate_report(result)
        print("\n" + report)
        
        # Create visualizations unless --no-plot is specified
        if not args.no_plot:
            analyzer.plot_results(result, save_path=args.output)
        else:
            print("\n📊 Skipping visualizations (--no-plot specified)")
        
        print("\n✅ Analysis complete!")
        
        return 0
        
    except Exception as e:
        print(f"\n❌ Error during analysis: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())

