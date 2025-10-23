#!/usr/bin/env python3
"""
Entry point for Bullish Market Regime Drawdown Analysis

This script runs the analysis to determine which bullish market regime
experiences the most drawdowns during upward trends.
"""

import argparse
import sys
import os
from datetime import datetime

# Add src to path
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(current_dir, '..')
sys.path.insert(0, src_dir)

from analysis.bullish_regime_drawdown_analysis import BullishRegimeDrawdownAnalyzer

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Analyze drawdowns during upward trends by bullish market regime"
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
        help='Number of months of data to analyze (default: 12). HMM will be trained on 24 months prior to this period.'
    )
    
    parser.add_argument(
        '--save-plot', 
        type=str, 
        default=None,
        help='Path to save the analysis plot (optional)'
    )
    
    parser.add_argument(
        '--quiet', 
        action='store_true',
        help='Suppress detailed progress output'
    )
    
    return parser.parse_args()

def main():
    """Main execution function"""
    args = parse_arguments()
    
    print("🚀 Starting Bullish Market Regime Drawdown Analysis")
    print(f"Symbol: {args.symbol}")
    print(f"Analysis Period: {args.months} months")
    print(f"HMM Training Period: 24 months prior to analysis period")
    print(f"Total Data Period: {args.months + 24} months")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    try:
        # Initialize analyzer
        analyzer = BullishRegimeDrawdownAnalyzer(
            symbol=args.symbol,
            analysis_period_months=args.months
        )
        
        # Run analysis
        result = analyzer.run_analysis()
        
        # Generate and print report
        report = analyzer.generate_report(result)
        print("\n" + report)
        
        # Create visualization
        analyzer.plot_results(result, save_path=args.save_plot)
        
        print(f"\n✅ Analysis completed successfully!")
        print(f"Regime with most trends: {result.regime_with_most_trends.value}")
        print(f"Regime with highest mean drawdown: {result.regime_with_highest_mean_drawdown.value}")
        
    except Exception as e:
        print(f"\n❌ Analysis failed: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
