"""
Runner script for VIX Big Move -> SPY Return Analysis

This script executes the VIX-SPY analysis and displays the results.

Usage:
    python -m algo_trading_engine.analysis.run_vix_spy_analysis [--no-plot] [--show-plot]
    
Options:
    --no-plot: Skip plot generation
    --show-plot: Display plot instead of saving to file
"""

import sys
import os
import argparse

# Ensure the package is in the path
current_dir = os.path.dirname(os.path.abspath(__file__))
package_dir = os.path.join(current_dir, '..', '..')
sys.path.insert(0, package_dir)

from algo_trading_engine.analysis.vix_spy_analysis import VIXSPYAnalyzer


def main():
    """Run the VIX-SPY analysis with command-line options."""
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description='VIX Big Move -> SPY Return Analysis',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('--no-plot', action='store_true',
                       help='Skip plot generation')
    parser.add_argument('--show-plot', action='store_true',
                       help='Display plot instead of saving to file')
    
    args = parser.parse_args()
    
    print("="*80)
    print("VIX BIG MOVE -> SPY RETURN ANALYSIS")
    print("="*80)
    print("\nThis analysis identifies big moves in VIX (log returns > 2 std deviations)")
    print("and calculates the average SPY returns in the 3, 6, and 12 days following.")
    print("\n")
    
    # Create analyzer with 5 years of lookback
    analyzer = VIXSPYAnalyzer(lookback_years=5)
    
    # Determine plot settings
    create_plot = not args.no_plot
    save_path = None if args.show_plot else 'predictions/vix_spy_analysis.png'
    
    if create_plot:
        if save_path:
            print(f"Plot will be saved to: {save_path}\n")
        else:
            print("Plot will be displayed (close the plot window to continue)\n")
    
    # Run the complete analysis
    stats = analyzer.run_analysis(create_plot=create_plot, save_plot_path=save_path)
    
    return analyzer, stats


if __name__ == '__main__':
    analyzer, stats = main()
