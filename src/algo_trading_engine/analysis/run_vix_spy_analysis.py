"""
Runner script for VIX Big Move -> SPY Return Analysis

This script executes the VIX-SPY analysis and displays the results.

Usage:
    python -m algo_trading_engine.analysis.run_vix_spy_analysis [OPTIONS]
    
Options:
    --no-plot: Skip plot generation
    --show-plot: Display plot instead of saving to file
    --no-bootstrap: Skip bootstrap significance testing
    --bootstrap-iterations N: Number of bootstrap iterations (default: 10000)
    --lookback-years N: Years of historical data (default: 5)
    --cooldown-days N: Minimum days between signals (default: 12)
    --market-state-filter N: Filter events to only include market state N (default: no filter)
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
    parser.add_argument('--no-bootstrap', action='store_true',
                       help='Skip bootstrap significance testing')
    parser.add_argument('--bootstrap-iterations', type=int, default=10000,
                       help='Number of bootstrap iterations (default: 10000)')
    parser.add_argument('--lookback-years', type=int, default=5,
                       help='Years of historical data to analyze (default: 5)')
    parser.add_argument('--cooldown-days', type=int, default=12,
                       help='Minimum days between independent signals (default: 12)')
    parser.add_argument('--market-state-filter', type=int, default=None,
                       help='Filter events to only include market state N (default: no filter)')
    
    args = parser.parse_args()
    
    print("="*80)
    print("VIX BIG MOVE -> SPY RETURN ANALYSIS")
    print("="*80)
    print("\nThis analysis identifies big moves in VIX (log returns > 2 std deviations)")
    print("and calculates the average SPY returns in the 3, 6, and 12 days following.")
    print(f"\nCooldown period: {args.cooldown_days} days (ensures independent signals)")
    
    if args.market_state_filter is not None:
        print(f"Market state filter: Only events with market state {args.market_state_filter}")
    
    if not args.no_bootstrap:
        print(f"Includes bootstrap significance testing ({args.bootstrap_iterations} iterations)")
    
    print("\n")
    
    # Create analyzer with specified parameters
    analyzer = VIXSPYAnalyzer(lookback_years=args.lookback_years, 
                             cooldown_days=args.cooldown_days)
    
    # Determine plot settings
    create_plot = not args.no_plot
    save_path = None if args.show_plot else 'predictions/vix_spy_analysis.png'
    run_bootstrap = not args.no_bootstrap
    
    if create_plot:
        if save_path:
            print(f"Summary plot will be saved to: {save_path}")
            print(f"Timeline plot will be saved to: {save_path.replace('.png', '_timeline.png')}")
            if run_bootstrap:
                print(f"Bootstrap plot will be saved to: {save_path.replace('.png', '_bootstrap.png')}")
        else:
            print("Plots will be displayed (close each plot window to continue)")
        print()
    
    # Run the complete analysis
    stats, bootstrap_results = analyzer.run_analysis(
        create_plot=create_plot, 
        save_plot_path=save_path,
        run_bootstrap=run_bootstrap,
        n_bootstrap=args.bootstrap_iterations,
        market_state_filter=args.market_state_filter
    )
    
    return analyzer, stats, bootstrap_results


if __name__ == '__main__':
    analyzer, stats, bootstrap_results = main()
