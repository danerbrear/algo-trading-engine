"""
Plot equity curves from closed trading positions.

This script reads decision JSON files and generates equity curves showing
capital remaining over time based on realized P&L. Supports filtering by strategy name.
Can overlay SPY price and treasury interest rates for comparison.

Usage:
    python -m src.prediction.plot_equity_curve                    # Plot all strategies
    python -m src.prediction.plot_equity_curve --strategy velocity_signal_momentum
    python -m src.prediction.plot_equity_curve --strategy upward_trend_reversal
    python -m src.prediction.plot_equity_curve --output equity_curve.png
    python -m src.prediction.plot_equity_curve --overlay-spy      # Overlay SPY price
    python -m src.prediction.plot_equity_curve --overlay-rates    # Overlay interest rates
    python -m src.prediction.plot_equity_curve --overlay-spy --overlay-rates  # Both
"""

import argparse
import json
import os
from datetime import datetime, timedelta
from typing import List, Dict, Optional
from dataclasses import dataclass
from pathlib import Path
from functools import reduce

import numpy as np
import pandas as pd
import yfinance as yf

from algo_trading_engine.plotting import PlotConfig, build_plot_spec, show_plot


@dataclass
class ClosedPosition:
    """Represents a closed position for equity curve calculation."""
    closed_at: datetime
    entry_price: float
    exit_price: float
    quantity: int
    strategy_name: str
    strategy_type: str  # e.g., put_credit_spread, put_debit_spread
    pnl: float
    
    @classmethod
    def from_decision_dict(cls, decision: dict) -> Optional['ClosedPosition']:
        """Create ClosedPosition from decision JSON dict."""
        # Only process closed positions
        if decision.get('closed_at') is None:
            return None
        
        try:
            entry_price = float(decision['entry_price'])
            exit_price = float(decision['exit_price'])
            quantity = int(decision['quantity'])
            
            # Get strategy name from proposal
            strategy_name = decision.get('proposal', {}).get('strategy_name', 'unknown')
            strategy_type = decision.get('proposal', {}).get('strategy_type', 'unknown')
            
            # Calculate P&L based on strategy type
            # For credit spreads: profit when exit_price < entry_price
            # For debit spreads: profit when exit_price > entry_price
            if 'credit' in strategy_type:
                pnl = (entry_price - exit_price) * quantity * 100
            else:  # debit spreads
                pnl = (exit_price - entry_price) * quantity * 100
            
            closed_at = datetime.fromisoformat(decision['closed_at'].replace('Z', '+00:00'))
            
            return cls(
                closed_at=closed_at,
                entry_price=entry_price,
                exit_price=exit_price,
                quantity=quantity,
                strategy_name=strategy_name,
                strategy_type=strategy_type,
                pnl=pnl
            )
        except (KeyError, ValueError, TypeError) as e:
            print(f"Warning: Failed to parse decision: {e}")
            return None


def load_closed_positions(decisions_dir: str = "predictions/decisions") -> List[ClosedPosition]:
    """Load all closed positions from decision JSON files."""
    positions = []
    
    if not os.path.exists(decisions_dir):
        print(f"Error: Decisions directory not found: {decisions_dir}")
        return positions
    
    # Find all decision JSON files
    json_files = sorted([
        f for f in os.listdir(decisions_dir)
        if f.startswith('decisions_') and f.endswith('.json')
    ])
    
    if not json_files:
        print(f"Warning: No decision files found in {decisions_dir}")
        return positions
    
    # Load and parse each file
    for filename in json_files:
        filepath = os.path.join(decisions_dir, filename)
        try:
            with open(filepath, 'r') as f:
                decisions = json.load(f)
                
            if not isinstance(decisions, list):
                print(f"Warning: {filename} does not contain a list")
                continue
            
            for decision in decisions:
                position = ClosedPosition.from_decision_dict(decision)
                if position:
                    positions.append(position)
                    
        except (json.JSONDecodeError, IOError) as e:
            print(f"Warning: Failed to read {filename}: {e}")
            continue
    
    # Sort by closed date
    positions.sort(key=lambda p: p.closed_at)
    
    return positions


def _map_strategy_name(strategy_name: str) -> str:
    """Map strategy name from decisions to config key.
    
    This handles cases where decision JSON has different names than config keys.
    """
    name_mapping = {
        "velocity_signal_momentum": "velocity_momentum",
        "credit_spread": "credit_spread",
    }
    return name_mapping.get(strategy_name, strategy_name)


def load_capital_allocations(config_path: str = "config/strategies/capital_allocations.json") -> Dict[str, float]:
    """Load capital allocations from config file.
    
    Returns:
        Dictionary mapping strategy_name to allocated_capital
    """
    if not os.path.exists(config_path):
        print(f"Warning: Capital allocation config not found: {config_path}")
        return {}
    
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        allocations = {}
        for strategy_name, strategy_config in config.get("strategies", {}).items():
            allocations[strategy_name] = float(strategy_config.get("allocated_capital", 0.0))
        return allocations
    except Exception as e:
        print(f"Warning: Failed to load capital allocations: {e}")
        return {}


def calculate_equity_curve(
    positions: List[ClosedPosition],
    initial_capital: float = 0.0
) -> tuple[List[datetime], List[float]]:
    """
    Calculate capital remaining curve from closed positions based on realized P&L.
    
    Starts with allocated capital and applies realized P&L from closed positions.
    Open positions do not affect the curve.
    
    The curve includes a starting point at the allocated capital before any trades,
    then shows capital remaining after each position closes.
    
    Returns:
        Tuple of (dates, capital_remaining_values)
    """
    if not positions:
        return [], []
    
    # Sort positions by closed date
    sorted_positions = sorted(positions, key=lambda p: p.closed_at)
    
    dates = []
    capital_values = []
    
    # Start with initial capital point (before any trades)
    if sorted_positions:
        # Use the earliest close date minus 1 day as the starting point
        start_date = sorted_positions[0].closed_at - timedelta(days=1)
        dates.append(start_date)
        capital_values.append(initial_capital)
    
    # Track capital as positions close
    capital_remaining = initial_capital
    for position in sorted_positions:
        # Apply realized P&L to capital
        capital_remaining += position.pnl
        dates.append(position.closed_at)
        capital_values.append(capital_remaining)
    
    return dates, capital_values


def fetch_spy_data(start_date: datetime, end_date: datetime) -> Optional[pd.DataFrame]:
    """
    Fetch SPY price data for the given date range.
    
    Args:
        start_date: Start date for SPY data
        end_date: End date for SPY data
        
    Returns:
        DataFrame with SPY data indexed by date, or None if fetch fails
    """
    try:
        print(f"📊 Fetching SPY data from {start_date.date()} to {end_date.date()}...")
        spy = yf.Ticker("SPY")
        spy_data = spy.history(start=start_date, end=end_date + timedelta(days=1))
        
        if spy_data.empty:
            print("⚠️  No SPY data available for the date range")
            return None
        
        print(f"✅ Fetched {len(spy_data)} days of SPY data")
        return spy_data
    except Exception as e:
        print(f"❌ Failed to fetch SPY data: {e}")
        return None


def fetch_treasury_rates(start_date: datetime, end_date: datetime) -> Optional[pd.DataFrame]:
    """
    Fetch treasury rates data for the given date range.
    
    Args:
        start_date: Start date for treasury rates
        end_date: End date for treasury rates
        
    Returns:
        DataFrame with treasury rates indexed by date, or None if fetch fails
    """
    try:
        print(f"📈 Fetching treasury rates from {start_date.date()} to {end_date.date()}...")
        
        # Use TNX (10-year treasury) as the interest rate indicator
        tnx = yf.Ticker("^TNX")
        tnx_data = tnx.history(start=start_date, end=end_date + timedelta(days=1))
        
        if tnx_data.empty:
            print("⚠️  No treasury rate data available for the date range")
            return None
        
        # Convert to percentage (TNX is already in percentage points)
        rates_df = pd.DataFrame({
            '10Y_Rate': tnx_data['Close']
        }, index=tnx_data.index)
        
        print(f"✅ Fetched {len(rates_df)} days of treasury rate data")
        return rates_df
    except Exception as e:
        print(f"❌ Failed to fetch treasury rate data: {e}")
        return None


def plot_equity_curve(
    positions: List[ClosedPosition],
    strategy_filter: Optional[str] = None,
    output_file: Optional[str] = None,
    display: bool = True,
    capital_allocations: Optional[Dict[str, float]] = None,
    overlay_spy: bool = False,
    overlay_rates: bool = False
):
    """
    Plot equity curve(s) from closed positions.
    
    Args:
        positions: List of closed positions
        strategy_filter: If provided, only plot this strategy
        output_file: If provided, save plot to this file
        display: Whether to display the plot interactively
        capital_allocations: Dictionary mapping strategy names to allocated capital
        overlay_spy: Whether to overlay SPY price on the plot
        overlay_rates: Whether to overlay treasury interest rates on the plot
    """
    if not positions:
        print("No closed positions to plot")
        return
    
    # Group positions by strategy if no filter
    if strategy_filter:
        filtered_positions = [p for p in positions if p.strategy_name == strategy_filter]
        if not filtered_positions:
            print(f"No positions found for strategy: {strategy_filter}")
            return
        strategy_groups = {strategy_filter: filtered_positions}
    else:
        strategy_groups = {}
        for position in positions:
            if position.strategy_name not in strategy_groups:
                strategy_groups[position.strategy_name] = []
            strategy_groups[position.strategy_name].append(position)
    
    # Get date range from all positions
    all_dates = [p.closed_at for p in positions]
    min_date = min(all_dates) - timedelta(days=1)  # For equity curve start point
    max_date = max(all_dates)
    
    # For overlays, only show data from first trade onwards (not the start point)
    overlay_start_date = min(all_dates)
    
    # Fetch overlay data if requested
    spy_data = None
    rates_data = None
    
    if overlay_spy:
        spy_data = fetch_spy_data(overlay_start_date, max_date)
    
    if overlay_rates:
        rates_data = fetch_treasury_rates(overlay_start_date, max_date)

    strategy_frames: list[pd.DataFrame] = []
    reference_capital: float | None = None
    for strategy_name, strategy_positions in strategy_groups.items():
        config_key = _map_strategy_name(strategy_name)
        initial_capital = capital_allocations.get(config_key, 0.0) if capital_allocations else 0.0
        dates, capital = calculate_equity_curve(strategy_positions, initial_capital)
        if not dates:
            continue

        final_capital = capital[-1] if capital else initial_capital
        total_pnl = final_capital - initial_capital
        num_positions = len(strategy_positions)
        wins = sum(1 for p in strategy_positions if p.pnl > 0)
        win_rate = (wins / num_positions * 100) if num_positions > 0 else 0
        column = (
            f"{strategy_name.replace('_', ' ').title()} | "
            f"Capital: ${final_capital:,.0f} | P&L: ${total_pnl:+,.0f} | "
            f"Trades: {num_positions} | Win Rate: {win_rate:.1f}%"
        )
        strategy_frames.append(
            pd.DataFrame({"timestamp": pd.to_datetime(dates), column: capital})
        )
        if reference_capital is None and initial_capital > 0:
            reference_capital = initial_capital

    if not strategy_frames:
        print("No equity data to plot")
        return

    merged = reduce(
        lambda left, right: pd.merge(left, right, on="timestamp", how="outer"),
        strategy_frames,
    )
    merged = merged.sort_values("timestamp")
    if reference_capital is not None:
        merged["Initial Capital"] = reference_capital

    right_axis: list[str] = []
    if overlay_spy and spy_data is not None:
        spy_frame = pd.DataFrame(
            {"timestamp": pd.to_datetime(spy_data.index), "SPY Price ($)": spy_data["Close"].values}
        )
        merged = merged.merge(spy_frame, on="timestamp", how="outer")
        merged = merged.sort_values("timestamp")
        right_axis.append("SPY Price ($)")

    if overlay_rates and rates_data is not None:
        rates_frame = pd.DataFrame(
            {
                "timestamp": pd.to_datetime(rates_data.index),
                "10Y Treasury Rate (%)": rates_data["10Y_Rate"].values,
            }
        )
        merged = merged.merge(rates_frame, on="timestamp", how="outer")
        merged = merged.sort_values("timestamp")
        right_axis.append("10Y Treasury Rate (%)")

    if strategy_filter:
        title = f'Equity Curve - {strategy_filter.replace("_", " ").title()}'
    else:
        title = "Equity Curves - All Strategies"
    if overlay_spy or overlay_rates:
        overlays = []
        if overlay_spy:
            overlays.append("SPY")
        if overlay_rates:
            overlays.append("Interest Rates")
        title += f' (with {" & ".join(overlays)})'

    spec = build_plot_spec(
        merged,
        name="equity_curves",
        title=title,
        y_label="Capital Remaining ($)",
        right_axis=right_axis,
    )
    plot_config = PlotConfig(enabled=True, show=display)
    show_plot(spec, config=plot_config, save_path=output_file)
    if output_file:
        print(f"✅ Plot saved to: {output_file}")


def calculate_drawdowns(capital_values: List[float]) -> List[float]:
    """
    Calculate drawdown percentages from equity curve.
    
    A drawdown is the percentage decline from a peak value.
    Returns a list of all drawdown values (as percentages).
    
    Args:
        capital_values: List of capital values over time
        
    Returns:
        List of drawdown percentages (positive values representing declines)
    """
    if not capital_values or len(capital_values) < 2:
        return []
    
    drawdowns = []
    peak = capital_values[0]
    
    for current in capital_values:
        if current > peak:
            peak = current
        elif peak > 0:
            # Calculate drawdown as percentage decline from peak
            drawdown = (peak - current) / peak * 100
            if drawdown > 0:
                drawdowns.append(drawdown)
    
    return drawdowns


def print_summary(positions: List[ClosedPosition], strategy_filter: Optional[str] = None, capital_allocations: Optional[Dict[str, float]] = None):
    """Print summary statistics for closed positions."""
    if not positions:
        print("\n📊 No closed positions found")
        return
    
    # Filter if requested
    if strategy_filter:
        positions = [p for p in positions if p.strategy_name == strategy_filter]
        if not positions:
            print(f"\n📊 No positions found for strategy: {strategy_filter}")
            return
    
    print(f"\n{'='*80}")
    print(f"TRADING SUMMARY")
    print(f"{'='*80}")
    
    # Group by strategy
    strategy_groups = {}
    for position in positions:
        if position.strategy_name not in strategy_groups:
            strategy_groups[position.strategy_name] = []
        strategy_groups[position.strategy_name].append(position)
    
    # Calculate overall equity curve for drawdown analysis
    overall_initial_capital = 0.0
    if capital_allocations:
        # Sum all allocated capital for overall analysis
        overall_initial_capital = sum(capital_allocations.values())
    
    # Calculate overall equity curve
    sorted_all_positions = sorted(positions, key=lambda p: p.closed_at)
    _, overall_capital = calculate_equity_curve(sorted_all_positions, overall_initial_capital)
    overall_drawdowns = calculate_drawdowns(overall_capital)
    
    # Overall stats
    total_pnl = sum(p.pnl for p in positions)
    total_trades = len(positions)
    total_wins = sum(1 for p in positions if p.pnl > 0)
    total_losses = sum(1 for p in positions if p.pnl < 0)
    win_rate = (total_wins / total_trades * 100) if total_trades > 0 else 0
    
    print(f"\nOverall Performance:")
    print(f"  Total P&L: ${total_pnl:,.2f}")
    print(f"  Total Trades: {total_trades}")
    print(f"  Wins: {total_wins} | Losses: {total_losses}")
    print(f"  Win Rate: {win_rate:.1f}%")
    print(f"  Avg P&L per Trade: ${total_pnl / total_trades:,.2f}" if total_trades > 0 else "")
    
    # Overall drawdown stats
    if overall_drawdowns:
        min_dd = min(overall_drawdowns)
        mean_dd = sum(overall_drawdowns) / len(overall_drawdowns)
        max_dd = max(overall_drawdowns)
        print(f"  Drawdowns: Min: {min_dd:.2f}% | Mean: {mean_dd:.2f}% | Max: {max_dd:.2f}%")
    else:
        print(f"  Drawdowns: No drawdowns detected")
    
    # Capital tracking summary if allocations available
    if capital_allocations:
        print(f"\nCapital Tracking:")
        for strategy_name in set(p.strategy_name for p in positions):
            # Map strategy name to config key
            config_key = _map_strategy_name(strategy_name)
            if config_key in capital_allocations:
                allocated = capital_allocations[config_key]
                strategy_positions = [p for p in positions if p.strategy_name == strategy_name]
                strategy_pnl = sum(p.pnl for p in strategy_positions)
                remaining = allocated + strategy_pnl
                print(f"  {strategy_name.replace('_', ' ').title()}:")
                print(f"    Allocated: ${allocated:,.2f}")
                print(f"    Remaining: ${remaining:,.2f}")
                print(f"    P&L: ${strategy_pnl:+,.2f}")
    
    # Per-strategy stats
    print(f"\nBy Strategy:")
    for strategy_name in sorted(strategy_groups.keys()):
        strategy_positions = strategy_groups[strategy_name]
        strategy_pnl = sum(p.pnl for p in strategy_positions)
        strategy_trades = len(strategy_positions)
        strategy_wins = sum(1 for p in strategy_positions if p.pnl > 0)
        strategy_win_rate = (strategy_wins / strategy_trades * 100) if strategy_trades > 0 else 0
        
        # Map strategy name to config key
        config_key = _map_strategy_name(strategy_name)
        allocated = capital_allocations.get(config_key, 0.0) if capital_allocations else 0.0
        
        # Calculate strategy equity curve for drawdown analysis
        sorted_strategy_positions = sorted(strategy_positions, key=lambda p: p.closed_at)
        _, strategy_capital = calculate_equity_curve(sorted_strategy_positions, allocated)
        strategy_drawdowns = calculate_drawdowns(strategy_capital)
        
        remaining = allocated + strategy_pnl
        
        print(f"\n  {strategy_name.replace('_', ' ').title()}:")
        if allocated > 0:
            print(f"    Allocated Capital: ${allocated:,.2f}")
            print(f"    Remaining Capital: ${remaining:,.2f}")
        print(f"    P&L: ${strategy_pnl:+,.2f}")
        print(f"    Trades: {strategy_trades}")
        print(f"    Win Rate: {strategy_win_rate:.1f}%")
        print(f"    Avg P&L: ${strategy_pnl / strategy_trades:,.2f}" if strategy_trades > 0 else "")
        
        # Strategy drawdown stats
        if strategy_drawdowns:
            min_dd = min(strategy_drawdowns)
            mean_dd = sum(strategy_drawdowns) / len(strategy_drawdowns)
            max_dd = max(strategy_drawdowns)
            print(f"    Drawdowns: Min: {min_dd:.2f}% | Mean: {mean_dd:.2f}% | Max: {max_dd:.2f}%")
        else:
            print(f"    Drawdowns: No drawdowns detected")
    
    print(f"\n{'='*80}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Plot equity curves from closed trading positions",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Plot all strategies
  python -m src.prediction.plot_equity_curve
  
  # Plot specific strategy
  python -m src.prediction.plot_equity_curve --strategy velocity_signal_momentum
  
  # Save to file without displaying
  python -m src.prediction.plot_equity_curve --output equity.png --no-show
  
  # Print summary only (no plot)
  python -m src.prediction.plot_equity_curve --summary-only
  
  # Overlay SPY price on the equity curve
  python -m src.prediction.plot_equity_curve --overlay-spy
  
  # Overlay treasury interest rates on the equity curve
  python -m src.prediction.plot_equity_curve --overlay-rates
  
  # Overlay both SPY and interest rates
  python -m src.prediction.plot_equity_curve --overlay-spy --overlay-rates
        """
    )
    
    parser.add_argument(
        '--strategy',
        type=str,
        help='Filter by strategy name (e.g., velocity_signal_momentum, upward_trend_reversal)'
    )
    
    parser.add_argument(
        '--output', '-o',
        type=str,
        help='Output file path for saving the plot (e.g., equity_curve.png)'
    )
    
    parser.add_argument(
        '--no-show',
        action='store_true',
        help='Do not display the plot interactively (only save to file)'
    )
    
    parser.add_argument(
        '--summary-only',
        action='store_true',
        help='Print summary statistics only, do not generate plot'
    )
    
    parser.add_argument(
        '--decisions-dir',
        type=str,
        default='predictions/decisions',
        help='Directory containing decision JSON files (default: predictions/decisions)'
    )
    
    parser.add_argument(
        '--overlay-spy',
        action='store_true',
        help='Overlay SPY price on the equity curve plot'
    )
    
    parser.add_argument(
        '--overlay-rates',
        action='store_true',
        help='Overlay treasury interest rates (10Y) on the equity curve plot'
    )
    
    args = parser.parse_args()
    
    # Load positions
    print(f"📂 Loading closed positions from {args.decisions_dir}...")
    positions = load_closed_positions(args.decisions_dir)
    
    if not positions:
        print("❌ No closed positions found")
        return 1
    
    print(f"✅ Loaded {len(positions)} closed positions")
    
    # Load capital allocations
    capital_allocations = load_capital_allocations()
    
    # Print summary
    print_summary(positions, args.strategy, capital_allocations)
    
    # Plot unless summary-only
    if not args.summary_only:
        print("\n📈 Generating equity curve plot...")
        plot_equity_curve(
            positions,
            strategy_filter=args.strategy,
            output_file=args.output,
            display=not args.no_show,
            capital_allocations=capital_allocations,
            overlay_spy=args.overlay_spy,
            overlay_rates=args.overlay_rates
        )
    
    return 0


if __name__ == "__main__":
    exit(main())

