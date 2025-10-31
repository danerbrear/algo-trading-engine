"""
Plot equity curves from closed trading positions.

This script reads decision JSON files and generates equity curves showing
cumulative P&L over time. Supports filtering by strategy name.

Usage:
    python -m src.prediction.plot_equity_curve                    # Plot all strategies
    python -m src.prediction.plot_equity_curve --strategy velocity_signal_momentum
    python -m src.prediction.plot_equity_curve --strategy upward_trend_reversal
    python -m src.prediction.plot_equity_curve --output equity_curve.png
"""

import argparse
import json
import os
from datetime import datetime
from typing import List, Dict, Optional
from dataclasses import dataclass
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path


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
        
        # Only process accepted decisions
        if decision.get('outcome') != 'accepted':
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


def calculate_equity_curve(
    positions: List[ClosedPosition],
    initial_capital: float = 0.0
) -> tuple[List[datetime], List[float]]:
    """
    Calculate cumulative equity curve from closed positions.
    
    Returns:
        Tuple of (dates, equity_values)
    """
    if not positions:
        return [], []
    
    dates = []
    equity_values = []
    cumulative_pnl = initial_capital
    
    for position in positions:
        cumulative_pnl += position.pnl
        dates.append(position.closed_at)
        equity_values.append(cumulative_pnl)
    
    return dates, equity_values


def plot_equity_curve(
    positions: List[ClosedPosition],
    strategy_filter: Optional[str] = None,
    output_file: Optional[str] = None,
    show_plot: bool = True
):
    """
    Plot equity curve(s) from closed positions.
    
    Args:
        positions: List of closed positions
        strategy_filter: If provided, only plot this strategy
        output_file: If provided, save plot to this file
        show_plot: Whether to display the plot interactively
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
    
    # Create figure
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Plot each strategy's equity curve
    colors = plt.cm.tab10(range(len(strategy_groups)))
    
    for (strategy_name, strategy_positions), color in zip(strategy_groups.items(), colors):
        dates, equity = calculate_equity_curve(strategy_positions)
        
        if not dates:
            continue
        
        # Calculate statistics
        final_pnl = equity[-1] if equity else 0
        num_positions = len(strategy_positions)
        wins = sum(1 for p in strategy_positions if p.pnl > 0)
        win_rate = (wins / num_positions * 100) if num_positions > 0 else 0
        avg_pnl = final_pnl / num_positions if num_positions > 0 else 0
        
        # Plot line
        label = (f"{strategy_name.replace('_', ' ').title()}\n"
                f"P&L: ${final_pnl:,.0f} | "
                f"Trades: {num_positions} | "
                f"Win Rate: {win_rate:.1f}%")
        
        ax.plot(dates, equity, marker='o', linestyle='-', linewidth=2,
                markersize=6, label=label, color=color, alpha=0.8)
        
        # Add zero line for reference
        ax.axhline(y=0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    
    # Format plot
    ax.set_xlabel('Date', fontsize=12, fontweight='bold')
    ax.set_ylabel('Cumulative P&L ($)', fontsize=12, fontweight='bold')
    
    if strategy_filter:
        title = f'Equity Curve - {strategy_filter.replace("_", " ").title()}'
    else:
        title = 'Equity Curves - All Strategies'
    
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    
    # Format x-axis dates
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    plt.xticks(rotation=45, ha='right')
    
    # Add grid
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # Legend
    ax.legend(loc='best', fontsize=10, framealpha=0.9)
    
    # Format y-axis with currency
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
    
    # Tight layout
    plt.tight_layout()
    
    # Save if requested
    if output_file:
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"✅ Plot saved to: {output_file}")
    
    # Show if requested
    if show_plot:
        plt.show()
    
    plt.close()


def print_summary(positions: List[ClosedPosition], strategy_filter: Optional[str] = None):
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
    
    # Per-strategy stats
    print(f"\nBy Strategy:")
    for strategy_name in sorted(strategy_groups.keys()):
        strategy_positions = strategy_groups[strategy_name]
        strategy_pnl = sum(p.pnl for p in strategy_positions)
        strategy_trades = len(strategy_positions)
        strategy_wins = sum(1 for p in strategy_positions if p.pnl > 0)
        strategy_win_rate = (strategy_wins / strategy_trades * 100) if strategy_trades > 0 else 0
        
        print(f"\n  {strategy_name.replace('_', ' ').title()}:")
        print(f"    P&L: ${strategy_pnl:,.2f}")
        print(f"    Trades: {strategy_trades}")
        print(f"    Win Rate: {strategy_win_rate:.1f}%")
        print(f"    Avg P&L: ${strategy_pnl / strategy_trades:,.2f}" if strategy_trades > 0 else "")
    
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
    
    args = parser.parse_args()
    
    # Load positions
    print(f"📂 Loading closed positions from {args.decisions_dir}...")
    positions = load_closed_positions(args.decisions_dir)
    
    if not positions:
        print("❌ No closed positions found")
        return 1
    
    print(f"✅ Loaded {len(positions)} closed positions")
    
    # Print summary
    print_summary(positions, args.strategy)
    
    # Plot unless summary-only
    if not args.summary_only:
        print("\n📈 Generating equity curve plot...")
        plot_equity_curve(
            positions,
            strategy_filter=args.strategy,
            output_file=args.output,
            show_plot=not args.no_show
        )
    
    return 0


if __name__ == "__main__":
    exit(main())

