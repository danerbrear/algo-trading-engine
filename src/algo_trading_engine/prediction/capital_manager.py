"""
Capital allocation and risk management for strategies.

This module provides capital tracking and risk-based position sizing
for multiple strategies with independent capital allocations.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple
from pathlib import Path
import json

from algo_trading_engine.backtest.models import StrategyType
from algo_trading_engine.prediction.decision_store import JsonDecisionStore, DecisionResponseDTO


class CapitalManager:
    """Manages capital allocation and risk checking for strategies.
    
    Tracks capital remaining per strategy by calculating from decisions JSON files,
    ensuring a single source of truth.
    """
    
    def __init__(self, allocations_config: Dict, decision_store: JsonDecisionStore):
        """Initialize capital manager with allocation configuration.
        
        Args:
            allocations_config: Dictionary with structure:
                {
                    "strategies": {
                        "strategy_name": {
                            "allocated_capital": float,
                            "max_risk_percentage": float
                        }
                    }
                }
            decision_store: JsonDecisionStore instance for reading decisions
        """
        self.allocations = allocations_config.get("strategies", {})
        self.decision_store = decision_store
    
    @classmethod
    def from_config_file(cls, config_path: str, decision_store: JsonDecisionStore) -> CapitalManager:
        """Load capital allocations from a JSON config file.
        
        Args:
            config_path: Path to capital_allocations.json file
            decision_store: JsonDecisionStore instance
            
        Returns:
            CapitalManager instance
            
        Raises:
            FileNotFoundError: If config file doesn't exist
            ValueError: If config is invalid
        """
        path = Path(config_path)
        if not path.exists():
            raise FileNotFoundError(f"Capital allocation config not found: {config_path}")
        
        with open(path, 'r') as f:
            config = json.load(f)
        
        return cls(config, decision_store)
    
    def get_allocated_capital(self, strategy_name: str) -> float:
        """Get the allocated capital for a strategy.
        
        Args:
            strategy_name: Name of the strategy
            
        Returns:
            Allocated capital amount
        """
        config = self.allocations.get(strategy_name, {})
        return float(config.get("allocated_capital", 0.0))
    
    def get_remaining_capital(self, strategy_name: str) -> float:
        """Get the remaining capital for a strategy.
        
        Calculated from decisions JSON files - single source of truth.
        
        Args:
            strategy_name: Name of the strategy
            
        Returns:
            Remaining capital amount
        """
        return self._calculate_remaining_capital(strategy_name)
    
    def get_max_risk_percentage(self, strategy_name: str) -> float:
        """Get the maximum risk percentage for a strategy.
        
        Args:
            strategy_name: Name of the strategy
            
        Returns:
            Max risk percentage (e.g., 0.05 for 5%)
        """
        config = self.allocations.get(strategy_name, {})
        return float(config.get("max_risk_percentage", 0.0))
    
    def get_max_allowed_risk(self, strategy_name: str) -> float:
        """Get the maximum allowed risk per position for a strategy.
        
        Calculated as: remaining_capital * max_risk_percentage
        
        This ensures risk limits account for capital already allocated to open positions.
        
        Args:
            strategy_name: Name of the strategy
            
        Returns:
            Maximum allowed risk per position
        """
        remaining = self.get_remaining_capital(strategy_name)
        max_risk_pct = self.get_max_risk_percentage(strategy_name)
        return remaining * max_risk_pct
    
    def check_risk_threshold(self, strategy_name: str, max_risk: float) -> Tuple[bool, str]:
        """Check if a position's max risk is within the threshold.
        
        Args:
            strategy_name: Name of the strategy
            max_risk: Maximum risk for the position
            
        Returns:
            Tuple of (is_allowed, reason_message)
        """
        if strategy_name not in self.allocations:
            return False, f"Strategy '{strategy_name}' not found in capital allocations"
        
        max_allowed = self.get_max_allowed_risk(strategy_name)
        remaining = self.get_remaining_capital(strategy_name)
        
        if max_risk > max_allowed:
            remaining = self.get_remaining_capital(strategy_name)
            risk_pct = self.get_max_risk_percentage(strategy_name) * 100
            return False, (
                f"${max_risk:.2f} exceeds maximum allowed risk of "
                f"${max_allowed:.2f} ({risk_pct:.1f}% of ${remaining:,.2f} remaining capital)"
            )
        
        if max_risk > remaining:
            return False, (
                f"${max_risk:.2f} exceeds remaining capital of "
                f"${remaining:.2f}"
            )
        
        return True, "Risk check passed"
    
    def is_credit_strategy(self, strategy_type: StrategyType) -> bool:
        """Determine if a strategy type is a credit strategy.
        
        Credit strategies receive premium when opening.
        
        Args:
            strategy_type: The strategy type to check
            
        Returns:
            True if credit strategy, False if debit strategy
        """
        credit_strategies = {
            StrategyType.PUT_CREDIT_SPREAD,
            StrategyType.CALL_CREDIT_SPREAD,
            StrategyType.SHORT_CALL,
            StrategyType.SHORT_PUT,
        }
        return strategy_type in credit_strategies
    
    def _calculate_remaining_capital(self, strategy_name: str) -> float:
        """Calculate remaining capital from decisions JSON files.
        
        Calculation method:
        1. Start with allocated capital from config
        2. Load all accepted decisions for the strategy
        3. For each decision:
           - Opening credit positions: Add entry_price (premium received)
           - Opening debit positions: Subtract entry_price (premium paid)
           - Closing credit positions: Subtract exit_price (premium paid to close)
           - Closing debit positions: Add exit_price (premium received when closing)
        
        Args:
            strategy_name: Name of the strategy
            
        Returns:
            Remaining capital amount
        """
        allocated = self.get_allocated_capital(strategy_name)
        
        # Get all accepted decisions for this strategy
        decisions = self.decision_store.get_all_decisions(strategy_name=strategy_name)
        
        capital = allocated
        
        for decision in decisions:
            if decision.entry_price is None:
                continue
                
            strategy_type = decision.proposal.strategy_type
            is_credit = self.is_credit_strategy(strategy_type)
            entry_price = float(decision.entry_price) * (decision.quantity or 1) * 100
            
            if is_credit:
                # Credit strategy: received premium when opening
                capital += entry_price
            else:
                # Debit strategy: paid premium when opening
                capital -= entry_price
            
            # If position is closed, adjust for closing premium
            if decision.closed_at is not None and decision.exit_price is not None:
                exit_price = float(decision.exit_price) * (decision.quantity or 1) * 100
                
                if is_credit:
                    # Credit strategy: paid premium to close
                    capital -= exit_price
                else:
                    # Debit strategy: received premium when closing
                    capital += exit_price
        
        return capital
    
    def get_status_summary(self, strategy_name: str) -> str:
        """Get a formatted status summary for a strategy.
        
        Args:
            strategy_name: Name of the strategy
            
        Returns:
            Formatted status string
        """
        allocated = self.get_allocated_capital(strategy_name)
        remaining = self.get_remaining_capital(strategy_name)
        max_allowed = self.get_max_allowed_risk(strategy_name)
        risk_pct = self.get_max_risk_percentage(strategy_name) * 100
        
        return (
            f"📊 Capital Allocation: ${allocated:,.2f} for {strategy_name}\n"
            f"💰 Remaining Capital: ${remaining:,.2f}\n"
            f"📈 Max Risk Per Position: ${max_allowed:,.2f} ({risk_pct:.1f}% of allocated capital)"
        )

