from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional, List

from algo_trading_engine.core.strategy import Strategy
from algo_trading_engine.backtest.models import Position, StrategyType
from algo_trading_engine.prediction.decision_store import (
    JsonDecisionStore,
    ProposedPositionRequestDTO,
    DecisionResponseDTO,
    generate_decision_id,
)
from algo_trading_engine.prediction.capital_manager import CapitalManager

class InteractiveStrategyRecommender:
    """Produce open/close recommendations and capture user decisions.

    Expects a strategy compatible with credit spread workflows (e.g.,
    CreditSpreadStrategy), with `.data`, `.options_data`, and helpers used
    in the existing backtest (`_find_best_spread`, `_ensure_volume_data`,
    `get_current_volumes_for_position`).
    """

    def __init__(self, strategy: Strategy, decision_store: JsonDecisionStore, capital_manager: CapitalManager, auto_yes: bool = False):
        self.strategy = strategy
        self.decision_store = decision_store
        self.capital_manager = capital_manager
        self.auto_yes = auto_yes

    def run(self, date: datetime, auto_yes: Optional[bool] = None) -> None:
        """Run recommendation flow by calling on_new_date once to handle both opens and closes.
        
        This ensures indicators are only updated once per bar, avoiding duplicate updates
        that would occur if we called recommend_open_position and recommend_close_positions
        separately (as they both call on_new_date internally).
        """
        if auto_yes is not None:
            self.auto_yes = auto_yes
        
        # Validate strategy data
        if getattr(self.strategy, "data", None) is None:
            print("No data found for strategy")
            return
        
        # Get current price (needed for both opens and closes)
        current_price = self._get_current_price(date)
        if current_price is None:
            return
        
        # Display recent prices for velocity strategies
        if hasattr(self.strategy, '__class__') and 'velocity' in self.strategy.__class__.__name__.lower():
            self._display_recent_underlying_prices(date)
        
        # Get open positions for closing logic
        open_records = self.decision_store.get_open_positions()
        strategy_positions = [self._position_from_decision(rec) for rec in open_records]
        
        # Capture both opens and closes in a single on_new_date call
        recommended_position = None
        positions_to_close = []
        
        def capture_add_position(position: Position):
            """Capture the position created by the strategy's on_new_date logic"""
            nonlocal recommended_position
            recommended_position = position
        
        def capture_remove_position(date_arg: datetime, position: Position, exit_price: float, 
                                   underlying_price: float = None, current_volumes: list = None):
            """Capture the position closure decision from the strategy's on_new_date logic"""
            positions_to_close.append({
                "position": position,
                "exit_price": exit_price,
                "underlying_price": underlying_price,
                "current_volumes": current_volumes,
                "rationale": "strategy_decision"
            })
        
        # Call on_new_date ONCE with both callbacks - indicators are updated once here
        try:
            self.strategy.on_new_date(date, tuple(strategy_positions), capture_add_position, capture_remove_position)
        except Exception as e:
            print(f"Error in strategy on_new_date: {e}")
            raise e
        
        # Process captured open position (if any)
        if recommended_position is not None:
            self._process_open_recommendation(date, recommended_position, current_price)
        else:
            print("No recommendation found for strategy")
        
        # Process captured position closures (if any)
        if positions_to_close:
            self._process_close_recommendations(date, positions_to_close, open_records)

    # ---------- Helper methods for unified run() ----------
    
    def _get_current_price(self, date: datetime) -> Optional[float]:
        """Determine current price for the given date."""
        current_price = None
        
        # Check if the specified date is the current date
        current_date = datetime.now().date()
        if date.date() == current_date:
            # Use live price via strategy's injected method
            print(f"📡 Fetching live price for {date.date()} (current date)")
            # Get symbol from strategy
            symbol = None
            if hasattr(self.strategy, 'symbol'):
                symbol = self.strategy.symbol
            elif hasattr(self.strategy, 'data') and self.strategy.data is not None:
                symbol = self.strategy.data.index.name if self.strategy.data.index.name else 'SPY'
            
            if symbol:
                try:
                    current_price = self.strategy.get_current_underlying_price(date, symbol)
                except Exception as e:
                    print(f"⚠️ Could not fetch live price via strategy method: {e}")
                    current_price = None
            
            if current_price is None:
                print("⚠️ Could not fetch live price, falling back to cached data")
        
        # Fallback to cached data if live price failed or date is not current
        if current_price is None:
            try:
                current_price = float(self.strategy.data.loc[date]["Close"])
            except Exception:
                # try normalize date
                try:
                    current_price = float(self.strategy.data.loc[date.date()]["Close"])  # type: ignore[index]
                except Exception:
                    print(f"Could not find current price for date {date.date()}")
                    if self.strategy.data is not None and len(self.strategy.data) > 0:
                        print(f"Available data range: {self.strategy.data.index[0].date()} to {self.strategy.data.index[-1].date()}")
                        print(f"Total data points: {len(self.strategy.data)}")
                    else:
                        print("Strategy data is empty or None")
                    return None
        
        return current_price
    
    def _process_open_recommendation(self, date: datetime, position: Position, current_price: float) -> Optional[DecisionResponseDTO]:
        """Process a captured open position recommendation."""
        # Extract the recommendation details from the created position
        if not position.spread_options or len(position.spread_options) != 2:
            return None
        
        atm_option, otm_option = position.spread_options
        width = abs(atm_option.strike - otm_option.strike)
        
        # Build recommendation dict
        recommendation = {
            "strategy_type": position.strategy_type,
            "legs": (atm_option, otm_option),
            "credit": position.entry_price,
            "width": width,
            "probability_of_profit": 0.7,  # Default confidence for rule-based strategies
            "confidence": 0.7,  # Default confidence for rule-based strategies
            "expiration_date": position.expiration_date.strftime("%Y-%m-%d"),
            "risk_reward": 0.5,  # Default
        }
        
        # Get strategy name from class and map to config key
        strategy_name = self._get_strategy_name_from_class()
        
        # Build proposal DTO
        proposal = ProposedPositionRequestDTO(
            symbol=self.strategy.symbol if hasattr(self.strategy, 'symbol') else 'SPY',
            strategy_type=position.strategy_type,
            legs=(atm_option, otm_option),
            credit=float(position.entry_price),
            width=float(width),
            probability_of_profit=0.7,
            confidence=0.7,
            expiration_date=position.expiration_date.strftime("%Y-%m-%d"),
            created_at=datetime.now(timezone.utc).isoformat(),
            strategy_name=strategy_name,
        )
        
        # Calculate max risk
        max_risk = self._calculate_max_risk(position.strategy_type, float(width), float(position.entry_price))
        
        # Check risk threshold
        is_allowed, risk_message = self.capital_manager.check_risk_threshold(strategy_name, max_risk)
        
        if not is_allowed:
            print(f"❌ Risk check failed: {risk_message}")
            print("Position rejected due to risk threshold.")
            return None
        
        # Determine if credit or debit strategy
        is_credit = self.capital_manager.is_credit_strategy(position.strategy_type)
        premium_amount = float(position.entry_price) * 100  # Convert to dollars per contract
        premium_label = "Premium received" if is_credit else "Premium paid"
        
        # Prompt user
        summary = self._format_open_summary(proposal, recommendation, max_risk, premium_amount, premium_label, risk_message)
        if not self.prompt(f"Open recommendation:\n{summary}\nOpen this position?"):
            return None
        
        decided_at = datetime.now(timezone.utc).isoformat()
        record = DecisionResponseDTO(
            id=generate_decision_id(proposal, decided_at),
            proposal=proposal,
            outcome="accepted",
            decided_at=decided_at,
            rationale=f"strategy_confidence={proposal.confidence:.2f}",
            quantity=1,
            entry_price=proposal.credit,
        )
        self.decision_store.append_decision(record)
        return record
    
    def _process_close_recommendations(self, date: datetime, close_recommendations: List[dict], open_records: List[DecisionResponseDTO]) -> List[DecisionResponseDTO]:
        """Process captured position closure recommendations."""
        closed_records: List[DecisionResponseDTO] = []
        
        for recommendation in close_recommendations:
            position = recommendation["position"]
            exit_price = recommendation["exit_price"]
            
            # If auto_yes is false, use _get_exit_price_from_user_prompts to get exit price from prompts
            if not self.auto_yes:
                computed_exit_price = self._get_exit_price_from_user_prompts(position, date)
                if computed_exit_price is not None:
                    exit_price = computed_exit_price
                else:
                    print(f"⚠️  Could not compute exit price for position {position.__str__()}, using strategy-provided price")
            
            # Find the corresponding decision record using Position equality
            for rec in open_records:
                rec_position = self._position_from_decision(rec)
                if rec_position == position:
                    # Mark closed in the store
                    self.decision_store.mark_closed(rec.id, exit_price=exit_price, closed_at=date)
                    # Return an updated record instance for the caller
                    updated = DecisionResponseDTO(
                        id=rec.id,
                        proposal=rec.proposal,
                        outcome=rec.outcome,
                        decided_at=rec.decided_at,
                        rationale=recommendation.get("rationale", "strategy_decision"),
                        quantity=rec.quantity,
                        entry_price=rec.entry_price,
                        exit_price=exit_price,
                        closed_at=date.isoformat(),
                    )
                    closed_records.append(updated)
                    break
        
        return closed_records

    def get_open_positions_status(self, date: datetime) -> List[dict]:
        """Return current stats for all open positions without prompting.

        Stats include exit price, P&L dollars and percent, days held, and DTE.
        """
        statuses: List[dict] = []
        for rec in self.decision_store.get_open_positions():
            position = self._position_from_decision(rec)
            # For status display, always use auto mode (no prompts)
            exit_price = self._get_exit_price_for_status(position, date)
            if exit_price is None:
                continue
            try:
                pnl_dollars = position.get_return_dollars(exit_price)
                # Percentage based on credit spread convention
                pnl_pct = ((position.entry_price * position.quantity * 100) - (exit_price * position.quantity * 100)) / (position.entry_price * position.quantity * 100)
            except Exception:
                pnl_dollars = None
                pnl_pct = None
            try:
                days_held = position.get_days_held(date)
                dte = position.get_days_to_expiration(date)
            except Exception:
                days_held = None
                dte = None

            statuses.append({
                "id": rec.id,
                "symbol": rec.proposal.symbol,
                "strategy_type": rec.proposal.strategy_type.value,
                "legs": tuple((leg.option_type.value, leg.strike, leg.expiration) for leg in rec.proposal.legs),
                "quantity": position.quantity,
                "entry_price": position.entry_price,
                "exit_price": exit_price,
                "pnl_dollars": pnl_dollars,
                "pnl_percent": pnl_pct,
                "days_held": days_held,
                "dte": dte,
            })
        return statuses

    # ---------- Utilities ----------

    def prompt(self, message: str) -> bool:
        if self.auto_yes:
            return True
        answer = input(f"{message} [y/N]: ").strip().lower()
        return answer in {"y", "yes"}

    def _position_from_decision(self, rec: DecisionResponseDTO) -> Position:
        # Determine representative strike for display based on strategy type
        legs = list(rec.proposal.legs)
        strike = legs[0].strike if legs else 0.0
        if rec.proposal.strategy_type == StrategyType.CALL_CREDIT_SPREAD:
            # short is lower strike
            strike = min(leg.strike for leg in legs)
        elif rec.proposal.strategy_type == StrategyType.PUT_CREDIT_SPREAD:
            # short is higher strike
            strike = max(leg.strike for leg in legs)

        position = Position(
            symbol=rec.proposal.symbol,
            expiration_date=datetime.strptime(rec.proposal.expiration_date, "%Y-%m-%d"),
            strategy_type=rec.proposal.strategy_type,
            strike_price=strike,
            entry_date=datetime.fromisoformat(rec.decided_at),
            entry_price=float(rec.entry_price if rec.entry_price is not None else rec.proposal.credit),
            spread_options=list(rec.proposal.legs),
        )
        position.set_quantity(int(rec.quantity) if rec.quantity is not None else 1)
        return position

    def _format_open_summary(self, proposal: ProposedPositionRequestDTO, best: dict, max_risk: float, premium_amount: float, premium_label: str, risk_message: str) -> str:
        legs_str = ", ".join(
            [f"{leg.option_type.value.upper()} {int(leg.strike)} exp {leg.expiration}" for leg in proposal.legs]
        )
        rr = best.get("risk_reward")
        return (
            f"Symbol: {proposal.symbol}\n"
            f"Strategy: {proposal.strategy_type.value}\n"
            f"Legs: {legs_str}\n"
            f"Credit: ${proposal.credit:.2f}  Width: {proposal.width}  R/R: {rr}  Prob: {proposal.probability_of_profit:.0%}\n"
            f"Max Risk: ${max_risk:.2f}\n"
            f"{premium_label}: ${premium_amount:.2f} ({'credit' if 'received' in premium_label else 'debit'} strategy)\n"
            f"✅ {risk_message}"
        )
    
    def _get_strategy_name_from_class(self) -> str:
        """Get strategy name from class and map to config key."""
        class_name = self.strategy.__class__.__name__.replace("Strategy", "")
        # Convert from CamelCase to snake_case
        import re
        strategy_name = re.sub(r'(?<!^)(?=[A-Z])', '_', class_name).lower()
        
        # Map class names to config keys
        name_mapping = {
            "credit_spread": "credit_spread",
            "velocity_signal_momentum": "velocity_momentum",
        }
        
        return name_mapping.get(strategy_name, strategy_name)
    
    def _calculate_max_risk(self, strategy_type: StrategyType, width: float, credit: float) -> float:
        """Calculate max risk for a position.
        
        Args:
            strategy_type: The strategy type
            width: Spread width
            credit: Net credit/debit received/paid
            
        Returns:
            Max risk in dollars (for 1 contract)
        """
        if strategy_type in (StrategyType.PUT_CREDIT_SPREAD, StrategyType.CALL_CREDIT_SPREAD):
            # Credit spread: max risk = (width - credit) * 100
            return (width - credit) * 100
        elif strategy_type in (StrategyType.SHORT_CALL, StrategyType.SHORT_PUT):
            # Naked options: max risk is variable, use credit as placeholder
            # For now, estimate as strike difference if available
            return abs(credit) * 100  # Placeholder - actual calculation would need strikes
        else:
            # Debit spreads and long options: max risk is the debit paid
            return abs(credit) * 100

    def _format_close_summary(self, position: Position, exit_price: float, rationale: str) -> str:
        # Compute P&L
        try:
            pnl_dollars = position.get_return_dollars(exit_price)
            pnl_pct = ((position.entry_price * position.quantity * 100) - (exit_price * position.quantity * 100)) / (position.entry_price * position.quantity * 100)
        except Exception:
            pnl_dollars = None
            pnl_pct = None
        try:
            days_held = position.get_days_held(datetime.fromisoformat(position.entry_date.isoformat()))  # keep type happy
        except Exception:
            days_held = None
        return (
            f"{position.__str__()}\n"
            f"Exit price: ${exit_price:.2f}\n"
            f"P&L: ${pnl_dollars:.2f} ({pnl_pct:.1%})\n" if pnl_dollars is not None and pnl_pct is not None else ""
            f"Rationale: {rationale}"
        )

    def _get_exit_price_from_user_prompts(self, position: Position, date: datetime) -> Optional[float]:
        """Get exit price by prompting user for individual option prices."""
        try:
            if not position.spread_options or len(position.spread_options) != 2:
                print("⚠️  Position doesn't have valid spread options")
                return None
                
            atm_option, otm_option = position.spread_options
            
            # If the date is the current date, try to fetch previous day's close data
            # since end-of-day data may not be available yet (processed after 4:30 PM ET)
            fetch_date = date
            current_date = datetime.now().date()
            if date.date() == current_date:
                from datetime import timedelta
                fetch_date = date - timedelta(days=1)
                print(f"📅 Current date detected, fetching previous day's data: {fetch_date.strftime('%Y-%m-%d')}")
            
            # Get bar data for both options
            atm_bar = self.strategy.get_option_bar(atm_option, fetch_date)
            otm_bar = self.strategy.get_option_bar(otm_option, fetch_date)
            
            if not atm_bar or not otm_bar:
                print(f"⚠️  No bar data available for options on {fetch_date.strftime('%Y-%m-%d')}")
                return None
            
            # Interactive mode: prompt user for exit prices and calculate manually
            print(f"\n💬 Interactive mode: Please provide exit prices for position {position.__str__()}")
            
            # Prompt for ATM option exit price
            atm_price_input = input(f"Enter exit price for {atm_option.ticker} (current: ${atm_bar.close_price}): ").strip()
            if atm_price_input:
                atm_price = float(atm_price_input)
                print(f"✅ Using custom ATM price: ${atm_price}")
            else:
                atm_price = float(atm_bar.close_price)
                print(f"✅ Using current ATM price: ${atm_price}")
            
            # Prompt for OTM option exit price
            otm_price_input = input(f"Enter exit price for {otm_option.ticker} (current: ${otm_bar.close_price}): ").strip()
            if otm_price_input:
                otm_price = float(otm_price_input)
                print(f"✅ Using custom OTM price: ${otm_price}")
            else:
                otm_price = float(otm_bar.close_price)
                print(f"✅ Using current OTM price: ${otm_price}")
            
            # Calculate exit price manually based on strategy type
            if position.strategy_type.value == "put_credit_spread":
                # For put credit spread: sell ATM put, buy OTM put
                # Exit price = current ATM price - current OTM price
                exit_price = atm_price - otm_price
                print(f"💰 Calculated exit price: ${atm_price:.2f} - ${otm_price:.2f} = ${exit_price:.2f}")
            else:
                # This should not happen for credit spread strategies
                raise ValueError(f"Unsupported strategy type for manual exit price calculation: {position.strategy_type.value}")
            
            return exit_price
            
        except Exception as e:
            print(f"⚠️  Error calculating exit price: {e}")
            return None

    def _get_exit_price_for_status(self, position: Position, date: datetime) -> Optional[float]:
        """Get exit price for status display without prompting user."""
        try:
            if not position.spread_options or len(position.spread_options) != 2:
                return None
                
            atm_option, otm_option = position.spread_options
            
            # If the date is the current date, try to fetch previous day's close data
            fetch_date = date
            current_date = datetime.now().date()
            if date.date() == current_date:
                from datetime import timedelta
                fetch_date = date - timedelta(days=1)
            
            # Get bar data for both options
            atm_bar = self.strategy.get_option_bar(atm_option, fetch_date)
            otm_bar = self.strategy.get_option_bar(otm_option, fetch_date)
            
            if not atm_bar or not otm_bar:
                return None
            
            # Use the calculate_exit_price_from_bars method for status display
            exit_price = position.calculate_exit_price_from_bars(atm_bar, otm_bar)
            return exit_price
    
        except Exception:
            return None

    def _display_recent_underlying_prices(self, date: datetime) -> None:
        """Display the most recent 5 days of underlying price for velocity_momentum strategy."""
        try:
            if self.strategy.data is None or self.strategy.data.empty:
                print("⚠️  No underlying price data available")
                return
            
            # Get the most recent 5 days of data
            recent_data = self.strategy.data.tail(5)
            
            print("\n📊 Recent 5 Days of Underlying Price:")
            print("=" * 50)
            
            for idx, (date_idx, row) in enumerate(recent_data.iterrows()):
                # Format the date
                if hasattr(date_idx, 'date'):
                    date_str = date_idx.date().strftime('%Y-%m-%d')
                else:
                    date_str = str(date_idx)
                
                # Get the close price
                close_price = row.get('Close', 'N/A')
                if isinstance(close_price, (int, float)):
                    price_str = f"${close_price:.2f}"
                else:
                    price_str = str(close_price)
                
                # Add indicator for current date
                current_date = datetime.now().date()
                if hasattr(date_idx, 'date') and date_idx.date() == current_date:
                    date_str += " (TODAY)"
                elif idx == len(recent_data) - 1:
                    date_str += " (LATEST)"
                
                print(f"  {date_str}: {price_str}")
            
            print("=" * 50)
            
        except Exception as e:
            print(f"⚠️  Error displaying recent prices: {e}")
