from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional, List

from src.backtest.models import Position, StrategyType, Strategy
from src.prediction.decision_store import (
    JsonDecisionStore,
    ProposedPositionRequest,
    DecisionResponse,
    generate_decision_id,
)
from src.prediction.capital_manager import CapitalManager

class InteractiveStrategyRecommender:
    """Produce open/close recommendations and capture user decisions.

    Expects a strategy compatible with credit spread workflows (e.g.,
    CreditSpreadStrategy), with `.data`, `.options_data`, and helpers used
    in the existing backtest (`_find_best_spread`, `_ensure_volume_data`,
    `get_current_volumes_for_position`).
    """

    def __init__(self, strategy: Strategy, options_handler, decision_store: JsonDecisionStore, capital_manager: CapitalManager, auto_yes: bool = False):
        self.strategy = strategy
        self.options_handler = options_handler
        self.decision_store = decision_store
        self.capital_manager = capital_manager
        self.auto_yes = auto_yes

    def run(self, date: datetime, auto_yes: Optional[bool] = None) -> None:
        if auto_yes is not None:
            self.auto_yes = auto_yes
        self.recommend_open_position(date)
        self.recommend_close_positions(date)

    # ---------- Open recommendation ----------

    def recommend_open_position(self, date: datetime) -> Optional[DecisionResponse]:
        """Use the strategy to propose an opening trade for the date and capture decision."""
        # Require strategy data
        if getattr(self.strategy, "data", None) is None:
            print("No data found for strategy")
            return None

        # Determine current price
        current_price = None
        
        # Check if the specified date is the current date
        current_date = datetime.now().date()
        if date.date() == current_date:
            # Use live price from Polygon API
            print(f"📡 Fetching live price for {date.date()} (current date)")
            if hasattr(self.options_handler, 'symbol'):
                from src.common.data_retriever import DataRetriever
                temp_retriever = DataRetriever(symbol=self.options_handler.symbol, use_free_tier=True, quiet_mode=True)
                current_price = temp_retriever.get_live_price()
            
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

        # Display recent 5 days of underlying price for velocity_momentum strategy
        if hasattr(self.strategy, '__class__') and 'velocity' in self.strategy.__class__.__name__.lower():
            self._display_recent_underlying_prices(date)

        # Delegate to strategy's recommend_open_position method
        recommendation = self.strategy.recommend_open_position(date, current_price)
        if recommendation is None:
            print("No recommendation found for strategy")
            return None

        # Extract recommendation details
        strategy_type = recommendation["strategy_type"]
        legs = recommendation["legs"]
        credit = recommendation["credit"]
        width = recommendation["width"]
        probability_of_profit = recommendation["probability_of_profit"]
        confidence = recommendation["confidence"]
        expiration_date = recommendation["expiration_date"]

        # Get strategy name from class and map to config key
        strategy_name = self._get_strategy_name_from_class()

        # Build proposal DTO
        proposal = ProposedPositionRequest(
            symbol=self.options_handler.symbol,
            strategy_type=strategy_type,
            legs=legs,
            credit=float(credit),
            width=float(width),
            probability_of_profit=float(probability_of_profit),
            confidence=float(confidence),
            expiration_date=expiration_date,
            created_at=datetime.now(timezone.utc).isoformat(),
            strategy_name=strategy_name,
        )

        # Calculate max risk
        max_risk = self._calculate_max_risk(strategy_type, float(width), float(credit))
        
        # Check risk threshold
        is_allowed, risk_message = self.capital_manager.check_risk_threshold(strategy_name, max_risk)
        
        if not is_allowed:
            print(f"❌ Risk check failed: {risk_message}")
            print("Position rejected due to risk threshold.")
            return None
        
        # Determine if credit or debit strategy
        is_credit = self.capital_manager.is_credit_strategy(strategy_type)
        premium_amount = float(credit) * 100  # Convert to dollars per contract
        premium_label = "Premium received" if is_credit else "Premium paid"

        # Prompt user
        summary = self._format_open_summary(proposal, recommendation, max_risk, premium_amount, premium_label, risk_message)
        if not self.prompt(f"Open recommendation:\n{summary}\nOpen this position?"):
            return None

        decided_at = datetime.now(timezone.utc).isoformat()
        record = DecisionResponse(
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

    # ---------- Close recommendation ----------

    def recommend_close_positions(self, date: datetime) -> List[DecisionResponse]:
        """Check open decisions and recommend closure when rules trigger."""
        open_records = self.decision_store.get_open_positions()
        closed_records: List[DecisionResponse] = []

        if not open_records:
            return closed_records

        # Convert open records to Position objects
        strategy_positions = []
        for rec in open_records:
            position = self._position_from_decision(rec)
            strategy_positions.append(position)

        # Use the strategy's recommend_close_positions method
        try:
            close_recommendations = self.strategy.recommend_close_positions(date, strategy_positions)
        except Exception as e:
            print(f"Error in strategy recommend_close_positions: {e}")
            raise e

        # Process the strategy's recommendations
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
                    updated = DecisionResponse(
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

    def _position_from_decision(self, rec: DecisionResponse) -> Position:
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

    def _format_open_summary(self, proposal: ProposedPositionRequest, best: dict, max_risk: float, premium_amount: float, premium_label: str, risk_message: str) -> str:
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
            
            # Get bar data for both options using new_options_handler
            atm_bar = self.strategy.new_options_handler.get_option_bar(atm_option, fetch_date)
            otm_bar = self.strategy.new_options_handler.get_option_bar(otm_option, fetch_date)
            
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
            
            # Get bar data for both options using new_options_handler
            atm_bar = self.strategy.new_options_handler.get_option_bar(atm_option, fetch_date)
            otm_bar = self.strategy.new_options_handler.get_option_bar(otm_option, fetch_date)
            
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
