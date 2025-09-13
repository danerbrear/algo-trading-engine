from __future__ import annotations

from datetime import datetime
from typing import Optional, List

from src.backtest.models import Position, StrategyType, Strategy
from src.common.models import Option, OptionChain
from src.prediction.decision_store import (
    JsonDecisionStore,
    ProposedPositionDTO,
    DecisionRecord,
    generate_decision_id,
)


class InteractiveStrategyRecommender:
    """Produce open/close recommendations and capture user decisions.

    Expects a strategy compatible with credit spread workflows (e.g.,
    CreditSpreadStrategy), with `.data`, `.options_data`, and helpers used
    in the existing backtest (`_find_best_spread`, `_ensure_volume_data`,
    `get_current_volumes_for_position`).
    """

    def __init__(self, strategy: Strategy, options_handler, decision_store: JsonDecisionStore, auto_yes: bool = False):
        self.strategy = strategy
        self.options_handler = options_handler
        self.decision_store = decision_store
        self.auto_yes = auto_yes

    def run(self, date: datetime, auto_yes: Optional[bool] = None) -> None:
        if auto_yes is not None:
            self.auto_yes = auto_yes
        self.recommend_open_position(date)
        self.recommend_close_positions(date)

    # ---------- Open recommendation ----------

    def recommend_open_position(self, date: datetime) -> Optional[DecisionRecord]:
        """Use the strategy to propose an opening trade for the date and capture decision."""
        # Require strategy data
        if getattr(self.strategy, "data", None) is None:
            print("No data found for strategy")
            return None

        # Determine current price
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

        # Build proposal DTO
        proposal = ProposedPositionDTO(
            symbol=self.options_handler.symbol,
            strategy_type=strategy_type,
            legs=legs,
            credit=float(credit),
            width=float(width),
            probability_of_profit=float(probability_of_profit),
            confidence=float(confidence),
            expiration_date=expiration_date,
            created_at=datetime.utcnow().isoformat(),
        )

        # Prompt user
        summary = self._format_open_summary(proposal, recommendation)
        if not self.prompt(f"Open recommendation:\n{summary}\nOpen this position?"):
            return None

        decided_at = datetime.utcnow().isoformat()
        record = DecisionRecord(
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

    def recommend_close_positions(self, date: datetime) -> List[DecisionRecord]:
        """Check open decisions and recommend closure when rules trigger."""
        open_records = self.decision_store.get_open_positions()
        closed_records: List[DecisionRecord] = []

        for rec in open_records:
            position = self._position_from_decision(rec)

            # Calculate exit price
            exit_price = self._compute_exit_price_with_fetch(position, date)

            if exit_price is None:
                continue

            # Evaluate rules
            rationale_parts: List[str] = []
            should_close = False

            if position.get_days_to_expiration(date) < 1:
                should_close = True
                rationale_parts.append("expiration proximity")
            elif hasattr(self.strategy, "_profit_target_hit") and self.strategy._profit_target_hit(position, exit_price):  # noqa: SLF001
                should_close = True
                rationale_parts.append("profit target hit")
            elif hasattr(self.strategy, "_stop_loss_hit") and self.strategy._stop_loss_hit(position, exit_price):  # noqa: SLF001
                should_close = True
                rationale_parts.append("stop loss hit")
            elif hasattr(self.strategy, "holding_period") and position.get_days_held(date) >= getattr(self.strategy, "holding_period", 15):
                should_close = True
                rationale_parts.append("holding period reached")

            # Enhanced volume validation (optional): fetch and report
            volume_note = None
            if hasattr(self.strategy, "get_current_volumes_for_position"):
                try:
                    vols = self.strategy.get_current_volumes_for_position(position, date)
                    if vols and any(v is None or v <= 0 for v in vols):
                        volume_note = "volume: missing/low"
                    else:
                        volume_note = "volume: ok"
                except Exception:
                    volume_note = "volume: error"

            if not should_close:
                continue

            rationale = "; ".join(rationale_parts + ([volume_note] if volume_note else []))

            # Prompt user
            msg = self._format_close_summary(position, exit_price, rationale)
            if not self.prompt(f"Close recommendation:\n{msg}\nClose this position?"):
                continue

            # Mark closed in the store
            self.decision_store.mark_closed(rec.id, exit_price=exit_price, closed_at=date)
            # Return an updated record instance for the caller
            updated = DecisionRecord(
                id=rec.id,
                proposal=rec.proposal,
                outcome=rec.outcome,
                decided_at=rec.decided_at,
                rationale=rationale,
                quantity=rec.quantity,
                entry_price=rec.entry_price,
                exit_price=exit_price,
                closed_at=date.isoformat(),
            )
            closed_records.append(updated)

        return closed_records

    def get_open_positions_status(self, date: datetime) -> List[dict]:
        """Return current stats for all open positions without prompting.

        Stats include exit price, P&L dollars and percent, days held, and DTE.
        """
        statuses: List[dict] = []
        for rec in self.decision_store.get_open_positions():
            position = self._position_from_decision(rec)
            exit_price = self._compute_exit_price_with_fetch(position, date)
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

    def _position_from_decision(self, rec: DecisionRecord) -> Position:
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

    def _format_open_summary(self, proposal: ProposedPositionDTO, best: dict) -> str:
        legs_str = ", ".join(
            [f"{leg.option_type.value.upper()} {int(leg.strike)} exp {leg.expiration}" for leg in proposal.legs]
        )
        rr = best.get("risk_reward")
        return (
            f"Symbol: {proposal.symbol}\n"
            f"Strategy: {proposal.strategy_type.value}\n"
            f"Legs: {legs_str}\n"
            f"Credit: ${proposal.credit:.2f}  Width: {proposal.width}  R/R: {rr}  Prob: {proposal.probability_of_profit:.0%}"
        )

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

    def _compute_exit_price_with_fetch(self, position: Position, date: datetime) -> Optional[float]:
        """Compute exit price using cached chain or by fetching missing contracts for legs."""
        option_chain = None
        date_key = date.strftime("%Y-%m-%d")
        if getattr(self.strategy, "options_data", None) and date_key in self.strategy.options_data:
            option_chain = self.strategy.options_data[date_key]

        exit_price = None
        if option_chain is not None:
            try:
                exit_price = position.calculate_exit_price(option_chain)
            except Exception:
                exit_price = None

        if exit_price is None:
            option_chain = option_chain or OptionChain()
            for leg in position.spread_options:
                contract = self.options_handler.get_specific_option_contract(
                    leg.strike,
                    leg.expiration,
                    leg.option_type.value,
                    date,
                )
                if contract:
                    option_chain = option_chain.add_option(contract)
            try:
                exit_price = position.calculate_exit_price(option_chain)
            except Exception:
                exit_price = None
        return exit_price


