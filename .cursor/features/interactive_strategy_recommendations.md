# Interactive Strategy Recommendations and Decision Capture

## Feature Description
Add a class under `src/prediction/` that executes a `Strategy` instance to produce actionable trade recommendations for the current market day, prompts the user to accept or reject proposed trades, persists accepted decisions to JSON, and, when a position is already open, produces a recommendation on whether to close it.

## Overview
This feature connects live prediction outputs to an actionable, auditable decision flow:
- Execute a `Strategy` (e.g., `CreditSpreadStrategy`) against current data to propose a single new position recommendation when appropriate.
- If a recommendation is produced, interactively prompt the user to accept or reject it.
- Persist accepted decisions as immutable JSON records under `predictions/` for traceability.
- Detect any currently open position(s) from the decision store and recommend whether to close them (profit target, stop loss, holding period, or expiration proximity), using current market data and existing logic (including current-date volume validation where available).

Key principles:
- Single responsibility per module (storage, recommendation, CLI wiring)
- DTOs/VOs instead of raw dicts
- No backward-compat shims; update call sites as needed
- Unit tests for all new functions/classes

## Implementation Plan

### Phase 1: Decision DTOs and JSON Store

- File: `src/prediction/decision_store.py`
- Components:
  - Reuse existing VOs where possible:
    - Legs use `src/common/models.Option`
    - Strategy uses `src/backtest/models.StrategyType`
    - Runtime position logic uses `src/backtest/models.Position` (not stored directly)
  - `ProposedPositionDTO` (DTO): represents a full proposed position (symbol, strategy_type [StrategyType], legs [tuple[Option, ...]], credit, width, probability_of_profit, confidence, expiration_date, created_at)
  - `DecisionRecord` (DTO): immutable record capturing the decision outcome (accepted/rejected), timestamps, and rationale
  - `JsonDecisionStore` (class): append-only JSON storage and retrieval
    - Default path: `predictions/decisions/decisions_YYYYMMDD.json`
    - Methods:
      - `append_decision(record: DecisionRecord) -> None`
      - `get_open_positions(symbol: str | None = None, strategy_type: StrategyType | None = None) -> list[DecisionRecord]`
      - `mark_closed(open_position_id: str, exit_price: float, closed_at: datetime) -> None`
    - Notes:
      - Supports multiple strategies per day/file; `DecisionRecord.proposal.strategy_type` is persisted.
      - Deterministic IDs should include `strategy_type` in addition to `symbol`, `legs`, and timestamps to avoid collisions across strategies.

Validation:
- Ensure file/folder creation is idempotent
- Lockless write safety for single-user CLI (append then fsync)

### Phase 2: Interactive Strategy Recommender

- File: `src/prediction/recommendation_engine.py`
- Class: `InteractiveStrategyRecommender`
- Responsibilities:
  - Use a provided `Strategy` instance (e.g., `CreditSpreadStrategy`) pre-wired with data and `OptionsHandler`
  - For the current date, produce a `ProposedPositionDTO` if the strategy recommends opening a position
  - Prompt user for acceptance via `input()` (with `--yes` flag for non-interactive environments)
  - Persist accepted decision via `JsonDecisionStore`
  - Detect open positions and recommend closure:
    - Construct a runtime `Position` from accepted decision(s) and compute exit price using `Position.calculate_exit_price()` when current `OptionChain` is available
    - Apply existing rules: profit target, stop loss, holding period, expiration proximity
    - If enhanced volume validation is enabled, fetch current volumes via strategy helper and use current-date volume thresholds when advising closure
  - Provide an explanation/rationale string for each recommendation (probability, R/R, rule hit, volume status)

Public methods:
- `recommend_open_position(date: datetime) -> DecisionRecord | None`
- `recommend_close_positions(date: datetime) -> list[DecisionRecord]`
- `run(date: datetime, auto_yes: bool = False) -> None` (orchestrates open/close flows)

Persistence decisions:
- Accepted open decisions create a new open position entry
- Accepted close decisions update the corresponding decision as closed (store `exit_price` and `closed_at`)

### Phase 3: CLI Wiring

- File: `src/prediction/recommend_cli.py`
- Purpose: Simple CLI to run the recommender end-to-end
- Behavior:
  - Ensures venv is active (documented in README; CI assumes venv)
  - Arguments: `--symbol`, `--date YYYY-MM-DD`, `--strategy`, `--yes`
    - `--strategy`: non-interactive strategy selection (default: `credit_spread`)
  - Open-position short-circuit: If the decision store contains any accepted-but-open positions for the selected `--symbol`, the CLI skips historical data fetching/model prediction entirely and only runs the close recommendation flow for the date
  - Otherwise, it instantiates `OptionsHandler`, loads recent data via `DataRetriever` (fixed lookback window to seed indicators and LSTM sequence), builds the selected Strategy via a registry, and runs `InteractiveStrategyRecommender` for the date

### Phase 4: Integration Touchpoints

- `src/prediction/predict_today.py`
  - Optional: expose a helper to return the predicted strategy label and confidence for UI display within the CLI
- `src/strategies/credit_spread_minimal.py`
  - Already includes methods to evaluate spreads, ensure volume, and fetch current volumes for closure
  - Ensure it can be reused in a single-day interactive context by providing the latest `data` and an `OptionsHandler`

### Phase 5: Testing

- File: `tests/test_interactive_recommender.py`
- Coverage:
  - JSON store append/read/mark-closed behavior
  - Open recommendation: acceptance path persists correct DTO
  - Close recommendation: recommendations produced for profit target, stop loss, holding period, and expiration proximity
  - Non-interactive mode (`--yes`) accepts by default
  - Volume validation usage on closure path (pass current volumes)
  - Deterministic prompts mocked via `unittest.mock`

## Implementation Details (Proposed APIs)

### DTOs and Store
```python
# src/prediction/decision_store.py
from dataclasses import dataclass
from datetime import datetime
from typing import Optional, Literal, Tuple
from src.common.models import Option
from src.backtest.models import StrategyType

DecisionOutcome = Literal['accepted', 'rejected']

@dataclass(frozen=True)
class ProposedPositionDTO:
    symbol: str
    strategy_type: StrategyType
    legs: Tuple[Option, ...]
    credit: float
    width: float
    probability_of_profit: float
    confidence: float
    expiration_date: str
    created_at: str  # ISO timestamp

@dataclass(frozen=True)
class DecisionRecord:
    id: str
    proposal: ProposedPositionDTO
    outcome: DecisionOutcome
    decided_at: str
    rationale: str
    quantity: Optional[int] = None
    entry_price: Optional[float] = None
    exit_price: Optional[float] = None
    closed_at: Optional[str] = None

class JsonDecisionStore:
    def append_decision(self, record: DecisionRecord) -> None: ...
    def get_open_positions(self, symbol: Optional[str] = None, strategy_type: Optional[StrategyType] = None) -> list[DecisionRecord]: ...
    def mark_closed(self, open_decision_id: str, exit_price: float, closed_at: datetime) -> None: ...
```

### Recommender Core
```python
# src/prediction/recommendation_engine.py
from dataclasses import asdict
from datetime import datetime
from typing import Optional

class InteractiveStrategyRecommender:
    def __init__(self, strategy, options_handler, decision_store, auto_yes: bool = False):
        self.strategy = strategy
        self.options_handler = options_handler
        self.decision_store = decision_store
        self.auto_yes = auto_yes

    def recommend_open_position(self, date: datetime):
        # Use strategy to propose spread (reusing _find_best_spread + _ensure_volume_data)
        # Build ProposedPositionDTO; if None, return None
        # Prompt user; on accept, persist DecisionRecord and return it
        ...

    def recommend_close_positions(self, date: datetime):
        # For each open position from store:
        #   - construct a runtime Position from the accepted DecisionRecord
        #   - compute exit price via Position.calculate_exit_price(current_chain)
        #   - fetch current volumes and apply volume validation for closure
        #   - derive rationale (profit target/stop loss/holding/expiration)
        #   - prompt user; on accept, mark closed in store
        ...

    def prompt(self, message: str) -> bool:
        if self.auto_yes:
            return True
        answer = input(f"{message} [y/N]: ").strip().lower()
        return answer in {"y", "yes"}
```

### CLI Entrypoint
```python
# src/prediction/recommend_cli.py
import argparse
from datetime import datetime

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--symbol', default='SPY')
    parser.add_argument('--yes', action='store_true')
    parser.add_argument('--date')
    parser.add_argument('--strategy', default='credit_spread',
                        help='Strategy to run (non-interactive). Default: credit_spread')
    args = parser.parse_args()

    # Wire OptionsHandler, DataRetriever, Strategy, Store, Recommender
    # Build strategy via a small registry mapping
    # STRATEGY_REGISTRY = { 'credit_spread': CreditSpreadStrategy }
    # strategy = STRATEGY_REGISTRY[args.strategy](...)
    # Ensure venv is active per project standards
    # Run recommender.run(date, auto_yes=args.yes)

if __name__ == '__main__':
    main()
```

## Success Criteria
1. A new class under `src/prediction/` can execute a `Strategy` and produce a proposed position when appropriate.
2. The user is prompted to accept or reject proposed positions; accepted ones are stored as immutable JSON decision records.
3. Existing open positions are detected and a close recommendation is produced with rationale; accepted closures update the JSON record.
4. DTOs are used for proposals and decisions; reuse existing VOs (`Option`, `StrategyType`, and runtime `Position`) and avoid duplicate leg/open-position VOs.
5. Tests cover storage, open recommendation, close recommendation, and non-interactive flows.

## Risk Mitigation
- Non-interactive environments: provide `--yes` flag to auto-accept recommendations (useful for batch runs). Default is safe reject.
- Data availability: if current option chain is unavailable, fall back to generic recommendations without persisting.
- Volume validation: closure decisions should use current-date volume when available to avoid stale liquidity assumptions.
- Idempotency: decision store appends are append-only; duplicates guarded via deterministic IDs (hash of symbol+timestamp+legs).

## Timeline Estimate
- Phase 1 (DTOs/Store): 0.5 day
- Phase 2 (Recommender): 1.0 day
- Phase 3 (CLI): 0.5 day
- Phase 4 (Integration): 0.5 day
- Phase 5 (Tests/Docs): 0.5–1.0 day

Total: 2.5–3.5 days

## Compliance with .cursor/rules

### Agentic Standards
- No deprecated functions; new modules are clean additions
- New classes covered by unit tests
- No new third-party libs required; if added, will be listed in `requirements.txt`
- All imports are used and validated

### DTO Rules
- Well-defined DTOs (`ProposedPositionDTO`, `DecisionRecord`); reuse `Option` for legs and `StrategyType` for strategy
- Clear naming, flat structures, immutability via `@dataclass(frozen=True)` where appropriate
- Validation at I/O boundaries (store)

### VO Rules
- Reuse existing Value Objects from `src/common/models.py` (`Option`) and `src/backtest/models.py` (`StrategyType`, runtime `Position`). No new leg or open-position VOs are introduced.
- Business logic remains in strategy and recommender; DTOs transport data only

### Project Structure
- New modules placed under `src/prediction/` per project organization
- CLI under prediction package runnable via `python -m src.prediction.recommend_cli`
- Outputs written under `predictions/decisions/` with timestamped filenames


