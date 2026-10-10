from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Optional, List
import hashlib
import json
import os

from algo_trading_engine.dto.decisions import DecisionResponseDTO
from algo_trading_engine.dto.trade import ProposedPositionRequestDTO
from algo_trading_engine.enums import StrategyType


def generate_decision_id(proposal: ProposedPositionRequestDTO, decided_at_iso: str) -> str:
    """Generate a deterministic ID for a decision.

    Incorporates symbol, strategy_type, legs signature (type/strike/expiration),
    expiration_date, and decided_at timestamp to guard against duplicates while
    remaining stable for the same input decision.
    """
    legs_sig_parts: List[str] = []
    for leg in proposal.legs:
        # Use a compact signature of leg attributes that uniquely identify the contract
        type_code = leg.option_type.value if hasattr(leg, "option_type") else "?"
        legs_sig_parts.append(f"{type_code}:{leg.strike}:{leg.expiration}")
    legs_signature = ",".join(sorted(legs_sig_parts))

    raw = "|".join(
        [
            proposal.symbol,
            proposal.strategy_type.value,
            proposal.strategy_name,
            legs_signature,
            str(proposal.width),
            str(proposal.credit),
            proposal.expiration_date,
            decided_at_iso,
        ]
    )
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


class DecisionStore(ABC):
    """Abstract base class for decision storage backends."""

    @abstractmethod
    def append_decision(self, record: DecisionResponseDTO) -> None:
        """Persist a decision record."""

    @abstractmethod
    def get_open_positions(
        self,
        symbol: Optional[str] = None,
        strategy_type: Optional[StrategyType] = None,
        strategy_name: Optional[str] = None,
    ) -> List[DecisionResponseDTO]:
        """Return all accepted-but-not-closed decisions.

        Filter by symbol, strategy_type, and/or strategy_name if provided.
        """

    @abstractmethod
    def get_all_decisions(
        self,
        strategy_name: Optional[str] = None,
        symbol: Optional[str] = None,
    ) -> List[DecisionResponseDTO]:
        """Return all accepted decisions (both open and closed).

        Filter by strategy_name and/or symbol if provided.
        """

    @abstractmethod
    def mark_closed(self, open_decision_id: str, exit_price: float, closed_at: datetime) -> None:
        """Mark a previously accepted decision as closed."""


class JsonDecisionStore(DecisionStore):
    """Append-only JSON storage for decision records per day.

    - Files are written to `<base_dir>/decisions_YYYYMMDD.json`
    - Appends are performed by reading the existing array, appending, then
      writing the whole file back with fsync for single-user safety.
    - Retrieval scans all files in the base directory.
    """

    def __init__(self, base_dir: str = "predictions/decisions") -> None:
        self.base_dir = base_dir
        self._ensure_base_dir()

    def append_decision(self, record: DecisionResponseDTO) -> None:
        """Append a decision to the file for the decision date.

        Chooses the file based on `record.decided_at` date.
        """
        decided_date = _iso_to_date(record.decided_at)
        path = self._file_path_for_date(decided_date)
        records = self._read_records(path)

        # Guard duplicates by ID
        if any(r.get("id") == record.id for r in records):
            return

        records.append(record.to_dict())
        self._write_records(path, records)

    def get_open_positions(
        self,
        symbol: Optional[str] = None,
        strategy_type: Optional[StrategyType] = None,
        strategy_name: Optional[str] = None,
    ) -> List[DecisionResponseDTO]:
        """Return all accepted-but-not-closed decisions across all files.

        Filter by symbol, strategy_type, and/or strategy_name if provided.
        """
        results: List[DecisionResponseDTO] = []
        for path in self._list_decision_files():
            for rec in self._read_records(path):
                try:
                    record = DecisionResponseDTO.from_dict(rec)
                except Exception:
                    continue
                if record.closed_at is not None:
                    continue
                if symbol and record.proposal.symbol != symbol:
                    continue
                if strategy_type and record.proposal.strategy_type != strategy_type:
                    continue
                if strategy_name and record.proposal.strategy_name != strategy_name:
                    continue
                results.append(record)
        return results

    def get_all_decisions(
        self,
        strategy_name: Optional[str] = None,
        symbol: Optional[str] = None,
    ) -> List[DecisionResponseDTO]:
        """Return all accepted decisions (both open and closed) across all files.
        
        Filter by strategy_name and/or symbol if provided.
        """
        results: List[DecisionResponseDTO] = []
        for path in self._list_decision_files():
            for rec in self._read_records(path):
                try:
                    record = DecisionResponseDTO.from_dict(rec)
                except Exception:
                    continue
                if strategy_name and record.proposal.strategy_name != strategy_name:
                    continue
                if symbol and record.proposal.symbol != symbol:
                    continue
                results.append(record)
        return results

    def mark_closed(self, open_decision_id: str, exit_price: float, closed_at: datetime) -> None:
        """Mark a previously accepted decision as closed by setting exit values.

        Searches across all decision files to locate the matching ID and updates
        that record in-place.
        """
        closed_iso = closed_at.isoformat()
        for path in self._list_decision_files():
            records = self._read_records(path)
            modified = False
            for rec in records:
                if rec.get("id") == open_decision_id:
                    if rec.get("closed_at"):
                        return  # already closed
                    rec["exit_price"] = float(exit_price)
                    rec["closed_at"] = closed_iso
                    modified = True
                    break
            if modified:
                self._write_records(path, records)
                return
        raise ValueError(f"Open decision id not found: {open_decision_id}")

    # ---------- internal helpers ----------

    def _ensure_base_dir(self) -> None:
        os.makedirs(self.base_dir, exist_ok=True)

    def _file_path_for_date(self, date_obj: datetime.date) -> str:
        fname = f"decisions_{date_obj.strftime('%Y%m%d')}.json"
        return os.path.join(self.base_dir, fname)

    def _list_decision_files(self) -> List[str]:
        if not os.path.isdir(self.base_dir):
            return []
        files = [
            os.path.join(self.base_dir, f)
            for f in os.listdir(self.base_dir)
            if f.startswith("decisions_") and f.endswith(".json")
        ]
        files.sort()
        return files

    def _read_records(self, path: str) -> List[dict]:
        if not os.path.exists(path):
            return []
        with open(path, "r", encoding="utf-8") as f:
            try:
                data = json.load(f)
                if isinstance(data, list):
                    return data
                return []
            except json.JSONDecodeError:
                return []

    def _write_records(self, path: str, records: List[dict]) -> None:
        # Ensure directory exists
        os.makedirs(os.path.dirname(path), exist_ok=True)

        # Write atomically then fsync to reduce corruption risk
        tmp_path = f"{path}.tmp"
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(records, f, indent=2, sort_keys=True)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, path)


def _iso_to_date(iso_str: str) -> datetime.date:
    # Support both date-only and full datetime strings
    try:
        return datetime.fromisoformat(iso_str).date()
    except Exception:
        return datetime.strptime(iso_str, "%Y-%m-%d").date()


