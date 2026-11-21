"""Tests for ProposedPositionRequestDTO persistence."""

from datetime import datetime

from algo_trading_engine.dto import ProposedPositionRequestDTO


def _leg(timestamp: str | None = None) -> dict:
    leg = {
        "ticker": "O:SPY261130C00740000",
        "symbol": "SPY",
        "strike": 740.0,
        "expiration": "2026-11-30",
        "type": "call",
        "last_price": 12.33,
    }
    if timestamp is not None:
        leg["timestamp"] = timestamp
    return leg


def _proposal(legs: list[dict]) -> dict:
    return {
        "symbol": "SPY",
        "strategy_type": "call_debit_spread",
        "legs": legs,
        "credit": 1.61,
        "width": 5.0,
        "probability_of_profit": 0.6,
        "confidence": 0.7,
        "expiration_date": "2026-11-30",
        "created_at": "2026-11-10T15:30:00",
        "strategy_name": "uptrend_swing",
    }


def test_from_dict_uses_created_at_when_leg_has_no_timestamp():
    created_at = "2026-11-10T15:30:00"
    stored_leg = _leg()
    restored = ProposedPositionRequestDTO.from_dict(_proposal([stored_leg]))

    assert restored.legs[0].timestamp == datetime.fromisoformat(created_at)
    assert "timestamp" not in stored_leg


def test_from_dict_keeps_an_existing_leg_timestamp():
    quote_time = "2026-11-10T14:00:00"
    restored = ProposedPositionRequestDTO.from_dict(_proposal([_leg(quote_time)]))

    assert restored.legs[0].timestamp == datetime.fromisoformat(quote_time)
