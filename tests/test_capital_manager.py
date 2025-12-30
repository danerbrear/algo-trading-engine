"""Unit tests for CapitalManager class."""

import pytest
import json
import tempfile
import os
from pathlib import Path

from algo_trading_engine.prediction.capital_manager import CapitalManager
from algo_trading_engine.prediction.decision_store import JsonDecisionStore, ProposedPositionRequest, DecisionResponse, generate_decision_id
from algo_trading_engine.backtest.models import StrategyType
from algo_trading_engine.common.models import Option
from datetime import datetime


def _make_option(symbol: str, strike: float, expiration: str, opt_type: str, last: float) -> Option:
    """Helper to create Option objects for testing."""
    return Option.from_dict({
        'symbol': symbol,
        'ticker': symbol,
        'strike': strike,
        'expiration': expiration,
        'type': opt_type,
        'last_price': last,
        'bid': last - 0.1,
        'ask': last + 0.1,
        'volume': 100,
    })


@pytest.fixture
def allocations_config():
    """Fixture for capital allocations config."""
    return {
        "strategies": {
            "credit_spread": {
                "allocated_capital": 10000.0,
                "max_risk_percentage": 0.05
            },
            "velocity_momentum": {
                "allocated_capital": 15000.0,
                "max_risk_percentage": 0.03
            }
        }
    }


@pytest.fixture
def temp_config_file(allocations_config, tmp_path):
    """Create a temporary config file."""
    config_path = tmp_path / "capital_allocations.json"
    with open(config_path, 'w') as f:
        json.dump(allocations_config, f)
    return str(config_path)


@pytest.fixture
def decision_store(tmp_path):
    """Create a JsonDecisionStore for testing."""
    return JsonDecisionStore(base_dir=str(tmp_path))


@pytest.fixture
def capital_manager(allocations_config, decision_store):
    """Create a CapitalManager for testing."""
    return CapitalManager(allocations_config, decision_store)


def test_capital_manager_init(allocations_config, decision_store):
    """Test CapitalManager initialization."""
    manager = CapitalManager(allocations_config, decision_store)
    assert manager.allocations == allocations_config["strategies"]
    assert manager.decision_store == decision_store


def test_capital_manager_from_config_file(temp_config_file, decision_store):
    """Test loading CapitalManager from config file."""
    manager = CapitalManager.from_config_file(temp_config_file, decision_store)
    assert manager.get_allocated_capital("credit_spread") == 10000.0
    assert manager.get_allocated_capital("velocity_momentum") == 15000.0


def test_capital_manager_from_config_file_not_found(decision_store):
    """Test loading CapitalManager with missing config file."""
    with pytest.raises(FileNotFoundError):
        CapitalManager.from_config_file("nonexistent.json", decision_store)


def test_get_allocated_capital(capital_manager):
    """Test getting allocated capital for a strategy."""
    assert capital_manager.get_allocated_capital("credit_spread") == 10000.0
    assert capital_manager.get_allocated_capital("velocity_momentum") == 15000.0
    assert capital_manager.get_allocated_capital("nonexistent") == 0.0


def test_get_max_risk_percentage(capital_manager):
    """Test getting max risk percentage."""
    assert capital_manager.get_max_risk_percentage("credit_spread") == 0.05
    assert capital_manager.get_max_risk_percentage("velocity_momentum") == 0.03


def test_get_max_allowed_risk(capital_manager):
    """Test getting max allowed risk per position (based on remaining capital)."""
    # With no open positions, remaining = allocated
    assert capital_manager.get_max_allowed_risk("credit_spread") == 500.0  # 10000 * 0.05
    assert capital_manager.get_max_allowed_risk("velocity_momentum") == 450.0  # 15000 * 0.03


def test_get_max_allowed_risk_with_open_position(allocations_config, decision_store):
    """Test max allowed risk adjusts based on remaining capital."""
    manager = CapitalManager(allocations_config, decision_store)
    
    # Open a debit position that reduces remaining capital
    legs = (
        _make_option('OPT1', 500, '2025-09-06', 'call', 20.0),
        _make_option('OPT2', 505, '2025-09-06', 'call', 1.0),
    )
    proposal = ProposedPositionRequest(
        symbol='SPY',
        strategy_type=StrategyType.LONG_CALL,
        legs=legs,
        credit=50.0,  # $5000 debit paid
        width=5.0,
        probability_of_profit=0.6,
        confidence=0.65,
        expiration_date='2025-09-06',
        created_at=datetime.now().isoformat(),
        strategy_name='credit_spread',
    )
    record = DecisionResponse(
        id=generate_decision_id(proposal, datetime.now().isoformat()),
        proposal=proposal,
        outcome='accepted',
        decided_at=datetime.now().isoformat(),
        rationale='test',
        quantity=1,
        entry_price=50.0,
    )
    decision_store.append_decision(record)
    
    # Remaining capital: 10000 - 5000 = 5000
    # Max allowed risk: 5000 * 0.05 = 250
    assert manager.get_max_allowed_risk("credit_spread") == 250.0


def test_is_credit_strategy(capital_manager):
    """Test credit strategy detection."""
    assert capital_manager.is_credit_strategy(StrategyType.PUT_CREDIT_SPREAD) is True
    assert capital_manager.is_credit_strategy(StrategyType.CALL_CREDIT_SPREAD) is True
    assert capital_manager.is_credit_strategy(StrategyType.SHORT_CALL) is True
    assert capital_manager.is_credit_strategy(StrategyType.SHORT_PUT) is True
    assert capital_manager.is_credit_strategy(StrategyType.LONG_CALL) is False
    assert capital_manager.is_credit_strategy(StrategyType.LONG_PUT) is False
    assert capital_manager.is_credit_strategy(StrategyType.LONG_STOCK) is False


def test_check_risk_threshold_pass(capital_manager):
    """Test risk threshold check that passes."""
    is_allowed, message = capital_manager.check_risk_threshold("credit_spread", 400.0)
    assert is_allowed is True
    assert "Risk check passed" in message


def test_check_risk_threshold_exceeds_max(capital_manager):
    """Test risk threshold check that exceeds max allowed risk."""
    is_allowed, message = capital_manager.check_risk_threshold("credit_spread", 600.0)
    assert is_allowed is False
    assert "exceeds maximum allowed risk" in message
    assert "$600.00" in message
    assert "$500.00" in message


def test_check_risk_threshold_strategy_not_found(capital_manager):
    """Test risk threshold check for unknown strategy."""
    is_allowed, message = capital_manager.check_risk_threshold("nonexistent", 100.0)
    assert is_allowed is False
    assert "not found in capital allocations" in message


def test_get_remaining_capital_no_decisions(capital_manager):
    """Test remaining capital calculation with no decisions."""
    remaining = capital_manager.get_remaining_capital("credit_spread")
    assert remaining == 10000.0  # Just the allocated capital


def test_get_remaining_capital_with_credit_position(allocations_config, decision_store):
    """Test remaining capital with opened credit position."""
    manager = CapitalManager(allocations_config, decision_store)
    
    # Create a credit spread decision
    legs = (
        _make_option('OPT1', 500, '2025-09-06', 'put', 2.0),
        _make_option('OPT2', 495, '2025-09-06', 'put', 1.0),
    )
    proposal = ProposedPositionRequest(
        symbol='SPY',
        strategy_type=StrategyType.PUT_CREDIT_SPREAD,
        legs=legs,
        credit=1.0,
        width=5.0,
        probability_of_profit=0.6,
        confidence=0.65,
        expiration_date='2025-09-06',
        created_at=datetime.now().isoformat(),
        strategy_name='credit_spread',
    )
    record = DecisionResponse(
        id=generate_decision_id(proposal, datetime.now().isoformat()),
        proposal=proposal,
        outcome='accepted',
        decided_at=datetime.now().isoformat(),
        rationale='test',
        quantity=1,
        entry_price=1.0,
    )
    decision_store.append_decision(record)
    
    # Credit spread: should add premium received (1.0 * 100 = $100)
    remaining = manager.get_remaining_capital("credit_spread")
    assert remaining == 10100.0  # 10000 + 100


def test_get_remaining_capital_with_debit_position(allocations_config, decision_store):
    """Test remaining capital with opened debit position."""
    manager = CapitalManager(allocations_config, decision_store)
    
    # Create a debit spread decision (simulated as LONG_CALL)
    legs = (
        _make_option('OPT1', 500, '2025-09-06', 'call', 2.0),
        _make_option('OPT2', 505, '2025-09-06', 'call', 1.0),
    )
    proposal = ProposedPositionRequest(
        symbol='SPY',
        strategy_type=StrategyType.LONG_CALL,
        legs=legs,
        credit=1.0,  # Actually a debit, but credit field represents net premium
        width=5.0,
        probability_of_profit=0.6,
        confidence=0.65,
        expiration_date='2025-09-06',
        created_at=datetime.now().isoformat(),
        strategy_name='credit_spread',
    )
    record = DecisionResponse(
        id=generate_decision_id(proposal, datetime.now().isoformat()),
        proposal=proposal,
        outcome='accepted',
        decided_at=datetime.now().isoformat(),
        rationale='test',
        quantity=1,
        entry_price=1.0,
    )
    decision_store.append_decision(record)
    
    # Debit strategy: should subtract premium paid (1.0 * 100 = $100)
    remaining = manager.get_remaining_capital("credit_spread")
    assert remaining == 9900.0  # 10000 - 100


def test_get_remaining_capital_with_closed_credit_position(allocations_config, decision_store):
    """Test remaining capital with closed credit position."""
    manager = CapitalManager(allocations_config, decision_store)
    
    # Create and close a credit spread
    legs = (
        _make_option('OPT1', 500, '2025-09-06', 'put', 2.0),
        _make_option('OPT2', 495, '2025-09-06', 'put', 1.0),
    )
    proposal = ProposedPositionRequest(
        symbol='SPY',
        strategy_type=StrategyType.PUT_CREDIT_SPREAD,
        legs=legs,
        credit=1.86,
        width=5.0,
        probability_of_profit=0.6,
        confidence=0.65,
        expiration_date='2025-09-06',
        created_at=datetime.now().isoformat(),
        strategy_name='credit_spread',
    )
    record = DecisionResponse(
        id=generate_decision_id(proposal, datetime.now().isoformat()),
        proposal=proposal,
        outcome='accepted',
        decided_at=datetime.now().isoformat(),
        rationale='test',
        quantity=1,
        entry_price=1.86,
        exit_price=2.95,  # Paid more to close (loss)
        closed_at=datetime.now().isoformat(),
    )
    decision_store.append_decision(record)
    
    # Credit spread closed:
    # Opening: +$186 (received)
    # Closing: -$295 (paid to close)
    # Net: $186 - $295 = -$109
    remaining = manager.get_remaining_capital("credit_spread")
    assert remaining == pytest.approx(9891.0, abs=1.0)  # 10000 + 186 - 295


def test_get_remaining_capital_with_closed_debit_position(allocations_config, decision_store):
    """Test remaining capital with closed debit position."""
    manager = CapitalManager(allocations_config, decision_store)
    
    # Create and close a debit position
    legs = (
        _make_option('OPT1', 500, '2025-09-06', 'call', 2.0),
        _make_option('OPT2', 505, '2025-09-06', 'call', 1.0),
    )
    proposal = ProposedPositionRequest(
        symbol='SPY',
        strategy_type=StrategyType.LONG_CALL,
        legs=legs,
        credit=1.50,  # Net debit paid
        width=5.0,
        probability_of_profit=0.6,
        confidence=0.65,
        expiration_date='2025-09-06',
        created_at=datetime.now().isoformat(),
        strategy_name='credit_spread',
    )
    record = DecisionResponse(
        id=generate_decision_id(proposal, datetime.now().isoformat()),
        proposal=proposal,
        outcome='accepted',
        decided_at=datetime.now().isoformat(),
        rationale='test',
        quantity=1,
        entry_price=1.50,
        exit_price=2.00,  # Received when closing
        closed_at=datetime.now().isoformat(),
    )
    decision_store.append_decision(record)
    
    # Debit position closed:
    # Opening: -$150 (paid)
    # Closing: +$200 (received)
    # Net: -$150 + $200 = +$50
    remaining = manager.get_remaining_capital("credit_spread")
    assert remaining == pytest.approx(10050.0, abs=1.0)  # 10000 - 150 + 200


def test_get_remaining_capital_multiple_positions(allocations_config, decision_store):
    """Test remaining capital with multiple positions."""
    manager = CapitalManager(allocations_config, decision_store)
    
    # Create multiple credit positions with different timestamps
    base_time = datetime.now()
    for i, credit in enumerate([1.0, 2.0]):
        # Use different timestamps to avoid collision
        decided_at = (base_time.replace(microsecond=i * 1000)).isoformat()
        legs = (
            _make_option(f'OPT{i}1', 500, '2025-09-06', 'put', 2.0),
            _make_option(f'OPT{i}2', 495, '2025-09-06', 'put', 1.0),
        )
        proposal = ProposedPositionRequest(
            symbol='SPY',
            strategy_type=StrategyType.PUT_CREDIT_SPREAD,
            legs=legs,
            credit=credit,
            width=5.0,
            probability_of_profit=0.6,
            confidence=0.65,
            expiration_date='2025-09-06',
            created_at=decided_at,
            strategy_name='credit_spread',
        )
        record = DecisionResponse(
            id=generate_decision_id(proposal, decided_at),
            proposal=proposal,
            outcome='accepted',
            decided_at=decided_at,
            rationale='test',
            quantity=1,
            entry_price=credit,
        )
        decision_store.append_decision(record)
    
    # Should have: 10000 + 100 + 200 = 10300
    remaining = manager.get_remaining_capital("credit_spread")
    assert remaining == 10300.0


def test_get_status_summary(capital_manager):
    """Test status summary generation."""
    summary = capital_manager.get_status_summary("credit_spread")
    assert "credit_spread" in summary
    assert "$10,000.00" in summary
    assert "$500.00" in summary  # max risk
    assert "5.0%" in summary


def test_check_risk_threshold_insufficient_capital(allocations_config, decision_store):
    """Test risk check when remaining capital is insufficient due to open positions."""
    manager = CapitalManager(allocations_config, decision_store)
    
    # Create a large debit position that uses most capital
    legs = (
        _make_option('OPT1', 500, '2025-09-06', 'call', 95.0),
        _make_option('OPT2', 505, '2025-09-06', 'call', 1.0),
    )
    proposal = ProposedPositionRequest(
        symbol='SPY',
        strategy_type=StrategyType.LONG_CALL,
        legs=legs,
        credit=98.0,  # Large debit paid
        width=5.0,
        probability_of_profit=0.6,
        confidence=0.65,
        expiration_date='2025-09-06',
        created_at=datetime.now().isoformat(),
        strategy_name='credit_spread',
    )
    record = DecisionResponse(
        id=generate_decision_id(proposal, datetime.now().isoformat()),
        proposal=proposal,
        outcome='accepted',
        decided_at=datetime.now().isoformat(),
        rationale='test',
        quantity=1,
        entry_price=98.0,
    )
    decision_store.append_decision(record)
    
    # Remaining capital should be 10000 - 9800 = 200
    # Max allowed risk is now 5% of remaining = 0.05 * 200 = 10
    # A new position with max_risk 600 should fail (exceeds both max allowed and remaining)
    is_allowed, message = manager.check_risk_threshold("credit_spread", 600.0)
    assert is_allowed is False
    assert "exceeds" in message

