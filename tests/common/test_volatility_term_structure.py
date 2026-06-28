"""Tests for implied forward variance term structure."""

from datetime import date

import pytest

from algo_trading_engine.common.volatility_term_structure import (
    compute_implied_forward_variance_term_structure,
)
from algo_trading_engine.enums import TermStructureType


def test_compute_implied_forward_variance_term_structure_real_numbers():
    """SPY-style IV term structure: 30/60/90 DTE with rising then falling IV."""
    current_date = date(2025, 1, 1)
    implied_volatilities = [0.20, 0.22, 0.18]
    maturities_dte_days = [30, 60, 90]

    result = compute_implied_forward_variance_term_structure(
        implied_volatilities=implied_volatilities,
        maturities=maturities_dte_days,
        current_date=current_date,
    )

    assert len(result.implied_forward_variances) == 2
    assert len(result.term_structure_types) == 2

    assert result.implied_forward_variances[0] == pytest.approx(0.0568)
    assert result.implied_forward_variances[1] == pytest.approx(0.0004)

    assert result.term_structure_types[0] == TermStructureType.CONTANGO
    assert result.term_structure_types[1] == TermStructureType.BACKWARDATION
