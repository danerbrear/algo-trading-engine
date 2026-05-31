"""
Implied forward variance and volatility term-structure analytics.

Pure functions for strategy/backtest use. Data fetching remains in OptionsHandler.
"""

from datetime import date
from typing import List, Sequence, Union

from algo_trading_engine.dto.volatility_term_structure_dtos import (
    ImpliedForwardVarianceTermStructureDTO,
)
from algo_trading_engine.enums.term_structure_type import TermStructureType

DAYS_PER_YEAR = 365.0

MaturityInput = Union[int, float, date]


def dte_to_years(dte_days: float) -> float:
    """Convert days-to-expiration to time in years."""
    if dte_days < 0:
        raise ValueError("DTE must be non-negative")
    return dte_days / DAYS_PER_YEAR


def total_implied_variance(time_years: float, implied_volatility: float) -> float:
    """Total implied variance for a single maturity: T * sigma(T)^2."""
    if time_years < 0:
        raise ValueError("Time in years must be non-negative")
    if implied_volatility < 0:
        raise ValueError("Implied volatility must be non-negative")
    return time_years * implied_volatility ** 2


def implied_forward_variance_between_maturities(
    t1_years: float,
    iv1: float,
    t2_years: float,
    iv2: float,
) -> float:
    """
    Implied forward variance between two maturities.

    f(T1, T2) = (T2 * sigma(T2)^2 - T1 * sigma(T1)^2) / (T2 - T1)
    """
    if t2_years <= t1_years:
        raise ValueError("T2 must be greater than T1")
    numerator = total_implied_variance(t2_years, iv2) - total_implied_variance(t1_years, iv1)
    return numerator / (t2_years - t1_years)


def _maturity_to_years(maturity: MaturityInput, current_date: date) -> float:
    if isinstance(maturity, date):
        dte_days = (maturity - current_date).days
        if dte_days < 0:
            raise ValueError("Maturity date cannot be before current_date")
        return dte_to_years(dte_days)
    return dte_to_years(float(maturity))


def _classify_term_structure(
    forward_variance: float,
    earlier_implied_volatility: float,
) -> TermStructureType:
    earlier_variance = earlier_implied_volatility ** 2
    if forward_variance > earlier_variance:
        return TermStructureType.CONTANGO
    return TermStructureType.BACKWARDATION


def compute_implied_forward_variance_term_structure(
    implied_volatilities: Sequence[float],
    maturities: Sequence[MaturityInput],
    current_date: date,
) -> ImpliedForwardVarianceTermStructureDTO:
    """
    Compute implied forward variance and term-structure type for each maturity step.

    For each pair of consecutive inputs (T_i, sigma_i) -> (T_{i+1}, sigma_{i+1}),
    computes forward variance and classifies:
    - contango when f(T_i, T_{i+1}) > sigma(T_i)^2
    - backwardation otherwise

    Args:
        implied_volatilities: IV at each maturity point (decimal, e.g. 0.20 for 20%).
        maturities: Parallel DTE in days (int/float) or expiration dates.
        current_date: Reference date for DTE when maturities are dates.

    Returns:
        ImpliedForwardVarianceTermStructureDTO with one entry per step after the first.
    """
    if not isinstance(current_date, date):
        raise ValueError("current_date must be a datetime.date")

    ivs = list(implied_volatilities)
    mats = list(maturities)

    if len(ivs) != len(mats):
        raise ValueError("implied_volatilities and maturities must have the same length")
    if len(ivs) < 2:
        raise ValueError("At least two IV/maturity points are required")

    times_years = [_maturity_to_years(m, current_date) for m in mats]

    for i, t in enumerate(times_years):
        if i > 0 and t <= times_years[i - 1]:
            raise ValueError("Maturities must be strictly increasing in time (years)")

    forward_variances: List[float] = []
    term_structure_types: List[TermStructureType] = []

    for i in range(1, len(ivs)):
        forward_var = implied_forward_variance_between_maturities(
            times_years[i - 1],
            ivs[i - 1],
            times_years[i],
            ivs[i],
        )
        forward_variances.append(forward_var)
        term_structure_types.append(
            _classify_term_structure(forward_var, ivs[i - 1])
        )

    return ImpliedForwardVarianceTermStructureDTO(
        implied_forward_variances=forward_variances,
        term_structure_types=term_structure_types,
    )
