"""Strategy._update_positions refreshes the quote stored on each open leg."""
from datetime import datetime
from decimal import Decimal

import pandas as pd

from algo_trading_engine.backtest.main import BacktestEngine
from algo_trading_engine.dto import OptionBarDTO
from algo_trading_engine.enums import BarTimeInterval, OptionType, StrategyType
from algo_trading_engine.strategy import Strategy
from algo_trading_engine.vo import Option, create_position


def _bar(ticker: str, close_price: str, volume: int, timestamp: datetime) -> OptionBarDTO:
    price = Decimal(close_price)
    return OptionBarDTO(
        ticker=ticker,
        timestamp=timestamp,
        open_price=price,
        high_price=price,
        low_price=price,
        close_price=price,
        volume=volume,
        volume_weighted_avg_price=price,
        number_of_transactions=1,
    )


class _RecordingStrategy(Strategy):
    def __init__(self):
        super().__init__()
        self.bars = {}

    def on_new_date(self, date, positions, add_position, remove_position):
        super().on_new_date(date, positions, add_position, remove_position)

    def on_end(self, positions, remove_position, date):
        pass

    def validate_data(self, data):
        return True

    def get_option_bar(self, contract, date, multiplier=1, timespan=BarTimeInterval.DAY):
        del multiplier
        return self.bars.get((contract.ticker, date, timespan))


def _leg(ticker: str, strike: float) -> Option:
    return Option(
        ticker=ticker,
        timestamp=datetime(2024, 1, 15),
        symbol="SPY",
        strike=strike,
        expiration="2024-03-15",
        option_type=OptionType.CALL,
        last_price=1.0,
        volume=10,
    )


def test_update_positions_replaces_price_timestamp_and_volume():
    strategy = _RecordingStrategy()
    strategy.option_bar_timespan = BarTimeInterval.HOUR
    as_of = datetime(2024, 2, 15, 10, 30)
    short_leg = _leg("O:SPY240315C00500000", 500.0)
    long_leg = _leg("O:SPY240315C00510000", 510.0)
    position = create_position(
        symbol="SPY",
        expiration_date=datetime(2024, 3, 15),
        strategy_type=StrategyType.CALL_CREDIT_SPREAD,
        strike_price=500.0,
        entry_date=datetime(2024, 1, 15),
        entry_price=0.75,
        legs=[short_leg, long_leg],
    )
    strategy.bars[(short_leg.ticker, as_of, BarTimeInterval.HOUR)] = _bar(
        short_leg.ticker, "1.80", 40, as_of
    )
    strategy.bars[(long_leg.ticker, as_of, BarTimeInterval.HOUR)] = _bar(
        long_leg.ticker, "0.40", 12, as_of
    )

    assert strategy._update_positions(as_of, (position,)) is True

    refreshed_short, refreshed_long = position.legs
    assert refreshed_short.last_price == 1.80
    assert refreshed_short.volume == 40
    assert refreshed_short.timestamp == as_of
    assert refreshed_short.strike == 500.0
    assert refreshed_long.last_price == 0.40
    assert refreshed_long.volume == 12
    assert refreshed_long is not long_leg


def test_update_positions_keeps_previous_quote_when_bar_is_missing():
    strategy = _RecordingStrategy()
    as_of = datetime(2024, 2, 15)
    leg = _leg("O:SPY240315C00500000", 500.0)
    position = create_position(
        symbol="SPY",
        expiration_date=datetime(2024, 3, 15),
        strategy_type=StrategyType.LONG_CALL,
        strike_price=500.0,
        entry_date=datetime(2024, 1, 15),
        entry_price=1.0,
        legs=[leg],
    )

    assert strategy._update_positions(as_of, (position,)) is False
    assert position.legs[0] is leg
    assert position.legs[0].last_price == 1.0


def test_compute_exit_price_uses_refreshed_leg_quotes():
    strategy = _RecordingStrategy()
    strategy.symbol = "SPY"
    as_of = datetime(2024, 2, 15)
    short_leg = _leg("O:SPY240315C00500000", 500.0)
    long_leg = _leg("O:SPY240315C00510000", 510.0)
    position = create_position(
        symbol="SPY",
        expiration_date=datetime(2024, 3, 15),
        strategy_type=StrategyType.CALL_CREDIT_SPREAD,
        strike_price=500.0,
        entry_date=datetime(2024, 1, 15),
        entry_price=0.75,
        legs=[short_leg, long_leg],
    )
    strategy.bars[(short_leg.ticker, as_of, BarTimeInterval.DAY)] = _bar(
        short_leg.ticker, "1.25", 20, as_of
    )
    strategy.bars[(long_leg.ticker, as_of, BarTimeInterval.DAY)] = _bar(
        long_leg.ticker, "0.25", 20, as_of
    )
    dates = pd.date_range("2024-02-15", periods=1)
    data = pd.DataFrame({"Close": [505.0]}, index=dates)
    engine = BacktestEngine(
        data=data,
        strategy=strategy,
        initial_capital=10000,
        start_date=as_of,
        end_date=as_of,
        enable_progress_tracking=False,
        quiet_mode=True,
    )

    assert strategy._update_positions(as_of, (position,)) is True
    assert engine.compute_exit_price(position, as_of) == 1.0
