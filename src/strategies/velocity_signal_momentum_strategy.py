from datetime import datetime
from typing import Callable

from src.backtest.models import Strategy, Position
from src.common.progress_tracker import progress_print

class VelocitySignalMomentumStrategy(Strategy):
    """
    A momentum strategy to trade credit spreads in order to capitalize on 
    the upward or downward trends. 
    """

    def __init__(self):
        super().__init__(start_date_offset=60)

    def on_new_date(self, date: datetime, positions: tuple['Position', ...], add_position: Callable[['Position'], None], remove_position: Callable[['Position'], None]):
        super().on_new_date(date, positions)

        if len(positions) == 0:
            # Determine if we should open a new position
            pass
        else:
            # Check if we should close any positions
            pass

    def on_end(self, positions: tuple['Position', ...], remove_position: Callable[['Position'], None], date: datetime):
        pass

    def _has_buy_signal(self, date: datetime) -> bool:
        """
        If the data matches the following criteria for a buy signal, return True:
            - Price must increase over the trend period
            - No significant reversals (>2% drop) during the trend
            - Trend must last at least 3 days
            - Trend must not exceed 60 days
        """
        return False

    def _determine_expiration_date(self, date: datetime) -> datetime:
        """
        Find an expiration date by looking for the highest risk weighted return (Sharpe ratio)
        for an ATM/+10 put credit spread for each daily option chain. Use a 5-40 day range.
        """
        pass

    def _calculate_sharpe_ratio(self, position: Position) -> float:
        """
        Calculate the Sharpe ratio for a position.
        """
        pass
