from typing import Callable

class Strategy:
    """
    Strategy is a class that represents a trading strategy.
    """

    def __init__(self, profit_target: float = None, stop_loss: float = None):
        self.profit_target = profit_target
        self.stop_loss = stop_loss

    def set_profit_target(self, profit_target: float):
        """
        Set the profit target for the strategy.
        """
        self.profit_target = profit_target

    def set_stop_loss(self, stop_loss: float):
        """
        Set the stop loss for the strategy.
        """
        self.stop_loss = stop_loss
    
    def on_new_date(self, positions: tuple['Position', ...], add_position: Callable[['Position'], None], remove_position: Callable[['Position'], None]):
        """
        On new date, execute strategy.
        """
        pass

class CreditSpreadStrategy(Strategy):
    """
    CreditSpreadStrategy is a class that represents a credit spread strategy.

    Stop Loss: 60%
    """

    def __init__(self):
        super().__init__(stop_loss=0.6)

    def on_new_date(self, positions: tuple['Position', ...], add_position: Callable[['Position'], None], remove_position: Callable[['Position'], None]):
        """
        On new date, determine if a new position should be opened. We should not open a position if we already have one.
        """

        if len(positions) == 0:
            # Determine if we should open a new position
            pass
        else:
            # Determine if we should close a position
            pass

class Position:
    """
    Position is a class that represents a position in a stock.
    """

    def __init__(self, ticker: str, quantity: int, entry_price: float, exit_price: float = None):
        self.ticker = ticker
        self.quantity = quantity
        self.entry_price = entry_price
        self.exit_price = exit_price
