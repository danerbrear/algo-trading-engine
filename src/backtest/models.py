from typing import Callable
from datetime import datetime
import pandas as pd

class Strategy:
    """
    Strategy is a class that represents a trading strategy.
    """

    def __init__(self, profit_target: float = None, stop_loss: float = None):
        self.profit_target = profit_target
        self.stop_loss = stop_loss
        self.data = None

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

    def set_data(self, data: pd.DataFrame):
        """
        Set the data for the strategy.
        """
        self.data = data

    def on_new_date(self, date: datetime, positions: tuple['Position', ...], add_position: Callable[['Position'], None], remove_position: Callable[['Position'], None]):
        """
        On new date, execute strategy.
        """
        pass

    def _profit_target_hit(self, position: 'Position') -> bool:
        """
        Check if the profit target has been hit for a position.
        """
        if self.profit_target is None:
            return False
        return position.profit_target_hit(self.profit_target)
    
    def _stop_loss_hit(self, position: 'Position') -> bool:
        """
        Check if the stop loss has been hit for a position.
        """
        if self.stop_loss is None:
            return False
        return position.stop_loss_hit(self.stop_loss)
    
    def on_end(self, positions: tuple['Position', ...]):
        """
        On end, execute strategy.
        """
        pass

class CreditSpreadStrategy(Strategy):
    """
    CreditSpreadStrategy is a class that represents a credit spread strategy.

    Stop Loss: 60%
    """

    holding_period = 10

    def __init__(self, lstm_model):
        super().__init__(stop_loss=0.6)
        self.lstm_model = lstm_model

    def on_new_date(self, date: datetime, positions: tuple['Position', ...], add_position: Callable[['Position'], None], remove_position: Callable[['Position'], None]):
        """
        On new date, determine if a new position should be opened. We should not open a position if we already have one.
        """

        if len(positions) == 0:
            # Determine if we should open a new position
            print(f"No positions, opening new position")
            pass
        else:
            for position in positions:
                # Determine if we should close a position
                if (self._profit_target_hit(positions[0]) or self._stop_loss_hit(positions[0])):
                    print(f"Profit target or stop loss hit for {positions[0].ticker}")
                    remove_position(positions[0])
                elif position.get_days_to_expiration(date) < self.holding_period:
                    print(f"Position {position.ticker} expired or near expiration")
                    remove_position(position)
    
    def _make_prediction(self, date: datetime):
        """
        Make a prediction for a given date.
        """
        pass

class Position:
    """
    Position is a class that represents a position in a stock.
    """

    def __init__(self, ticker: str, quantity: int, entry_date: datetime, entry_price: float, exit_price: float = None):
        self.ticker = ticker
        self.quantity = quantity
        self.entry_date = entry_date
        self.entry_price = entry_price
        self.exit_price = exit_price
        
        # Initialize option-related properties
        self.symbol = None
        self.expiration_date = None
        self.option_type = None
        self.strike_price = None
        self.days_to_expiration = None
        self.is_option = False
        
        # Try to parse the ticker as an options ticker
        try:
            parsed_info = self.parse_options_ticker()
            self.symbol = parsed_info['symbol']
            self.expiration_date = parsed_info['expiration_date']
            self.option_type = parsed_info['option_type']
            self.strike_price = parsed_info['strike_price']
            self.days_to_expiration = parsed_info['days_to_expiration']
            self.is_option = True
        except ValueError:
            # Not an options ticker, leave properties as None
            pass
    
    def profit_target_hit(self, profit_target: float) -> bool:
        """
        Check if the profit target has been hit for a position.
        """
        return self._get_return() >= profit_target
    
    def stop_loss_hit(self, stop_loss: float) -> bool:
        """
        Check if the stop loss has been hit for a position.
        """
        return self._get_return() <= -stop_loss
    
    def get_days_to_expiration(self, current_date: datetime) -> int:
        """
        Get the number of days to expiration for a position from the given current_date.
        """
        if self.expiration_date is not None:
            return (self.expiration_date - current_date).days
        else:
            raise ValueError("Expiration date is not set")
        
    def get_return_dollars(self) -> float:
        """
        Get the return in dollars for a position.
        """
        return (self.exit_price * self.quantity * 100) - (self.entry_price * self.quantity * 100)
    
    def _get_return(self) -> float:
        """
        Get the percentage return for a position.
        """
        return ((self.exit_price * self.quantity * 100) - (self.entry_price * self.quantity * 100)) / (self.entry_price * self.quantity * 100)
    
    def parse_options_ticker(self, ticker: str = None) -> dict:
        """
        Parse an options ticker into its components.
        
        Supports multiple formats:
        - OCC format: SPY240119C00500000
        - Yahoo Finance format: SPY-240119-C-500
        - Interactive Brokers format: SPY 240119 C 500
        
        Args:
            ticker: Options ticker to parse (uses self.ticker if None)
            
        Returns:
            dict: Dictionary with keys:
                - symbol: Stock symbol (e.g., 'SPY')
                - expiration_date: datetime object
                - option_type: 'C' for call, 'P' for put
                - strike_price: float strike price
                - days_to_expiration: int days until expiration
                
        Raises:
            ValueError: If ticker format is not recognized
        """
        if ticker is None:
            ticker = self.ticker
        
        if not ticker:
            raise ValueError("Ticker cannot be empty")
        
        # Try Yahoo Finance format (e.g., SPY-240119-C-500)
        if '-' in ticker and len(ticker.split('-')) == 4:
            return self._parse_yahoo_format(ticker)
        # Try Interactive Brokers format (e.g., SPY 240119 C 500)
        elif ' ' in ticker and len(ticker.split()) == 4:
            return self._parse_ib_format(ticker)
        # Try OCC format (e.g., SPY240119C00500000) only if no '-' or spaces
        elif len(ticker) >= 15 and '-' not in ticker and ' ' not in ticker:
            # Find where the date starts (after symbol)
            for i, char in enumerate(ticker):
                if char.isdigit():
                    # Check if we have exactly 6 digits for date, followed by C/P, then 8 digits for strike
                    if (i + 6 < len(ticker) and 
                        ticker[i:i+6].isdigit() and 
                        ticker[i+6] in ['C', 'P'] and 
                        i + 15 <= len(ticker) and 
                        ticker[i+7:i+15].isdigit()):
                        # Additional validation: ensure the option type is actually at the correct position
                        # and that we have exactly 8 digits for the strike price
                        if len(ticker) == i + 15:  # Total length should be symbol + 6 digits + 1 char + 8 digits
                            return self._parse_occ_format(ticker)
                    # Do not break here; continue searching for a valid OCC pattern
            raise ValueError(f"Unrecognized ticker format: {ticker}")
        # If no format matched, raise error
        raise ValueError(f"Unrecognized ticker format: {ticker}")
    
    def _parse_occ_format(self, ticker: str) -> dict:
        print(f"[DEBUG] Entered _parse_occ_format with ticker: {ticker}")
        """Parse OCC format ticker (e.g., SPY240119C00500000)"""
        # Find the position where the date starts (after symbol)
        symbol_end = 0
        for i, char in enumerate(ticker):
            if char.isdigit():
                symbol_end = i
                break
        
        symbol = ticker[:symbol_end]
        date_str = ticker[symbol_end:symbol_end+6]  # YYMMDD
        option_type = ticker[symbol_end+6]  # C or P
        if option_type not in ['C', 'P']:
            print(f"[DEBUG] Invalid option type detected: {option_type}")
            raise ValueError(f"Invalid option type in OCC ticker: {option_type}")
        strike_str = ticker[symbol_end+7:]  # 8-digit strike (in thousands)
        
        # Convert strike from thousands to actual price
        strike_price = float(strike_str) / 1000
        
        # Parse date (YYMMDD format)
        year = 2000 + int(date_str[:2])
        month = int(date_str[2:4])
        day = int(date_str[4:6])
        expiration_date = datetime(year, month, day)
        
        return {
            'symbol': symbol,
            'expiration_date': expiration_date,
            'option_type': option_type,
            'strike_price': strike_price,
            'days_to_expiration': (expiration_date - datetime.now()).days
        }
    
    def _parse_yahoo_format(self, ticker: str) -> dict:
        """Parse Yahoo Finance format ticker (e.g., SPY-240119-C-500)"""
        parts = ticker.split('-')
        symbol = parts[0]
        date_str = parts[1]  # YYMMDD
        option_type = parts[2]  # C or P
        strike_price = float(parts[3])
        
        # Parse date (YYMMDD format)
        year = 2000 + int(date_str[:2])
        month = int(date_str[2:4])
        day = int(date_str[4:6])
        expiration_date = datetime(year, month, day)
        
        return {
            'symbol': symbol,
            'expiration_date': expiration_date,
            'option_type': option_type,
            'strike_price': strike_price,
            'days_to_expiration': (expiration_date - datetime.now()).days
        }
    
    def _parse_ib_format(self, ticker: str) -> dict:
        """Parse Interactive Brokers format ticker (e.g., SPY 240119 C 500)"""
        parts = ticker.split()
        symbol = parts[0]
        date_str = parts[1]  # YYMMDD
        option_type = parts[2]  # C or P
        strike_price = float(parts[3])
        
        # Parse date (YYMMDD format)
        year = 2000 + int(date_str[:2])
        month = int(date_str[2:4])
        day = int(date_str[4:6])
        expiration_date = datetime(year, month, day)
        
        return {
            'symbol': symbol,
            'expiration_date': expiration_date,
            'option_type': option_type,
            'strike_price': strike_price,
            'days_to_expiration': (expiration_date - datetime.now()).days
        }
    