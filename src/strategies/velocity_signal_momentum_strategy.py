from src.backtest.models import Strategy

class VelocitySignalMomentumStrategy(Strategy):
    def __init__(self, symbol: str = 'SPY'):
        super().__init__(start_date_offset=60)

    def run(self):
        pass