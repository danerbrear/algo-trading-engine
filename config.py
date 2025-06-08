# Model Training Parameters
EPOCHS = 50
BATCH_SIZE = 16
VALIDATION_SPLIT = 0.20
SEQUENCE_LENGTH = 60

# Model Architecture Parameters
LSTM_UNITS = [64, 32]
DENSE_UNITS = [32]
DROPOUT_RATE = 0.3

# Training Parameters
LEARNING_RATE = 0.0005
EARLY_STOPPING_PATIENCE = 8
EARLY_STOPPING_MIN_DELTA = 0.001

# Enhanced Strategy Parameters
STRATEGY_MIN_RETURN_THRESHOLD = 0.10  # 10% minimum return threshold
STRATEGY_HOLDING_PERIOD = 5  # Days to hold strategy
VOLATILITY_LOOKBACK = 60  # Days for volatility percentile calculation

# Strategy Risk Parameters
CALL_DEBIT_SPREAD_PROFIT_RATIO = 0.7  # % of call return kept after short leg
PUT_DEBIT_SPREAD_PROFIT_RATIO = 0.7   # % of put return kept after short leg
IRON_CONDOR_LOW_VOL_THRESHOLD = 0.3   # Volatility percentile threshold for Iron Condor
STRADDLE_HIGH_VOL_THRESHOLD = 0.6     # Volatility percentile threshold for Long Straddle
STRONG_TREND_THRESHOLD = 0.02         # Stock return threshold for strong trends 