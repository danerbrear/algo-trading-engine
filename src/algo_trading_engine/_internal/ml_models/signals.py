"""
LSTM label mapping for SignalType.

The LSTM model outputs exactly three classes mapped to SignalType in enums.
"""

from algo_trading_engine.enums import SignalType

_LSTM_LABEL_TO_SIGNAL = {
    0: SignalType.HOLD,
    1: SignalType.CALL_CREDIT_SPREAD,
    2: SignalType.PUT_CREDIT_SPREAD,
}

_SIGNAL_TO_LSTM_LABEL = {v: k for k, v in _LSTM_LABEL_TO_SIGNAL.items()}


def signal_from_lstm_label(label: int) -> SignalType:
    """Map LSTM integer output (0, 1, 2) to SignalType."""
    if label not in _LSTM_LABEL_TO_SIGNAL:
        raise ValueError(f"Invalid LSTM label: {label}. Expected 0, 1, or 2.")
    return _LSTM_LABEL_TO_SIGNAL[label]


def lstm_label_from_signal(signal_type: SignalType) -> int:
    """Map SignalType to LSTM integer label (0, 1, 2)."""
    return _SIGNAL_TO_LSTM_LABEL[signal_type]
