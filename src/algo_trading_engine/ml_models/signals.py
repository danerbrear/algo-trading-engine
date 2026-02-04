"""
LSTM signal types.

The LSTM model was designed to output exactly three classes:
- HOLD (no trade)
- CALL_CREDIT_SPREAD
- PUT_CREDIT_SPREAD

Do not add new cases without retraining the model. This enum lives in ml_models
to reflect that it is tied to the LSTM's output layer.
"""

from enum import Enum


class SignalType(Enum):
    """LSTM output signal types (3 classes only)."""
    HOLD = "hold"
    CALL_CREDIT_SPREAD = "call_credit_spread"
    PUT_CREDIT_SPREAD = "put_credit_spread"


# LSTM uses integer labels: 0=HOLD, 1=CALL_CREDIT_SPREAD, 2=PUT_CREDIT_SPREAD
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
