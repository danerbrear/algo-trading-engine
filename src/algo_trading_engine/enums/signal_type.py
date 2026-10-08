from enum import Enum


class SignalType(Enum):
    """LSTM output signal types (3 classes only)."""

    HOLD = "hold"
    CALL_CREDIT_SPREAD = "call_credit_spread"
    PUT_CREDIT_SPREAD = "put_credit_spread"
