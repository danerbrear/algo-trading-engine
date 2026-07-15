from enum import Enum


class UniversalCloseCondition(Enum):
    """Close conditions applied by the engine after strategy.on_new_date."""

    PROFIT_TARGET = "profit_target"
    STOP_LOSS = "stop_loss"
    ASSIGNMENT = "assignment"
