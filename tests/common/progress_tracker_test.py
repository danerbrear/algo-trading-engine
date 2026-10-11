"""Unit tests for ProgressTracker observer integration."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

from algo_trading_engine._internal.common.progress_tracker import ProgressTracker


@dataclass
class RecordingObserver:
    progress_events: list[tuple[int, int, str]] = field(default_factory=list)

    def log(self, level: str, message: str) -> None:
        pass

    def progress(self, current: int, total: int, label: str = "") -> None:
        self.progress_events.append((current, total, label))

    def result(self, stats: dict[str, str]) -> None:
        pass


class TestProgressTrackerObserver:
    def test_disables_tqdm_when_observer_present(self) -> None:
        observer = RecordingObserver()
        tracker = ProgressTracker(total_dates=10, observer=observer)
        assert tracker.pbar.disable is True
        tracker.close()

    def test_forwards_progress_to_observer(self) -> None:
        observer = RecordingObserver()
        tracker = ProgressTracker(total_dates=4, unit="bar", observer=observer)
        tracker.update(current_date=datetime(2025, 1, 1, 10, 0))
        tracker.update(current_date=datetime(2025, 1, 1, 11, 0))
        tracker.close()
        assert len(observer.progress_events) == 2
        current, total, label = observer.progress_events[0]
        assert current == 1
        assert total == 4
        assert "Processing" in label
