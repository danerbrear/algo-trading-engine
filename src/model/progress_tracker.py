from time import time
from tqdm import tqdm
from typing import Dict, Any, Optional
from datetime import datetime, timedelta
import sys
import threading

class ProgressTracker:
    def __init__(self, start_date: datetime, end_date: datetime, desc: str = "Processing", quiet_mode: bool = True):
        self.start_time = time()
        self.processed_dates = 0
        self.successful_api_calls = 0
        self.total_dates = self._count_trading_days(start_date, end_date)
        self.total_api_calls = self.total_dates * 5  # 5 API calls per date
        self._lock = threading.Lock()
        self.quiet_mode = quiet_mode
        
        # Initialize progress bar with improved settings
        self.pbar = tqdm(
            total=self.total_dates,
            desc=desc,
            unit="date",
            position=0,
            leave=True,
            bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]',
            dynamic_ncols=True,
            miniters=1,
            file=sys.stdout,
            mininterval=0.1,
            ncols=100,  # Fixed width to prevent jumping
            ascii=False,  # Use Unicode characters for better display
            smoothing=0.1  # Smooth ETA calculations
        )
        
    def _count_trading_days(self, start_date: datetime, end_date: datetime) -> int:
        """Count number of trading days between two dates (excluding weekends)"""
        days = 0
        current_date = start_date
        while current_date <= end_date:
            if current_date.weekday() < 5:  # 0-4 are weekdays
                days += 1
            current_date += timedelta(days=1)
        return days
    
    def write(self, message: str):
        """Write a message above the progress bar with thread safety"""
        with self._lock:
            # Clear current line and move cursor up to avoid conflicts
            self.pbar.clear()
            # Write the message using tqdm.write to maintain bar position
            tqdm.write(message, file=sys.stdout)
            # Force refresh the progress bar to redraw it at the bottom
            self.pbar.refresh()
    
    def update(self, current_date: datetime = None, additional_info: Dict[str, Any] = None, increment_operations: int = 0, summary_info: str = ""):
        """Update progress with current date and additional information"""
        with self._lock:
            if current_date:
                self.processed_dates += 1
                
                # Calculate progress percentages
                date_progress = (self.processed_dates / self.total_dates) * 100
                api_progress = (self.successful_api_calls / self.total_api_calls) * 100
                
                # Update progress bar description with compact format
                desc = f"Processing {current_date.date()} ({date_progress:.1f}% dates, {api_progress:.1f}% API calls)"
                self.pbar.set_description(desc)
                
                # Show summary info in postfix (key strike prices, etc.)
                if summary_info:
                    self.pbar.set_postfix_str(summary_info)
                elif additional_info:
                    # Create a compact postfix
                    postfix_str = f"calls: {additional_info.get('calls', 0)}, puts: {additional_info.get('puts', 0)}"
                    self.pbar.set_postfix_str(postfix_str)
            
            if increment_operations:
                self.successful_api_calls += increment_operations
            
            # Update progress bar position
            self.pbar.n = self.processed_dates
            self.pbar.refresh()
    
    def get_progress_stats(self) -> Dict[str, Any]:
        """Get current progress statistics"""
        elapsed_time = time() - self.start_time
        avg_time_per_date = elapsed_time / self.processed_dates if self.processed_dates > 0 else 0
        
        return {
            'elapsed_time': elapsed_time,
            'avg_time_per_date': avg_time_per_date,
            'processed_dates': self.processed_dates,
            'total_dates': self.total_dates,
            'successful_api_calls': self.successful_api_calls,
            'total_api_calls': self.total_api_calls
        }
    
    def close(self):
        """Close progress bar and print final summary"""
        with self._lock:
            stats = self.get_progress_stats()
            
            # Close the progress bar properly
            self.pbar.close()
            
            # Print final summary
            print(f"\n✅ Processing completed:")
            print(f"   Total time: {timedelta(seconds=int(stats['elapsed_time']))}")
            print(f"   Average time per date: {stats['avg_time_per_date']:.2f} seconds")
            print(f"   Total successful API calls: {stats['successful_api_calls']}")

# Global progress tracker instance for use across modules
_global_progress_tracker: Optional[ProgressTracker] = None
_tracker_lock = threading.Lock()

def set_global_progress_tracker(tracker: Optional[ProgressTracker]):
    """Set the global progress tracker instance"""
    global _global_progress_tracker
    with _tracker_lock:
        _global_progress_tracker = tracker

def get_global_progress_tracker() -> Optional[ProgressTracker]:
    """Get the global progress tracker instance"""
    global _global_progress_tracker
    with _tracker_lock:
        return _global_progress_tracker

def progress_print(message: str, force: bool = False):
    """Print a message through the progress tracker if available, otherwise use regular print"""
    tracker = get_global_progress_tracker()
    if tracker:
        # In quiet mode, suppress most messages unless forced
        if tracker.quiet_mode and not force:
            return  # Suppress message in quiet mode
        tracker.write(message)
    else:
        print(message)

def is_quiet_mode() -> bool:
    """Check if we're in quiet mode"""
    tracker = get_global_progress_tracker()
    return tracker is not None and tracker.quiet_mode 