from time import time
from tqdm import tqdm
from typing import Dict, Any
from datetime import datetime, timedelta
import sys

class ProgressTracker:
    def __init__(self, start_date: datetime, end_date: datetime, desc: str = "Processing"):
        self.start_time = time()
        self.processed_dates = 0
        self.successful_api_calls = 0
        self.total_dates = self._count_trading_days(start_date, end_date)
        self.total_api_calls = self.total_dates * 5  # 5 API calls per date
        
        # Initialize progress bar at the bottom
        self.pbar = tqdm(
            total=self.total_dates,
            desc=desc,
            unit="date",
            position=0,
            leave=True,
            bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]',
            dynamic_ncols=True,
            miniters=1,  # Update at least every iteration
            file=sys.stdout,  # Ensure we're writing to stdout
            mininterval=0.1  # Update at least every 0.1 seconds
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
        """Write a message above the progress bar"""
        # Clear the progress bar
        self.pbar.clear()
        # Write the message
        tqdm.write(message)
        # Refresh the progress bar
        self.pbar.refresh()
    
    def update(self, current_date: datetime = None, additional_info: Dict[str, Any] = None, increment_operations: int = 0):
        """Update progress with current date and additional information"""
        if current_date:
            self.processed_dates += 1
            
            # Calculate progress percentages
            date_progress = (self.processed_dates / self.total_dates) * 100
            api_progress = (self.successful_api_calls / self.total_api_calls) * 100
            
            # Update progress bar description
            desc = f"Processing {current_date.date()} ({date_progress:.1f}% dates, {api_progress:.1f}% API calls)"
            self.pbar.set_description(desc)
            
            # Update postfix with additional info
            if additional_info:
                self.pbar.set_postfix(additional_info)
        
        if increment_operations:
            self.successful_api_calls += increment_operations
        
        # Update progress bar with the correct percentage
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
        stats = self.get_progress_stats()
        self.pbar.close()
        
        self.write(f"\nProcessing completed:")
        self.write(f"Total time: {timedelta(seconds=int(stats['elapsed_time']))}")
        self.write(f"Average time per date: {stats['avg_time_per_date']:.2f} seconds")
        self.write(f"Total successful API calls: {stats['successful_api_calls']}") 