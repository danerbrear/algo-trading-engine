import time
import requests
from typing import Callable, Any


class APIRetryHandler:
    """Handles API retries and rate limiting for Polygon.io requests"""
    
    def __init__(self, rate_limit_delay: int = 13, use_rate_limit: bool = True):
        """Initialize the retry handler
        
        Args:
            rate_limit_delay: Delay in seconds between API calls for rate limiting
            use_rate_limit: Whether to apply rate limiting delays
        """
        self.rate_limit_delay = rate_limit_delay
        self.use_rate_limit = use_rate_limit
    
    def fetch_with_retry(self, fetch_func: Callable[[], Any], error_msg: str, max_retries: int = 3, retry_delay: int = 60) -> Any:
        """Generic retry mechanism for API calls that may hit rate limits
        
        Args:
            fetch_func: Function to execute that makes the API call
            error_msg: Error message to display if all retries fail
            max_retries: Maximum number of retry attempts
            retry_delay: Delay in seconds between retry attempts
            
        Returns:
            Result from the fetch_func if successful
            
        Raises:
            Exception: If all retries are exhausted or non-recoverable error occurs
        """
        # Rate limiting for free tier: 5 calls per minute = 12 seconds between calls
        # Adding 13 seconds to be safe and stay under the limit
        if self.use_rate_limit:
            print(f"🕐 Free tier rate limiting: waiting {self.rate_limit_delay} seconds...")
            time.sleep(self.rate_limit_delay)
        
        for attempt in range(max_retries):
            if attempt > 0:
                print(f"Waiting {retry_delay} seconds before retry {attempt + 1}/{max_retries}")
                time.sleep(retry_delay)
                
            try:
                return fetch_func()
            except requests.exceptions.HTTPError as e:
                if e.response.status_code == 429:  # Rate limit exceeded
                    if attempt < max_retries - 1:
                        print("Rate limit exceeded, will retry after delay")
                        continue
                    print(f"Max retries ({max_retries}) exceeded for rate limit")
                    raise
                print(f"HTTP error occurred: {str(e)}")
                raise
            except Exception as e:
                raise e