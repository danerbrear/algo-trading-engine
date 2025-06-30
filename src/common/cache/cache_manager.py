import pickle
from pathlib import Path
from datetime import datetime
from typing import Optional, Any

class CacheManager:
    """Manages caching operations for the application"""
    
    def __init__(self, base_dir: str = 'data_cache'):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        
    def get_cache_dir(self, *subdirs: str) -> Path:
        """Get cache directory path for given subdirectories"""
        cache_dir = self.base_dir.joinpath(*subdirs)
        cache_dir.mkdir(parents=True, exist_ok=True)
        return cache_dir
        
    def get_cache_path(self, filename: str, *subdirs: str) -> Path:
        """Get cache file path for given filename and subdirectories"""
        return self.get_cache_dir(*subdirs) / filename
        
    def save_to_cache(self, data: Any, filename: str, *subdirs: str) -> None:
        """Save data to cache file"""
        cache_path = self.get_cache_path(filename, *subdirs)
        with open(cache_path, 'wb') as f:
            pickle.dump(data, f)
            
    def load_from_cache(self, filename: str, *subdirs: str) -> Optional[Any]:
        """Load data from cache file if it exists"""
        cache_path = self.get_cache_path(filename, *subdirs)
        if cache_path.exists():
            with open(cache_path, 'rb') as f:
                return pickle.load(f)
        return None
        
    def exists_in_cache(self, filename: str, *subdirs: str) -> bool:
        """Check if file exists in cache"""
        cache_path = self.get_cache_path(filename, *subdirs)
        return cache_path.exists()
        
    def get_date_cache_path(self, date: datetime, suffix: str = '', *subdirs: str) -> Path:
        """Get cache path for a specific date"""
        date_str = date.strftime('%Y-%m-%d')
        filename = f"{date_str}{suffix}.pkl"
        return self.get_cache_path(filename, *subdirs)
        
    def save_date_to_cache(self, date: datetime, data: Any, suffix: str = '', *subdirs: str) -> None:
        """Save data to cache with date-based filename"""
        cache_path = self.get_date_cache_path(date, suffix, *subdirs)
        with open(cache_path, 'wb') as f:
            pickle.dump(data, f)
            
    def load_date_from_cache(self, date: datetime, suffix: str = '', *subdirs: str) -> Optional[Any]:
        """Load data from cache with date-based filename"""
        cache_path = self.get_date_cache_path(date, suffix, *subdirs)
        if cache_path.exists():
            with open(cache_path, 'rb') as f:
                return pickle.load(f)
        return None 