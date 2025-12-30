import pandas as pd
import numpy as np
from pathlib import Path


class CalendarFeatureProcessor:
    """Process economic calendar data to create features for LSTM model"""
    
    def __init__(self, calendar_data_path=None, event_types=None):
        """Initialize the calendar feature processor
        
        Args:
            calendar_data_path: Path to the calendar CSV file (if None, will try to find files automatically)
            event_types: List of economic event types to process (e.g., ["Core CPI", "CB Consumer Confidence", "Fed Funds Rate"])
        """
        if event_types is None:
            event_types = ["Core CPI", "CB Consumer Confidence", "Fed Funds Rate"]
        
        self.event_types = event_types
        self.events_data = {}  # Dictionary to store events for each type
        
        # Load events for each event type
        for event_type in event_types:
            self._load_events_for_type(event_type, calendar_data_path)
    
    def _load_events_for_type(self, event_type, calendar_data_path=None):
        """Load events for a specific event type"""
        if calendar_data_path is None:
            # Try multiple possible paths for the calendar data
            possible_paths = []
            
            # Path 1: Relative to current file (when run as module)
            current_dir = Path(__file__).parent.parent.parent
            possible_paths.append(current_dir / "data_cache" / "calendar" / f"Historic Economic Event Calendar Data - {event_type}.csv")
            
            # Path 2: Relative to working directory (when run from project root)
            possible_paths.append(Path("data_cache") / "calendar" / f"Historic Economic Event Calendar Data - {event_type}.csv")
            
            # Path 3: Absolute path from project root
            project_root = Path.cwd()
            possible_paths.append(project_root / "data_cache" / "calendar" / f"Historic Economic Event Calendar Data - {event_type}.csv")
            
            # Find the first path that exists
            event_calendar_data_path = None
            for path in possible_paths:
                if path.exists():
                    event_calendar_data_path = path
                    break
            
            if event_calendar_data_path is None:
                # If none found, use the first path and let the error be raised with a helpful message
                event_calendar_data_path = possible_paths[0]
                print(f"⚠️  Warning: {event_type} calendar data not found. Tried these paths:")
                for i, path in enumerate(possible_paths, 1):
                    print(f"   {i}. {path}")
                print(f"   Please ensure the file exists at one of these locations.")
        else:
            event_calendar_data_path = Path(calendar_data_path)
        
        self._load_events(event_type, event_calendar_data_path)
    
    def _load_events(self, event_type, calendar_data_path):
        """Load and parse calendar events from CSV for a specific event type"""
        if not calendar_data_path.exists():
            raise FileNotFoundError(f"{event_type} calendar data not found at {calendar_data_path}")
        
        print(f"📅 Loading {event_type} calendar data from {calendar_data_path}")
        
        # Read CSV with no header (data starts from first row)
        df = pd.read_csv(calendar_data_path, header=None, names=[
            'Date', 'Time', 'Currency', 'Impact', 'Event', 'Actual', 'Forecast', 'Previous'
        ])
        
        # Filter for the specific event type
        if event_type == "Core CPI":
            event_filter = 'Core CPI m/m'
        elif event_type == "CB Consumer Confidence":
            event_filter = 'CB Consumer Confidence'
        elif event_type == "Fed Funds Rate":
            event_filter = 'Federal Funds Rate'
        else:
            event_filter = event_type
        
        event_mask = df['Event'].str.contains(event_filter, na=False)
        events = df[event_mask].copy()
        
        # Convert date column to datetime
        events['Date'] = pd.to_datetime(events['Date'], format='%Y/%m/%d')
        
        # Sort by date
        events = events.sort_values('Date').reset_index(drop=True)
        
        # Store events for this type
        self.events_data[event_type] = events
        
        print(f"✅ Loaded {len(events)} {event_type} events from {events['Date'].min().strftime('%Y-%m-%d')} to {events['Date'].max().strftime('%Y-%m-%d')}")
    
    def calculate_all_features(self, data):
        """Calculate calendar features for all event types at once
        
        Args:
            data: DataFrame with 'Date' column or datetime index
            
        Returns:
            DataFrame with calendar features for all event types
        """
        if not self.events_data:
            raise ValueError("No events loaded. Call _load_events_for_type() first.")
        
        print(f"🔮 Calculating calendar features for {len(data)} data points...")
        
        # Handle both cases: Date column or datetime index
        if 'Date' in data.columns:
            # Data has a Date column
            date_series = data['Date']
        elif isinstance(data.index, pd.DatetimeIndex):
            # Data uses datetime index
            date_series = data.index
            # Create a copy to avoid modifying the original
            data = data.copy()
        else:
            raise ValueError("Data must contain a 'Date' column or have a datetime index")
        
        # Convert to datetime if it's not already
        if not pd.api.types.is_datetime64_any_dtype(date_series):
            date_series = pd.to_datetime(date_series)
        
        # Calculate features for each event type
        for event_type in self.event_types:
            if event_type not in self.events_data:
                print(f"⚠️  Warning: No events found for {event_type}")
                continue
                
            # Generate feature prefix
            if event_type == "Core CPI":
                feature_prefix = "CPI"
            elif event_type == "CB Consumer Confidence":
                feature_prefix = "CC"
            elif event_type == "Fed Funds Rate":
                feature_prefix = "FFR"
            else:
                feature_prefix = event_type.replace(" ", "_")
            
            # Initialize new columns
            days_since_col = f'Days_Since_Last_{feature_prefix}'
            days_until_col = f'Days_Until_Next_{feature_prefix}'
            
            data[days_since_col] = np.nan
            data[days_until_col] = np.nan
            
            # Get unique dates from events for faster lookup
            events = self.events_data[event_type]
            event_dates = events['Date'].dt.date.unique()
            event_dates = sorted(event_dates)
            
            # Calculate features for each row in the data
            for idx, date in enumerate(date_series):
                current_date = date.date()
                
                # Find days since last event
                days_since_last = self._calculate_days_since_last(current_date, event_dates)
                data.iloc[idx, data.columns.get_loc(days_since_col)] = days_since_last
                
                # Find days until next event
                days_until_next = self._calculate_days_until_next(current_date, event_dates)
                data.iloc[idx, data.columns.get_loc(days_until_col)] = days_until_next
            
            # Fill any remaining NaN values with reasonable defaults
            data = data.fillna({
                days_since_col: 365,
                days_until_col: 365
            })
            
            print(f"✅ {event_type} features calculated:")
            print(f"   📊 {days_since_col} - Mean: {data[days_since_col].mean():.1f}, Std: {data[days_since_col].std():.1f}")
            print(f"   📊 {days_until_col} - Mean: {data[days_until_col].mean():.1f}, Std: {data[days_until_col].std():.1f}")
        
        return data
    
    def calculate_features(self, data, feature_prefix=None, event_type=None):
        """Calculate days until next and days since last events for a specific event type
        
        Args:
            data: DataFrame with 'Date' column or datetime index
            feature_prefix: Prefix for feature column names (e.g., 'CPI', 'CC')
            event_type: Specific event type to calculate features for
            
        Returns:
            DataFrame with two new columns: 'Days_Until_Next_{prefix}' and 'Days_Since_Last_{prefix}'
        """
        if event_type is None:
            # Use the first event type if none specified
            event_type = self.event_types[0]
        
        if event_type not in self.events_data:
            raise ValueError(f"Events for {event_type} not loaded.")
        
        if feature_prefix is None:
            # Generate prefix from event type
            if event_type == "Core CPI":
                feature_prefix = "CPI"
            elif event_type == "CB Consumer Confidence":
                feature_prefix = "CC"
            else:
                feature_prefix = event_type.replace(" ", "_")
        
        print(f"🔮 Calculating {event_type} calendar features for {len(data)} data points...")
        
        # Handle both cases: Date column or datetime index
        if 'Date' in data.columns:
            # Data has a Date column
            date_series = data['Date']
        elif isinstance(data.index, pd.DatetimeIndex):
            # Data uses datetime index
            date_series = data.index
            # Create a copy to avoid modifying the original
            data = data.copy()
        else:
            raise ValueError("Data must contain a 'Date' column or have a datetime index")
        
        # Convert to datetime if it's not already
        if not pd.api.types.is_datetime64_any_dtype(date_series):
            date_series = pd.to_datetime(date_series)
        
        # Initialize new columns
        days_since_col = f'Days_Since_Last_{feature_prefix}'
        days_until_col = f'Days_Until_Next_{feature_prefix}'
        
        data[days_since_col] = np.nan
        data[days_until_col] = np.nan
        
        # Get unique dates from events for faster lookup
        events = self.events_data[event_type]
        event_dates = events['Date'].dt.date.unique()
        event_dates = sorted(event_dates)
        
        # Calculate features for each row in the data
        for idx, date in enumerate(date_series):
            current_date = date.date()
            
            # Find days since last event
            days_since_last = self._calculate_days_since_last(current_date, event_dates)
            data.iloc[idx, data.columns.get_loc(days_since_col)] = days_since_last
            
            # Find days until next event
            days_until_next = self._calculate_days_until_next(current_date, event_dates)
            data.iloc[idx, data.columns.get_loc(days_until_col)] = days_until_next
        
        # Fill any remaining NaN values with reasonable defaults
        data = data.fillna({
            days_since_col: 365,
            days_until_col: 365
        })
        
        print(f"✅ {event_type} features calculated:")
        print(f"   📊 {days_since_col} - Mean: {data[days_since_col].mean():.1f}, Std: {data[days_since_col].std():.1f}")
        print(f"   📊 {days_until_col} - Mean: {data[days_until_col].mean():.1f}, Std: {data[days_until_col].std():.1f}")
        
        return data
    
    def calculate_cpi_features(self, data):
        """Calculate days until next and days since last Core CPI events for each date in the dataset
        
        Args:
            data: DataFrame with 'Date' column or datetime index
            
        Returns:
            DataFrame with two new columns: 'Days_Until_Next_CPI' and 'Days_Since_Last_CPI'
        """
        return self.calculate_features(data, feature_prefix="CPI", event_type="Core CPI")
    
    def _calculate_days_since_last(self, current_date, event_dates):
        """Calculate days since the last event
        
        Args:
            current_date: Current date to check from
            event_dates: List of event dates
            
        Returns:
            Number of days since last event, or NaN if no previous event
        """
        # Find the most recent event before or on the current date
        previous_events = [d for d in event_dates if d <= current_date]
        
        if not previous_events:
            return np.nan
        
        last_event_date = max(previous_events)
        days_since = (current_date - last_event_date).days
        
        return days_since
    
    def _calculate_days_until_next(self, current_date, event_dates):
        """Calculate days until the next event
        
        Args:
            current_date: Current date to check from
            event_dates: List of event dates
            
        Returns:
            Number of days until next event, or NaN if no next event
        """
        # Find the next event after the current date
        future_events = [d for d in event_dates if d > current_date]
        
        if not future_events:
            return np.nan
        
        next_event_date = min(future_events)
        days_until = (next_event_date - current_date).days
        
        return days_until
    
    def get_event_summary(self, event_type=None):
        """Get summary statistics of events
        
        Args:
            event_type: Specific event type to get summary for (if None, returns summary for first event type)
            
        Returns:
            Dictionary with event statistics
        """
        if event_type is None:
            event_type = self.event_types[0]
        
        if event_type not in self.events_data:
            return None
        
        events = self.events_data[event_type]
        
        # Calculate typical intervals between events
        intervals = []
        for i in range(1, len(events)):
            interval = (events.iloc[i]['Date'] - events.iloc[i-1]['Date']).days
            intervals.append(interval)
        
        summary = {
            'event_type': event_type,
            'total_events': len(events),
            'date_range': {
                'start': events['Date'].min().strftime('%Y-%m-%d'),
                'end': events['Date'].max().strftime('%Y-%m-%d')
            },
            'intervals': {
                'mean_days': np.mean(intervals) if intervals else 0,
                'median_days': np.median(intervals) if intervals else 0,
                'std_days': np.std(intervals) if intervals else 0,
                'min_days': np.min(intervals) if intervals else 0,
                'max_days': np.max(intervals) if intervals else 0
            },
            'impact_distribution': events['Impact'].value_counts().to_dict()
        }
        
        return summary
    
    def get_all_event_summaries(self):
        """Get summary statistics for all event types
        
        Returns:
            Dictionary with summaries for all event types
        """
        summaries = {}
        for event_type in self.event_types:
            summaries[event_type] = self.get_event_summary(event_type)
        return summaries
    
    
    def get_ffr_event_summary(self):
        """Get summary statistics of Fed Funds Rate events
        
        Returns:
            Dictionary with Fed Funds Rate event statistics
        """
        return self.get_event_summary("Fed Funds Rate")
    
    def plot_features(self, data, feature_prefix=None, event_type=None, save_path=None):
        """Create a plot showing the calendar features over time
        
        Args:
            data: DataFrame with features
            feature_prefix: Prefix for feature column names (e.g., 'CPI', 'CC')
            event_type: Specific event type to plot
            save_path: Optional path to save the plot
        """
        if event_type is None:
            event_type = self.event_types[0]
        
        if feature_prefix is None:
            # Generate prefix from event type
            if event_type == "Core CPI":
                feature_prefix = "CPI"
            elif event_type == "CB Consumer Confidence":
                feature_prefix = "CC"
            elif event_type == "Fed Funds Rate":
                feature_prefix = "FFR"
            else:
                feature_prefix = event_type.replace(" ", "_")
        
        days_since_col = f'Days_Since_Last_{feature_prefix}'
        days_until_col = f'Days_Until_Next_{feature_prefix}'
        
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            
            # Set up the plot
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))
            
            # Plot days since last event
            ax1.plot(data['Date'], data[days_since_col], 
                    color='blue', alpha=0.7, linewidth=1)
            ax1.set_title(f'Days Since Last {event_type} Event', fontsize=14, fontweight='bold')
            ax1.set_ylabel('Days', fontsize=12)
            ax1.grid(True, alpha=0.3)
            
            # Add event markers
            events = self.events_data[event_type]
            event_dates = events['Date']
            ax1.scatter(event_dates, [0] * len(event_dates), 
                       color='red', s=50, alpha=0.7, marker='|', label=f'{event_type} Events')
            ax1.legend()
            
            # Plot days until next event
            ax2.plot(data['Date'], data[days_until_col], 
                    color='green', alpha=0.7, linewidth=1)
            ax2.set_title(f'Days Until Next {event_type} Event', fontsize=14, fontweight='bold')
            ax2.set_ylabel('Days', fontsize=12)
            ax2.set_xlabel('Date', fontsize=12)
            ax2.grid(True, alpha=0.3)
            
            # Add event markers
            ax2.scatter(event_dates, [0] * len(event_dates), 
                       color='red', s=50, alpha=0.7, marker='|', label=f'{event_type} Events')
            ax2.legend()
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                print(f"📊 {event_type} features plot saved to {save_path}")
            
            plt.show()
            
        except ImportError:
            print("⚠️  matplotlib not available. Skipping plot generation.")
        except Exception as e:
            print(f"⚠️  Error creating plot: {e}")
    


def add_cpi_features_to_data(data, calendar_data_path=None):
    """Convenience function to add CPI features to a dataset
    
    Args:
        data: DataFrame with 'Date' column
        calendar_data_path: Optional path to calendar data file
        
    Returns:
        DataFrame with CPI features added
    """
    processor = CalendarFeatureProcessor(calendar_data_path, event_types=["Core CPI"])
    return processor.calculate_cpi_features(data)

def add_cc_features_to_data(data, calendar_data_path=None):
    """Convenience function to add CB Consumer Confidence features to a dataset
    
    Args:
        data: DataFrame with 'Date' column
        calendar_data_path: Optional path to calendar data file
        
    Returns:
        DataFrame with CC features added
    """
    processor = CalendarFeatureProcessor(calendar_data_path, event_types=["CB Consumer Confidence"])
    return processor.calculate_features(data, feature_prefix="CC", event_type="CB Consumer Confidence")

def add_ffr_features_to_data(data, calendar_data_path=None):
    """Convenience function to add Fed Funds Rate features to a dataset
    
    Args:
        data: DataFrame with 'Date' column
        calendar_data_path: Optional path to calendar data file
        
    Returns:
        DataFrame with Fed Funds Rate features added
    """
    processor = CalendarFeatureProcessor(calendar_data_path, event_types=["Fed Funds Rate"])
    return processor.calculate_features(data, feature_prefix="FFR", event_type="Fed Funds Rate")

def add_all_calendar_features_to_data(data, calendar_data_path=None):
    """Convenience function to add all calendar features to a dataset
    
    Args:
        data: DataFrame with 'Date' column
        calendar_data_path: Optional path to calendar data file
        
    Returns:
        DataFrame with all calendar features added
    """
    processor = CalendarFeatureProcessor(calendar_data_path)
    return processor.calculate_all_features(data) 