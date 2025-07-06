#!/usr/bin/env python3
"""
Calendar Configuration Manager for prediction module
Handles loading and validating calendar event dates for CPI and CC features
"""

import json
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Optional, Tuple


class CalendarConfigManager:
    """Manages calendar event configuration for prediction module"""
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize the calendar config manager
        
        Args:
            config_path: Path to the calendar config JSON file
        """
        if config_path is None:
            # Default to the prediction folder
            config_path = Path(__file__).parent / "calendar_config.json"
        
        self.config_path = Path(config_path)
        self.config = None
        self._load_config()
    
    def _load_config(self):
        """Load the calendar configuration from JSON file"""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Calendar config file not found: {self.config_path}")
        
        try:
            with open(self.config_path, 'r') as f:
                self.config = json.load(f)
            print(f"📅 Loaded calendar config from {self.config_path}")
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in calendar config file: {e}")
        except Exception as e:
            raise RuntimeError(f"Error loading calendar config: {e}")
    
    def validate_dates(self) -> Dict[str, Dict[str, str]]:
        """Validate calendar event dates and return warnings
        
        Returns:
            Dictionary with validation results and warnings
        """
        if not self.config:
            return {"error": "No configuration loaded"}
        
        today = datetime.now().date()
        current_month = today.month
        current_year = today.year
        
        # Calculate date ranges for validation
        last_month_start = (today.replace(day=1) - timedelta(days=1)).replace(day=1)
        last_month_end = today.replace(day=1) - timedelta(days=1)
        this_month_start = today.replace(day=1)
        
        warnings = {}
        
        for event_type, event_data in self.config.items():
            if event_type in ["last_updated", "update_frequency", "notes"]:
                continue
                
            warnings[event_type] = {
                "last_event_warning": None,
                "next_event_warning": None,
                "last_event_date": event_data.get("last_event"),
                "next_event_date": event_data.get("next_event")
            }
            
            # Validate last event date
            try:
                last_event_date = datetime.strptime(event_data["last_event"], "%Y-%m-%d").date()
                
                # Check if last event is not in this month or last month
                if not (last_month_start <= last_event_date <= today):
                    warnings[event_type]["last_event_warning"] = (
                        f"Last {event_type} event ({last_event_date}) is not in this month or last month. "
                        f"Consider updating the configuration."
                    )
            except (KeyError, ValueError) as e:
                warnings[event_type]["last_event_warning"] = f"Invalid last event date: {e}"
            
            # Validate next event date
            try:
                next_event_date = datetime.strptime(event_data["next_event"], "%Y-%m-%d").date()
                
                # Check if next event is in the past
                if next_event_date < today:
                    warnings[event_type]["next_event_warning"] = (
                        f"Next {event_type} event ({next_event_date}) is in the past. "
                        f"Configuration needs immediate update."
                    )
            except (KeyError, ValueError) as e:
                warnings[event_type]["next_event_warning"] = f"Invalid next event date: {e}"
        
        return warnings
    
    def get_calendar_features_for_date(self, target_date: datetime) -> Dict[str, int]:
        """Calculate calendar features for a specific date
        
        Args:
            target_date: Date to calculate features for
            
        Returns:
            Dictionary with calendar feature values
        """
        if not self.config:
            raise ValueError("No configuration loaded")
        
        target_date = target_date.date()
        features = {}
        
        for event_type, event_data in self.config.items():
            if event_type in ["last_updated", "update_frequency", "notes"]:
                continue
            
            try:
                last_event_date = datetime.strptime(event_data["last_event"], "%Y-%m-%d").date()
                next_event_date = datetime.strptime(event_data["next_event"], "%Y-%m-%d").date()
                
                # Calculate days since last event
                days_since_last = (target_date - last_event_date).days
                if days_since_last < 0:
                    days_since_last = 365  # Default if date is before last event
                
                # Calculate days until next event
                days_until_next = (next_event_date - target_date).days
                if days_until_next < 0:
                    days_until_next = 365  # Default if next event is in the past
                
                # Generate feature names based on event type
                if event_type == "Core CPI":
                    prefix = "CPI"
                elif event_type == "CB Consumer Confidence":
                    prefix = "CC"
                else:
                    prefix = event_type.replace(" ", "_")
                
                features[f"Days_Since_Last_{prefix}"] = days_since_last
                features[f"Days_Until_Next_{prefix}"] = days_until_next
                
            except (KeyError, ValueError) as e:
                print(f"⚠️  Warning: Could not calculate features for {event_type}: {e}")
                # Use default values
                if event_type == "Core CPI":
                    features["Days_Since_Last_CPI"] = 365
                    features["Days_Until_Next_CPI"] = 365
                elif event_type == "CB Consumer Confidence":
                    features["Days_Since_Last_CC"] = 365
                    features["Days_Until_Next_CC"] = 365
        
        return features
    
    def print_validation_report(self):
        """Print a validation report for the calendar configuration"""
        warnings = self.validate_dates()
        
        print("\n📅 Calendar Configuration Validation Report")
        print("=" * 60)
        
        for event_type, validation in warnings.items():
            if event_type in ["last_updated", "update_frequency", "notes"]:
                continue
                
            print(f"\n🔍 {event_type}:")
            print(f"   Last Event: {validation['last_event_date']}")
            print(f"   Next Event: {validation['next_event_date']}")
            
            if validation["last_event_warning"]:
                print(f"   ⚠️  {validation['last_event_warning']}")
            
            if validation["next_event_warning"]:
                print(f"   ❌ {validation['next_event_warning']}")
            
            if not validation["last_event_warning"] and not validation["next_event_warning"]:
                print("   ✅ Dates appear to be current")
        
        # Show last updated info
        if "last_updated" in self.config:
            print(f"\n📝 Last Updated: {self.config['last_updated']}")
        
        print("=" * 60)
    
    def update_config(self, event_type: str, last_event: str, next_event: str):
        """Update the configuration for a specific event type
        
        Args:
            event_type: Type of event to update
            last_event: Last event date (YYYY-MM-DD)
            next_event: Next event date (YYYY-MM-DD)
        """
        if not self.config:
            raise ValueError("No configuration loaded")
        
        if event_type not in self.config:
            raise ValueError(f"Unknown event type: {event_type}")
        
        # Validate date format
        try:
            datetime.strptime(last_event, "%Y-%m-%d")
            datetime.strptime(next_event, "%Y-%m-%d")
        except ValueError as e:
            raise ValueError(f"Invalid date format. Use YYYY-MM-DD: {e}")
        
        # Update the configuration
        self.config[event_type]["last_event"] = last_event
        self.config[event_type]["next_event"] = next_event
        self.config["last_updated"] = datetime.now().strftime("%Y-%m-%d")
        
        # Save the updated configuration
        try:
            with open(self.config_path, 'w') as f:
                json.dump(self.config, f, indent=2)
            print(f"✅ Updated calendar config for {event_type}")
        except Exception as e:
            raise RuntimeError(f"Error saving updated config: {e}")


def validate_calendar_config():
    """Convenience function to validate calendar configuration"""
    try:
        manager = CalendarConfigManager()
        manager.print_validation_report()
        return True
    except Exception as e:
        print(f"❌ Calendar config validation failed: {e}")
        return False


if __name__ == "__main__":
    # Test the calendar config manager
    validate_calendar_config() 