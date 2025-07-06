#!/usr/bin/env python3
"""
Simple script to update calendar configuration dates
Usage: python update_calendar_config.py [event_type] [last_event_date] [next_event_date]
"""

import sys
import argparse
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from prediction.calendar_config_manager import CalendarConfigManager

def update_calendar_config(event_type, last_event, next_event):
    """Update calendar configuration for a specific event type"""
    
    try:
        print(f"📅 Updating calendar configuration...")
        print(f"   Event Type: {event_type}")
        print(f"   Last Event: {last_event}")
        print(f"   Next Event: {next_event}")
        
        # Load configuration manager
        config_manager = CalendarConfigManager()
        
        # Update the configuration
        config_manager.update_config(event_type, last_event, next_event)
        
        print(f"✅ Successfully updated {event_type} configuration")
        
        # Show validation report
        print(f"\n📋 Updated Configuration:")
        config_manager.print_validation_report()
        
    except Exception as e:
        print(f"❌ Error updating configuration: {e}")
        return False
    
    return True

def show_current_config():
    """Show current calendar configuration"""
    
    try:
        config_manager = CalendarConfigManager()
        config_manager.print_validation_report()
    except Exception as e:
        print(f"❌ Error loading configuration: {e}")

def main():
    """Main function to handle command line arguments"""
    
    parser = argparse.ArgumentParser(description='Update calendar configuration dates')
    parser.add_argument('--show', action='store_true', help='Show current configuration')
    parser.add_argument('--event-type', choices=['Core CPI', 'CB Consumer Confidence'], 
                       help='Type of event to update')
    parser.add_argument('--last-event', help='Last event date (YYYY-MM-DD)')
    parser.add_argument('--next-event', help='Next event date (YYYY-MM-DD)')
    
    args = parser.parse_args()
    
    # Show current configuration if requested
    if args.show:
        show_current_config()
        return
    
    # Check if all required arguments are provided
    if not all([args.event_type, args.last_event, args.next_event]):
        print("❌ Error: All arguments are required for update")
        print("Usage: python update_calendar_config.py --event-type 'Core CPI' --last-event '2025-06-11' --next-event '2025-07-10'")
        print("Or use --show to view current configuration")
        return
    
    # Update configuration
    success = update_calendar_config(args.event_type, args.last_event, args.next_event)
    
    if success:
        print(f"\n🎉 Calendar configuration updated successfully!")
    else:
        print(f"\n❌ Failed to update calendar configuration")

if __name__ == "__main__":
    main() 