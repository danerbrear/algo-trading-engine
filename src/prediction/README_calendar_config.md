# Calendar Configuration for Predictions

This document explains how to use the calendar configuration system for accurate CPI and CC feature calculations in predictions.

## Overview

The calendar configuration system provides a centralized way to manage economic calendar event dates for Core CPI and CB Consumer Confidence releases. This ensures that the LSTM model has accurate calendar features when making predictions.

## Files

- `calendar_config.json` - Configuration file containing event dates
- `calendar_config_manager.py` - Manager class for loading and validating dates
- `predict_today.py` - Updated prediction script that uses the configuration

## Configuration File Structure

The `calendar_config.json` file contains:

```json
{
  "Core CPI": {
    "last_event": "2025-06-11",
    "next_event": "2025-07-10",
    "description": "Core Consumer Price Index (CPI) monthly release",
    "typical_interval_days": 30,
    "source": "Bureau of Labor Statistics"
  },
  "CB Consumer Confidence": {
    "last_event": "2025-06-24",
    "next_event": "2025-07-29",
    "description": "Conference Board Consumer Confidence Index",
    "typical_interval_days": 30,
    "source": "Conference Board"
  },
  "last_updated": "2025-07-06",
  "update_frequency": "monthly",
  "notes": "Update these dates when new economic calendar data becomes available."
}
```

## Validation Rules

The system validates calendar dates and shows warnings for:

1. **Last Event Date**: Warns if the last event is not in the current month or last month
2. **Next Event Date**: Warns if the next event is in the past

## Usage

### Automatic Validation

When running `predict_today.py`, the system automatically:

1. Loads the calendar configuration
2. Validates the dates and shows warnings if needed
3. Uses the configuration to calculate calendar features
4. Falls back to the calendar processor if configuration fails

### Manual Validation

You can manually validate the configuration:

```python
from src.prediction.calendar_config_manager import CalendarConfigManager

# Load and validate
config_manager = CalendarConfigManager()
config_manager.print_validation_report()

# Get features for a specific date
from datetime import datetime
today = datetime.now()
features = config_manager.get_calendar_features_for_date(today)
print(features)
```

### Updating Configuration

#### Method 1: Edit JSON File Directly

Edit `src/prediction/calendar_config.json` and update the dates:

```json
{
  "Core CPI": {
    "last_event": "2025-07-10",
    "next_event": "2025-08-14"
  }
}
```

#### Method 2: Programmatic Update

```python
from src.prediction.calendar_config_manager import CalendarConfigManager

config_manager = CalendarConfigManager()
config_manager.update_config(
    event_type="Core CPI",
    last_event="2025-07-10",
    next_event="2025-08-14"
)
```

## Calendar Feature Calculation

The system calculates four calendar features:

- `Days_Since_Last_CPI` - Days since the last Core CPI release
- `Days_Until_Next_CPI` - Days until the next Core CPI release
- `Days_Since_Last_CC` - Days since the last CB Consumer Confidence release
- `Days_Until_Next_CC` - Days until the next CB Consumer Confidence release

## Integration with Prediction Script

The `predict_today.py` script now:

1. **Validates Configuration**: Shows warnings for outdated dates
2. **Uses Configuration**: Calculates features using the configuration file
3. **Fallback Support**: Falls back to calendar processor if needed
4. **Real-time Features**: Provides accurate calendar features for predictions

## Maintenance

### Monthly Updates

Economic calendar events typically occur monthly. Update the configuration:

1. After each CPI release (usually around the 10th-15th of each month)
2. After each CB Consumer Confidence release (usually around the 25th-30th of each month)

### Sources for Dates

- **Core CPI**: Bureau of Labor Statistics (BLS) economic calendar
- **CB Consumer Confidence**: Conference Board economic calendar

### Validation Schedule

Run validation before each prediction to ensure accuracy:

```python
from src.prediction.calendar_config_manager import validate_calendar_config
validate_calendar_config()
```

## Troubleshooting

### Common Issues

1. **Outdated Dates**: Update the configuration file with current dates
2. **Missing File**: Ensure `calendar_config.json` exists in the prediction folder
3. **Invalid JSON**: Check JSON syntax in the configuration file
4. **Fallback Mode**: If configuration fails, the system uses the calendar processor

### Error Messages

- `Calendar config file not found`: Check file path and existence
- `Invalid JSON in calendar config file`: Fix JSON syntax
- `Next event is in the past`: Update the next event date
- `Last event is not current`: Update the last event date

## Benefits

1. **Accuracy**: Ensures calendar features are current and accurate
2. **Efficiency**: Faster than loading full calendar data
3. **Maintainability**: Centralized date management
4. **Validation**: Automatic warnings for outdated information
5. **Reliability**: Fallback to full calendar processor if needed 