# Calendar Configuration Usage Guide

This guide shows you how to use the calendar configuration system for managing CPI and CC event dates.

## Quick Commands

### View Current Configuration
```bash
python update_calendar_config.py --show
```

### Update Core CPI Dates
```bash
python update_calendar_config.py --event-type "Core CPI" --last-event "2025-06-11" --next-event "2025-07-10"
```

### Update CB Consumer Confidence Dates
```bash
python update_calendar_config.py --event-type "CB Consumer Confidence" --last-event "2025-06-24" --next-event "2025-07-29"
```

## What the System Does

1. **Validates Dates**: Shows warnings if dates are outdated
2. **Calculates Features**: Provides accurate calendar features for LSTM predictions
3. **Shows Warnings**: Alerts you when dates need updating

## When to Update

### Core CPI (Monthly)
- **Last Event**: After each CPI release (usually 10th-15th of month)
- **Next Event**: When the next release date is announced
- **Source**: Bureau of Labor Statistics (BLS)

### CB Consumer Confidence (Monthly)
- **Last Event**: After each release (usually 25th-30th of month)
- **Next Event**: When the next release date is announced
- **Source**: Conference Board

## Example Workflow

1. **Check current status**:
   ```bash
   python update_calendar_config.py --show
   ```

2. **Update after CPI release** (example):
   ```bash
   python update_calendar_config.py --event-type "Core CPI" --last-event "2025-07-10" --next-event "2025-08-14"
   ```

3. **Verify the update**:
   ```bash
   python update_calendar_config.py --show
   ```

## Understanding Warnings

- **⚠️ Last event not current**: The last event date is too old
- **❌ Next event in past**: The next event date has already passed
- **✅ Dates appear current**: Everything is up to date

## Integration with Predictions

The `predict_today.py` script automatically:
- Loads the calendar configuration
- Shows validation warnings
- Uses the dates to calculate accurate calendar features
- Falls back to full calendar data if needed

## File Locations

- **Configuration**: `src/prediction/calendar_config.json`
- **Manager**: `src/prediction/calendar_config_manager.py`
- **Update Script**: `update_calendar_config.py`

## Troubleshooting

If you get errors:
1. Make sure you're in the project root directory
2. Ensure the virtual environment is activated
3. Check that the date format is YYYY-MM-DD
4. Verify the event type names are exact: "Core CPI" or "CB Consumer Confidence" 