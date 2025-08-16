# Moving Average Velocity/Elasticity Analysis

## Analysis Goals

- What is the elasticity/velocity (short term ma / longer term ma) that best signals a period of upward trend?
    - The best solution should be determined by percentage times a signal produces a up trend of SPY of at least 3 days
- What is the elasticity/velocity (short term ma / longer term ma) that best signals a period of downward trend?
    - The best solution should be determined by percentage times a signal produces a down trend of SPY of at least 3 days

### Notes

- When tracking return for SPY, note that a downward movement can still indicate a larger upward/downward trend

## Constraints

- The only dependency should be on data retrieval classes
- The output should be a short term MA and a longer term MA for upward swings and a short term MA and longer term MA for downward swings
- Don't consider upward/downward trends longer than 60 days
- Only analyze the most recent 6 months of data
- Include new files in a subfolder of /analysis
- **Focus purely on moving average analysis from SPY price changes**
- **No trading strategy, position management, or actual trades**
- **Only use daily close prices for SPY - no intraday or OHLC data needed**

## Output

- The output should be the best two Moving Averages selected for both upward and downward trends
