# Bull Market SPY Mean Reversion v2

# Hypothesis

During a bull market, there are days where the price of the underlying is higher than its historical average and will revert towards it’s historical value. 

# Definitions

**Upward trends** are defined as:

- Increasing width between SMA15 and SMA30 when SMA15 is above SMA30
    - Should compare the width between SMA15 and SMA30 to the previous day to determine if the width is increasing

# Resources

[Trading Volatility with Options](https://www.quantconnect.com/research/17882/trading-volatility-with-options/p1)

# Optimization

Variables we can fine tune by running back tests.

- Entry
    - Z-Score entry value
- Exit
    - Z-Score exit value
    - Z-Score change since open
    - Stop loss
    - Profit take

# Entry Selection Criteria

1. SPY Put Debit Spread criteria
    - Receive signal
        - There should be a current ongoing upward trend (based on our definition of an upward trend)
        - The Z-Score of the underlying price is >1
    - Width ≤ $6
    - 7-10 DTE
2. Hold
    1. There is no signal

# Opening a Position

1. Only 1 open position at a time
2. Determine size based on maximum risk per trade of 8%

# Closing a Position

1. Close if stop loss has been met
2. Close if profit take has been met
3. Close if Z-Score is < 0.5
4. Close if Z-Score decreased by > 0.7
5. Close at 2 DTE

---