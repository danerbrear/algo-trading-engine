---
name: polygon-options-snapshot
description: Reference for the Polygon (massive.com) Option Contract Snapshot REST endpoint (GET /v3/snapshot/options/{underlyingAsset}/{optionContract}). Use when fetching a single option contract snapshot, live/near-live option quotes (bid/ask/midpoint), greeks, implied volatility, open interest, or break-even price. This is the preferred endpoint for live options quotes.
compatibility: Requires POLYGON_API_KEY; uses polygon RESTClient (see options_handler.py)
paths:
  - "src/algo_trading_engine/common/options_handler.py"
  - "src/algo_trading_engine/**/options*.py"
---

# Polygon Options Snapshot

The Option Contract Snapshot endpoint returns a comprehensive snapshot for a single options contract: latest quote (bid/ask/midpoint), latest trade, greeks, implied volatility, open interest, break-even price, the most recent daily bar, and the underlying asset price.

**Prefer this endpoint for live options quotes** — `last_quote.bid`, `last_quote.ask`, and `last_quote.midpoint` give the freshest pricing available on the plan.

## Endpoint

```
GET /v3/snapshot/options/{underlyingAsset}/{optionContract}
```

- `underlyingAsset` — underlying ticker (e.g. `AAPL`)
- `optionContract` — contract identifier (e.g. `O:AAPL230616C00150000`)

## Usage in this repo

This codebase uses the `polygon` `RESTClient` with `POLYGON_API_KEY` (see `src/algo_trading_engine/common/options_handler.py`). Use the client's snapshot method:

```python
from polygon import RESTClient

client = RESTClient(api_key)
snapshot = client.get_snapshot_option("AAPL", "O:AAPL230616C00150000")
bid = snapshot.last_quote.bid
ask = snapshot.last_quote.ask
midpoint = snapshot.last_quote.midpoint
```

## Key response fields

- `results.last_quote` — bid, ask, bid_size, ask_size, midpoint, timeframe (live quotes)
- `results.last_trade` — price, size, timeframe
- `results.greeks` — delta, gamma, theta, vega (omitted for some deep-ITM contracts)
- `results.implied_volatility`
- `results.open_interest`
- `results.break_even_price`
- `results.day` — most recent daily bar (open/high/low/close/volume/vwap)
- `results.underlying_asset` — underlying price and ticker

## Plan notes

- `last_quote` requires a plan that includes quotes; `last_trade` requires trades.
- `fmv` is Business plans only.
- Recency: Starter/Developer are 15-minute delayed; Advanced/Business are real-time.

## Full documentation

For the complete query parameters, response attribute table, sample response, and plan access matrix, see [reference.md](reference.md).
