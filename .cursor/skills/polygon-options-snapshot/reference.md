> For the full documentation index, see: https://massive.com/docs/llms.txt

# REST
## Options

### Option Contract Snapshot

**Endpoint:** `GET /v3/snapshot/options/{underlyingAsset}/{optionContract}`

**Description:**

Retrieve a comprehensive snapshot of a specified options contract, consolidating vital metrics and market data into a single response. This endpoint provides details such as break-even price, day-over-day changes, implied volatility, open interest, greeks (delta, gamma, theta, vega), and the latest quote and trade information. Users also gain insights into the underlying asset’s current price, enabling a full evaluation of the contract’s value and potential.

Use Cases: Trade evaluation, market analysis, risk assessment, and strategy refinement.

## Query Parameters

| Parameter | Type | Required | Description |
| --- | --- | --- | --- |
| `underlyingAsset` | string | Yes | The underlying ticker symbol of the option contract. |
| `optionContract` | string | Yes | The option contract identifier. |

## Response Attributes

| Field | Type | Description |
| --- | --- | --- |
| `next_url` | string | If present, this value can be used to fetch the next page of data. |
| `request_id` | string | A request id assigned by the server. |
| `results` | object | Contains the requested snapshot data for the specified contract. |
| `results.break_even_price` | number | The price of the underlying asset for the contract to break even. For a call, this value is (strike price + premium paid). For a put, this value is (strike price - premium paid). |
| `results.day` | object | The most recent daily bar for this contract. |
| `results.details` | object | The details for this contract. |
| `results.fmv` | number | Fair Market Value is only available on Business plans. It is our proprietary algorithm to generate a real-time, accurate, fair market value of a tradable security. For more information, <a rel="nofollow" target="_blank" href="https://massive.com/contact">contact us</a>. |
| `results.fmv_last_updated` | integer | If Fair Market Value (FMV) is available, this field is the nanosecond timestamp of the last FMV calculation. |
| `results.greeks` | object | The greeks for this contract. There are certain circumstances where greeks will not be returned, such as options contracts that are deep in the money. See this <a href="https://massive.com/blog/greeks-and-implied-volatility/#testing" alt="link">article</a> for more information. |
| `results.implied_volatility` | number | The market's forecast for the volatility of the underlying asset, based on this option's current price. |
| `results.last_quote` | object | The most recent quote for this contract. This is only returned if your current plan includes quotes. |
| `results.last_trade` | object | The most recent trade for this contract. This is only returned if your current plan includes trades. |
| `results.open_interest` | number | The quantity of this contract held at the end of the last trading day. |
| `results.underlying_asset` | object | Information on the underlying stock for this options contract.  The market data returned depends on your current stocks plan. |
| `status` | string | The status of this request's response. |

## Sample Response

```json
{
  "request_id": "d9ff18dac69f55c218f69e4753706acd",
  "results": {
    "break_even_price": 171.075,
    "day": {
      "change": -1.05,
      "change_percent": -4.67,
      "close": 21.4,
      "high": 22.49,
      "last_updated": 1636520400000000000,
      "low": 21.35,
      "open": 22.49,
      "previous_close": 22.45,
      "volume": 37,
      "vwap": 21.6741
    },
    "details": {
      "contract_type": "call",
      "exercise_style": "american",
      "expiration_date": "2023-06-16",
      "shares_per_contract": 100,
      "strike_price": 150,
      "ticker": "O:AAPL230616C00150000"
    },
    "fmv": 0.05,
    "fmv_last_updated": 1636573458757383400,
    "greeks": {
      "delta": 0.5520187372272933,
      "gamma": 0.00706756515659829,
      "theta": -0.018532772783847958,
      "vega": 0.7274811132998142
    },
    "implied_volatility": 0.3048997097864957,
    "last_quote": {
      "ask": 21.25,
      "ask_exchange": 301,
      "ask_size": 110,
      "bid": 20.9,
      "bid_exchange": 301,
      "bid_size": 172,
      "last_updated": 1636573458756383500,
      "midpoint": 21.075,
      "timeframe": "REAL-TIME"
    },
    "last_trade": {
      "conditions": [
        209
      ],
      "exchange": 316,
      "price": 0.05,
      "sip_timestamp": 1675280958783136800,
      "size": 2,
      "timeframe": "REAL-TIME"
    },
    "open_interest": 8921,
    "underlying_asset": {
      "change_to_break_even": 23.123999999999995,
      "last_updated": 1636573459862384600,
      "price": 147.951,
      "ticker": "AAPL",
      "timeframe": "REAL-TIME"
    }
  },
  "status": "OK"
}
```


## Plan Access

**Plan Access:** Included in select Options plans

#### Individual Plans

| Plan | Access |
| --- | --- |
| Options Basic | Not included |
| Options Starter | Included |
| Options Developer | Included |
| Options Advanced | Included |

#### Business Plans

| Plan | Access |
| --- | --- |
| Options Business | Included |

## Plan Recency

**Plan Recency:** 15-minute delayed or real-time

#### Individual Plans

| Plan | Recency |
| --- | --- |
| Options Basic | Not included |
| Options Starter | 15-minute delayed |
| Options Developer | 15-minute delayed |
| Options Advanced | Real-time |

#### Business Plans

| Plan | Recency |
| --- | --- |
| Options Business | Real-time |

## Plan History

**Plan History:** Not applicable to this endpoint
