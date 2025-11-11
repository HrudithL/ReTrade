# Opening Range Breakout Strategy Backtester

A clean, testable Python implementation for backtesting an Opening Range Breakout (ORB) strategy using OANDA v20 REST API data.

## Strategy Overview

The strategy trades the first 5-minute candle (9:30-9:35 AM New York time) each trading day:

- **Opening Range**: High, Low, and Mid of the 9:30-9:35 candle
- **Bias**: Determined by whether the close is above (long) or below (short) the OR mid
- **Entry**: Breakout above OR high (long) or below OR low (short) after 9:35 AM
- **Risk Management**: Stop at opposite OR boundary (or OR mid if configured), take profit at 2R by default
- **Filters**: ATR-based session filter to avoid low volatility days

## Setup

### Prerequisites

- Python 3.11+
- OANDA API credentials (practice or live account)

### Installation

Using `uv` (recommended):
```bash
uv pip install -e .
```

Or using `pip`:
```bash
pip install -e .
```

For development dependencies:
```bash
pip install -e ".[dev]"
```

### Configuration

1. Copy `.env.example` to `.env`:
```bash
cp .env.example .env
```

2. Edit `.env` with your OANDA credentials:
```
OANDA_API_KEY=your_api_key_here
OANDA_ACCOUNT_ID=your_account_id_here
OANDA_ENV=practice  # or 'live'
```

## Usage

### Basic Example

```bash
python scripts/backtest.py \
  --instrument US500_USD \
  --granularity M5 \
  --start 2023-01-01 \
  --end 2025-11-01 \
  --rr 2.0 \
  --atr_tf M15 \
  --atr_mult 1.2
```

### Parameters

- `--instrument`: OANDA instrument (e.g., US500_USD, NAS100_USD, SPX500_USD, EUR_USD)
- `--granularity`: Candle granularity for OR (M1 or M5 recommended)
- `--start`: Start date (YYYY-MM-DD)
- `--end`: End date (YYYY-MM-DD)
- `--use_mid_stop`: Use OR mid as stop instead of opposite boundary (flag, default: False)
- `--rr`: Risk-reward ratio (default: 2.0)
- `--atr_tf`: Timeframe for ATR calculation (default: M15, but uses daily candles)
- `--atr_mult`: ATR multiplier threshold (default: 1.2)
- `--tz`: Timezone (default: America/New_York)
- `--no_cache`: Disable cache

### Output

- Metrics printed to console
- Trades saved to `./runs/{instrument}_{start}_{end}_{granularity}.csv`

## OANDA Instruments and Granularity

### Supported Instruments
- Index CFDs: `US500_USD`, `NAS100_USD`, `SPX500_USD`
- Forex: `EUR_USD`, `GBP_USD`, etc.

### Granularity Mapping
- `M1`: 1-minute candles
- `M5`: 5-minute candles (recommended for OR)
- `M15`: 15-minute candles
- `D`: Daily candles (used for ATR calculation)
- `H1`, `H4`: Hourly, 4-hour

Note: OANDA uses specific granularity codes. See [OANDA API documentation](https://developer.oanda.com/rest-live-v20/instrument-df/) for full list.

## Common Pitfalls

### DST Handling
The code uses `pytz` to handle Daylight Saving Time transitions correctly. The 9:30-9:35 window is always in America/New_York time, regardless of DST.

### Index CFDs vs ETFs
This strategy is designed for OANDA index CFDs which trade 24/5. The session logic enforces a 9:30 AM NY open, but data is available throughout the week.

### OANDA Weekend Gaps
OANDA markets close Friday evening and reopen Sunday evening. The backtester handles gaps by only processing trading days with valid 9:30-9:35 candles.

### Rate Limits
OANDA has rate limits. The client includes retry logic with exponential backoff. For large date ranges, consider using the cache feature.

### ATR Filter
The ATR filter uses daily candles. If daily candles are unavailable for an instrument, the system will resample intraday data to daily bars. Days with insufficient ATR data are skipped entirely.

## Testing

Run all tests:
```bash
pytest
```

Run with coverage:
```bash
pytest --cov=src --cov-report=html
```

## Code Quality

Format code:
```bash
black src tests scripts
```

Lint:
```bash
ruff check src tests scripts
```

Type check:
```bash
mypy src
```

## Extending Filters

To add custom filters, edit `src/strategy/filters.py`:

```python
def custom_filter(df_day, params):
    # Your filter logic
    return True  # or False
```

Then integrate in `src/backtest/engine.py` in the `run()` method.

## Project Structure

```
.
├── README.md
├── .env.example
├── pyproject.toml
├── scripts/
│   └── backtest.py
├── src/
│   ├── config/
│   │   └── settings.py
│   ├── data/
│   │   ├── oanda_client.py
│   │   └── fetch.py
│   ├── utils/
│   │   ├── timezones.py
│   │   ├── calendar.py
│   │   └── math.py
│   ├── indicators/
│   │   └── atr.py
│   ├── strategy/
│   │   ├── opening_range.py
│   │   └── filters.py
│   ├── backtest/
│   │   ├── engine.py
│   │   └── metrics.py
│   └── io/
│       └── cache.py
└── tests/
    ├── test_timezones.py
    ├── test_opening_range.py
    ├── test_engine.py
    └── test_atr.py
```

## License

MIT

