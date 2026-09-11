# ReTrade — Opening Range Breakout Backtester

A Python backtesting engine for an **Opening Range Breakout (ORB)** day-trading strategy on OANDA instruments (index CFDs and forex). It fetches historical candles from the OANDA v20 REST API, simulates the strategy bar-by-bar with realistic trading costs and risk-based position sizing, and reports a full set of performance metrics.

## Strategy Overview

Each trading day, the strategy watches the opening range candle (default **9:30–9:35 AM America/New_York**) and trades the breakout that follows:

1. **Opening Range** — high, low, and mid of the OR window, built from one or more candles
2. **Bias** — long if the OR close is above the OR mid, short if below; days where it's exactly equal are skipped
3. **ATR Filter** — the day is only traded if today's daily ATR is at least `atr_mult × ` the trailing average ATR, to avoid low-volatility chop
4. **Entry** — first breakout above OR high (long) or below OR low (short) after the OR window closes
5. **Stop** — the opposite OR boundary by default, or the OR mid if `use_mid_stop` is set
6. **Take Profit** — a configurable risk-reward multiple of the stop distance (default 2R)
7. **Exit** — stop, take profit, or end-of-day if neither is hit; a configurable policy decides which wins if both are touched on the same bar

## What the Engine Simulates

Beyond the entry/exit logic, the backtest engine models the mechanics of actually trading the strategy:

- **Position sizing** — each trade is sized so the stop distance risks a fixed fraction of account equity (`risk_per_trade`), with a per-day cap (`max_daily_risk`) that skips further trades once the day's risk budget is used
- **Trading costs** — spread, slippage, and commission are applied to both entry and exit, and R-multiples are reported both gross (before costs) and net (after costs)
- **Equity curve** — currency and R-multiple equity are tracked trade-by-trade for drawdown and Sharpe analysis
- **Train/test splits** — run parameters on a training window and evaluate out-of-sample on a separate test window
- **Grid search** — sweep a parameter grid (RR, ATR multiplier, stop placement, etc.) over the training period and carry the best combination (by Sharpe or total R) into the test period

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

For development dependencies (pytest, black, ruff, mypy):
```bash
pip install -e ".[dev]"
```

### Configuration

Create a `.env` file in the project root with your OANDA credentials:
```
OANDA_API_KEY=your_api_key_here
OANDA_ACCOUNT_ID=your_account_id_here
OANDA_ENV=practice  # or 'live'
```

## Usage

### Basic backtest

```bash
python scripts/backtest.py \
  --instrument US500_USD \
  --granularity M5 \
  --start 2023-01-01 \
  --end 2025-11-01 \
  --rr 2.0 \
  --atr_mult 1.2
```

### Train/test split with grid search

```bash
python scripts/backtest.py \
  --instrument US500_USD \
  --granularity M5 \
  --train_start 2022-01-01 --train_end 2023-12-31 \
  --test_start 2024-01-01 --test_end 2024-12-31 \
  --grid_search \
  --selection_metric sharpe
```

Best parameters are selected on the training window and then re-run out-of-sample on the test window.

### Key parameters

| Flag | Description | Default |
|---|---|---|
| `--instrument` | OANDA instrument (e.g. `US500_USD`, `NAS100_USD`, `EUR_USD`) | required |
| `--granularity` | Candle granularity for the OR window | `M5` |
| `--rr` | Risk-reward ratio for take profit | `2.0` |
| `--atr_mult` | ATR filter multiplier | `1.2` |
| `--atr_period` | ATR lookback period (daily bars) | `20` |
| `--use_mid_stop` | Use OR mid as stop instead of the opposite boundary | off |
| `--or_start_time` / `--or_end_time` | Opening range window (`HH:MM`) | `09:30` / `09:35` |
| `--tp_sl_conflict_policy` | `sl_first` or `tp_first` when both trigger on the same bar | `sl_first` |
| `--risk_per_trade` | Fraction of equity risked per trade | `0.0025` (0.25%) |
| `--max_daily_risk` | Max fraction of equity risked per day | `0.01` (1%) |
| `--starting_equity` | Starting account equity for sizing/curve | `100000.0` |
| `--train_start/end`, `--test_start/end` | Train/test date ranges (`YYYY-MM-DD`) | — |
| `--grid_search` | Sweep a parameter grid on the training period | off |
| `--selection_metric` | `sharpe` or `total_R` for picking the best grid result | `sharpe` |
| `--tz` | Timezone for OR window and session logic (IANA name) | `America/New_York` |
| `--no_cache` | Disable the local data cache | off |

Run `python scripts/backtest.py --help` for the full list.

### Output

Console output includes a metrics table (win rate, expectancy, profit factor, Sharpe, max drawdown, etc.), equity curve statistics, and breakdowns by day of week and month. Trades and the equity curve are also saved as CSVs to `./runs/`.

## OANDA Instruments and Granularity

- Index CFDs: `US500_USD`, `NAS100_USD`, `SPX500_USD`
- Forex: `EUR_USD`, `GBP_USD`, etc.
- Granularity: `M1`, `M5` (recommended for the OR window), `M15`, `H1`, `H4`, `D` (used for ATR)

See the [OANDA API documentation](https://developer.oanda.com/rest-live-v20/instrument-df/) for the full list.

## Common Pitfalls

- **DST handling** — the OR window is always evaluated in `America/New_York` local time (or whichever timezone you pass), correctly shifting across daylight saving transitions.
- **Index CFDs vs. ETFs** — this strategy targets OANDA index CFDs, which trade nearly 24/5; the session logic enforces the OR window regardless of when the underlying cash market opens.
- **Weekend gaps** — OANDA markets close Friday evening and reopen Sunday evening; the engine only processes days with a valid OR window.
- **Rate limits** — the OANDA client retries with exponential backoff. For large date ranges, keep caching enabled.
- **ATR data** — ATR uses daily candles when available, falling back to resampled intraday data otherwise. Days without enough history for the ATR lookback are skipped.

## Project Structure

```
.
├── README.md
├── pyproject.toml
├── scripts/
│   └── backtest.py          # CLI entry point
├── src/
│   ├── config/
│   │   ├── settings.py      # env-driven settings (API keys, cost defaults)
│   │   └── logging.py       # logging setup
│   ├── data/
│   │   ├── oanda_client.py  # OANDA v20 REST client with retry/backoff
│   │   └── fetch.py         # cached data fetching + daily resampling
│   ├── io/
│   │   └── cache.py         # local parquet cache for candle data
│   ├── utils/
│   │   ├── timezones.py     # UTC <-> local timezone conversion
│   │   ├── calendar.py      # daily session grouping, OR window extraction
│   │   └── math.py          # slippage/commission helpers
│   ├── indicators/
│   │   └── atr.py           # Average True Range
│   ├── strategy/
│   │   ├── params.py        # StrategyParams (validated config dataclass)
│   │   ├── opening_range.py # OR construction, bias, entry/stop/TP logic
│   │   └── filters.py       # ATR and other session filters
│   ├── risk/
│   │   └── position_sizing.py  # risk-based position sizing, daily risk cap
│   ├── execution/
│   │   └── costs.py         # spread/slippage/commission cost modeling
│   └── backtest/
│       ├── types.py         # Trade / BacktestResult dataclasses
│       ├── engine.py        # bar-by-bar simulation engine
│       ├── metrics.py       # performance metrics, equity curve stats, breakdowns
│       └── grid_search.py   # hyperparameter grid search over a training window
└── tests/
    └── ...                  # pytest suite covering each module above
```

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

```bash
black src tests scripts     # format
ruff check src tests scripts  # lint
mypy src                    # type check
```

## Extending Filters

To add a custom session filter, add a function to `src/strategy/filters.py`:

```python
def custom_filter(df_day, params):
    # Your filter logic
    return True  # or False
```

Then call it from `BacktestEngine.run()` in `src/backtest/engine.py`.

## Disclaimer

This project is for research and educational purposes. Backtested results do not guarantee future performance, and nothing here constitutes financial advice.

## License

MIT
