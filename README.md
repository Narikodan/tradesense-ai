# TradeSense AI

A production-ready Django web application that provides intraday and swing trading suggestions using a rule-based technical analysis engine.

## Features
- **Zerodha‑style search** with HTMX autocomplete (NSE/BSE symbols)
- **Swing & Intraday trading cards** with entry, stop‑loss, targets, and confidence scores
- **Disciplined trading logic**: EMA trend, RSI momentum, ATR/VWAP/pivots, market regime, liquidity, R:R gates, and AVOID outcomes
- **Risk guidance**: stop-loss, two targets, R:R, invalidation, trade grade, and optional position sizing
- **Data guards**: OHLCV normalization, empty/stale data handling, yfinance failure handling, and cached market-data calls
- Beautiful dark/light theme with Tailwind CSS
- Caching of API calls to avoid rate limits

## Environment

Set these values outside source control for deployed environments:

```bash
export DJANGO_SECRET_KEY="replace-me"
export DJANGO_DEBUG="False"
export DJANGO_ALLOWED_HOSTS="your-domain.com,127.0.0.1"
export DJANGO_LOG_LEVEL="INFO"
export DJANGO_STATIC_ROOT="/srv/tradesense/staticfiles"
export DJANGO_REDIS_URL="redis://127.0.0.1:6379/1"  # optional; DB cache is the default
export DJANGO_SECURE_PROXY_SSL_HEADER="True"         # when behind an HTTPS proxy
export DJANGO_SECURE_SSL_REDIRECT="True"
```

The current frontend still loads Tailwind and HTMX from public CDNs. For offline or locked-down production deployments, vendor those assets into `static/` and update `core/templates/base.html` to use `{% static %}` paths.

## Commands

```bash
python manage.py migrate
python manage.py test
python manage.py runserver
```

## Trading System Notes

BUY/SELL is only emitted when trend, momentum, volume/liquidity, structure, and risk/reward are aligned. Otherwise the system returns AVOID with reasons. Trade quality is graded A/B/SKIP, with SKIP forcing AVOID. The included `analysis/backtest.py` module is only scaffolding for walk-forward validation and is not a complete execution simulator.

## Disclaimer

Outputs are educational decision-support only and are not financial advice, recommendations, or guarantees of future performance. Validate independently before risking capital.

## Quick Start

1. **Clone the repository** and navigate to the project folder.

2. **Create a virtual environment** (Python 3.11+):
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/macOS
   venv\Scripts\activate     # Windows
