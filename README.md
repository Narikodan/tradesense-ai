# TradeSense AI

A production-ready Django web application that provides intraday and swing trading suggestions using a rule-based technical analysis engine.

## Features
- **Zerodha‑style search** with HTMX autocomplete (NSE/BSE symbols)
- **Swing & Intraday trading cards** with entry, stop‑loss, targets, and confidence scores
- **Veteran trader logic**: EMA crossovers, RSI, MACD, Bollinger Bands, ATR, VWAP, pivot points, candlestick patterns
- Beautiful dark/light theme with Tailwind CSS
- Caching of API calls to avoid rate limits

## Quick Start

1. **Clone the repository** and navigate to the project folder.

2. **Create a virtual environment** (Python 3.11+):
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/macOS
   venv\Scripts\activate     # Windows