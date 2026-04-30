#!/usr/bin/env python
"""Fetch ALL NSE symbols (equity + ETFs) using nsetools and save as JSON."""
import json
import os
from nsetools import Nse

nse = Nse()
all_codes = nse.get_stock_codes()

if isinstance(all_codes, dict):
    if 'SYMBOL' in all_codes:
        del all_codes['SYMBOL']
    symbols = []
    for ticker, name in all_codes.items():
        if not ticker or not name:
            continue
        symbols.append({
            'symbol': ticker.strip(),
            'name': name.strip(),
            'exchange': 'NSE',
            'instrument_token': '',
            'sector': ''
        })
elif isinstance(all_codes, list):
    symbols = []
    for item in all_codes:
        if isinstance(item, str):
            ticker = item.strip()
            # Try to get the company name from the quote (may fail for some)
            try:
                quote = nse.get_quote(ticker)
                name = quote.get('companyName', ticker)
            except:
                name = ticker
        elif isinstance(item, dict):
            ticker = item.get('symbol', '').strip()
            name = item.get('companyName', ticker).strip()
        else:
            continue
        if ticker and name:
            symbols.append({
                'symbol': ticker,
                'name': name,
                'exchange': 'NSE',
                'instrument_token': '',
                'sector': ''
            })
else:
    print("Unexpected data type:", type(all_codes))
    exit(1)

output_path = os.path.join('stocks', 'fixtures', 'nse_symbols.json')
os.makedirs(os.path.dirname(output_path), exist_ok=True)
with open(output_path, 'w') as f:
    json.dump(symbols, f, indent=2)

print(f"✅ Saved {len(symbols)} symbols (including ETFs) to {output_path}")