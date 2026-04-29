#!/usr/bin/env python
"""Fetch all NSE symbols using nsetools and save as JSON."""
import json
import os
from nsetools import Nse

nse = Nse()
all_codes = nse.get_stock_codes()   # could be dict or list

symbols = []

if isinstance(all_codes, dict):
    # Remove the index name entry if it exists
    if 'SYMBOL' in all_codes:
        del all_codes['SYMBOL']
    for ticker, name in all_codes.items():
        if not ticker or not name:
            continue
        symbols.append({
            'symbol': ticker.strip(),
            'name': name.strip() if isinstance(name, str) else str(name),
            'exchange': 'NSE',
            'instrument_token': '',
            'sector': ''
        })
elif isinstance(all_codes, list):
    # If it's a list, elements might be just ticker strings
    for item in all_codes:
        if isinstance(item, str):
            ticker = item.strip()
            # For name we could use the ticker itself, or leave blank,
            # but it's better to have a name. We can try to get it via
            # nse.get_quote(ticker) but that's slow. We'll just set name = ticker
            name = ticker
        elif isinstance(item, dict):
            ticker = item.get('symbol', '').strip()
            name = item.get('companyName', ticker).strip()
        else:
            continue
        if ticker:
            symbols.append({
                'symbol': ticker,
                'name': name,
                'exchange': 'NSE',
                'instrument_token': '',
                'sector': ''
            })
else:
    print("Unexpected data type from nse.get_stock_codes():", type(all_codes))
    exit(1)

output_path = os.path.join('stocks', 'fixtures', 'nse_symbols.json')
os.makedirs(os.path.dirname(output_path), exist_ok=True)
with open(output_path, 'w') as f:
    json.dump(symbols, f, indent=2)

print(f"✅ Saved {len(symbols)} symbols to {output_path}")