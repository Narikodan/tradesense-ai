import json
from datetime import datetime, timedelta
import pandas as pd
import yfinance as yf
from django.core.cache import cache
from django.http import HttpResponse
from django.shortcuts import render
from django.views.decorators.http import require_GET

from analysis.engine import SwingAnalyzer, IntradayAnalyzer

# In-memory symbol list (loaded once from DB to avoid repeated queries)
_symbol_cache = None

def get_all_symbols():
    """Load symbols from DB and cache them in memory."""
    global _symbol_cache
    if _symbol_cache is None:
        from stocks.models import Symbol
        _symbol_cache = list(Symbol.objects.all().values('ticker', 'name', 'exchange'))
    return _symbol_cache

@require_GET
def search_suggestions(request):
    """Return an HTML snippet of matching symbols for HTMX autocomplete."""
    query = request.GET.get('q', '').strip().upper()
    if not query or len(query) < 1:
        return HttpResponse('<div class="text-gray-500 p-2">Type to search...</div>')
    symbols = get_all_symbols()
    matches = [s for s in symbols if query in s['ticker'] or query in s['name'].upper()]
    # Limit to 10 results
    matches = matches[:10]
    if not matches:
        return HttpResponse('<div class="text-gray-500 p-2">No results found</div>')
    # Render a snippet
    html = '<div class="bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-700 rounded-lg shadow-lg max-h-60 overflow-y-auto">'
    for m in matches:
        html += f'''<div class="px-4 py-2 hover:bg-blue-50 dark:hover:bg-gray-700 cursor-pointer suggestion-item"
                       data-symbol="{m['ticker']}.NS"
                       data-name="{m['name']}">
                       {m['ticker']} - {m['name']} ({m['exchange']})
                    </div>'''
    html += '</div>'
    return HttpResponse(html)

@require_GET
def analyze_stock(request):
    """Fetch data, run analyzers, return partial HTML with two cards."""
    symbol = request.GET.get('symbol', '').strip()
    if not symbol:
        return HttpResponse('<div class="text-red-500">No symbol provided.</div>')
    # Ensure .NS suffix for NSE
    if not symbol.endswith('.NS') and not symbol.endswith('.BO'):
        symbol += '.NS'  # default NSE

    cache_key = f'analysis_{symbol}'
    cached = cache.get(cache_key)
    if cached:
        # If cached, return the pre-rendered HTML
        return HttpResponse(cached)

    try:
        # Download data
        # For intraday we need recent 15-min data. yfinance's interval "15m" gives up to 60 days.
        end = datetime.now()
        start_daily = end - timedelta(days=365)
        start_intra = end - timedelta(days=6)  # get 5 days + some buffer

        # Daily data
        df_daily = yf.download(symbol, start=start_daily, end=end, interval='1d', progress=False)
        if df_daily.empty:
            return HttpResponse('<div class="text-red-500">No data found for this symbol.</div>')

        # Intraday 15m data
        df_intra = yf.download(symbol, start=start_intra, end=end, interval='15m', progress=False)
        if df_intra.empty:
            # Fallback to 5m if 15m not available?
            df_intra = yf.download(symbol, start=start_intra, end=end, interval='5m', progress=False)
            if df_intra.empty:
                return HttpResponse('<div class="text-red-500">No intraday data available.</div>')

        # Get current price and day change
        ticker = yf.Ticker(symbol)
        info = ticker.info
        ltp = info.get('regularMarketPrice', df_daily['Close'].iloc[-1])
        prev_close = info.get('previousClose', df_daily['Close'].iloc[-2] if len(df_daily)>1 else ltp)
        pct_change = ((ltp - prev_close) / prev_close) * 100 if prev_close else 0

        name = info.get('shortName', symbol)

        # Run analysis engines
        swing_result = SwingAnalyzer(df_daily).analyze()
        intra_result = IntradayAnalyzer(df_intra, info).analyze()

        # Build context for partial template
        context = {
            'name': name,
            'symbol': symbol,
            'ltp': round(ltp, 2),
            'pct_change': round(pct_change, 2),
            'swing': swing_result,
            'intraday': intra_result,
        }
        rendered = render(request, 'partials/analysis_cards.html', context).content.decode('utf-8')
        # Cache for 5 minutes (300s)
        cache.set(cache_key, rendered, 300)
        return HttpResponse(rendered)

    except Exception as e:
        return HttpResponse(f'<div class="text-red-500">Error: {str(e)}</div>')

def home(request):
    return render(request, 'home.html')