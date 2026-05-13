import logging
from django.core.cache import cache
from django.http import HttpResponse
from django.shortcuts import render
from django.views.decorators.http import require_GET

from analysis.engine import IntradayAnalyzer
from services.analysis_service import get_analysis_bundle
from services.market_data import (
    extract_ticker_frame,
    get_multi_ticker_data,
    get_previous_day_info,
    normalize_symbol,
    safe_cache_key,
)
from services.result_formatters import analysis_context, error_context, pick_context

# In-memory symbol cache for search
from stocks.models import Symbol
_symbol_cache = None
logger = logging.getLogger(__name__)
TOP_PICKS_CACHE_SECONDS = 15

def get_all_symbols():
    global _symbol_cache
    if _symbol_cache is None:
        _symbol_cache = list(Symbol.objects.all().values('ticker', 'name', 'exchange'))
    return _symbol_cache

# Nifty 50 watchlist (tickers without .NS suffix)
WATCHLIST = [
    'RELIANCE', 'TCS', 'HDFCBANK', 'INFY', 'ICICIBANK', 'HINDUNILVR', 'SBIN', 'BHARTIARTL',
    'KOTAKBANK', 'ITC', 'LT', 'DMART', 'SUNPHARMA', 'AXISBANK', 'MARUTI', 'TITAN',
    'WIPRO', 'HCLTECH', 'ASIANPAINT', 'ULTRACEMCO', 'BAJFINANCE', 'ADANIPORTS',
    'NTPC', 'POWERGRID', 'JSWSTEEL', 'TATASTEEL', 'HDFCLIFE', 'DRREDDY',
    'EICHERMOT', 'TECHM', 'BPCL', 'CIPLA', 'GAIL', 'ONGC', 'IOC', 'INDUSINDBK',
    'SHREECEM', 'HINDALCO', 'DIVISLAB', 'HEROMOTOCO', 'BRITANNIA',
    'GRASIM', 'VEDL', 'COALINDIA', 'UPL', 'BAJAJFINSV', 'JINDALSTEL',
    'BAJAJ-AUTO', 'TATAMOTORS', 'NESTLEIND'
]

def get_intraday_picks():
    """Return top 6 intraday picks from the watchlist (sorted by confidence)."""
    cache_key = 'top_intraday_picks'
    cached = cache.get(cache_key)
    if cached:
        return cached

    # Append .NS suffix
    tickers = [s + '.NS' for s in WATCHLIST]

    df_daily, df_intra = get_multi_ticker_data(tickers)
    if df_daily.empty or df_intra.empty:
        return []

    picks = []

    for symbol in tickers:
        daily = extract_ticker_frame(df_daily, symbol)
        intra = extract_ticker_frame(df_intra, symbol)

        if daily.empty or intra.empty:
            continue

        try:
            info = get_previous_day_info(daily)
            analyzer = IntradayAnalyzer(intra, info)
            result = analyzer.analyze()
        except Exception:
            logger.exception("Failed to analyze watchlist symbol %s", symbol)
            continue

        # Only include actionable picks (BUY/SELL)
        if result['recommendation'] != 'AVOID' and result['confidence'] >= 55:
            try:
                # Extract name from our local DB, fallback to ticker
                stock_obj = Symbol.objects.filter(ticker=symbol.replace('.NS', '')).first()
                name = stock_obj.name if stock_obj else symbol.replace('.NS', '')
            except Exception:
                name = symbol.replace('.NS', '')
            picks.append(pick_context(symbol, name, result))

    # Sort by confidence descending and take top 6
    picks = sorted(picks, key=lambda x: x['confidence'], reverse=True)[:6]
    cache.set(cache_key, picks, TOP_PICKS_CACHE_SECONDS)
    return picks


@require_GET
def top_picks_view(request):
    """HTMX endpoint to return the top picks HTML snippet."""
    picks = get_intraday_picks()
    return render(request, 'partials/top_picks.html', {'picks': picks})


@require_GET
def search_suggestions(request):
    query = request.GET.get('q', '').strip().upper()
    if not query or len(query) < 1:
        return render(request, 'partials/suggestions.html', {'symbols': []})

    symbols = get_all_symbols()
    # Split query into individual words; each must be found in ticker or name
    words = query.split()
    matches = []
    for s in symbols:
        ticker = s['ticker'].upper()
        name = s['name'].upper()
        # All words must be present in either the ticker or the name
        if all(word in ticker or word in name for word in words):
            matches.append(s)

    # Limit to 20 results (up from 10)
    matches = matches[:20]
    return render(request, 'partials/suggestions.html', {'symbols': matches})


@require_GET
def analyze_stock(request):
    symbol = request.GET.get('symbol', '').strip()
    if not symbol:
        return render(request, 'partials/error_state.html', error_context('No symbol provided.'))
    symbol = normalize_symbol(symbol)

    cache_key = safe_cache_key('analysis_html', symbol)
    cached = cache.get(cache_key)
    if cached:
        return HttpResponse(cached)

    try:
        bundle = get_analysis_bundle(symbol)
        if bundle.error:
            rendered = render(request, 'partials/error_state.html', error_context(bundle.error)).content.decode('utf-8')
            cache.set(cache_key, rendered, 120)
            return HttpResponse(rendered)

        context = analysis_context(bundle.snapshot, bundle.swing, bundle.intraday)
        rendered = render(request, 'partials/analysis_cards.html', context).content.decode('utf-8')
        cache.set(cache_key, rendered, 300)
        return HttpResponse(rendered)

    except Exception:
        logger.exception("Unexpected analysis error for %s", symbol)
        return render(request, 'partials/error_state.html', error_context('Analysis failed. Please try again later.'))


def home(request):
    return render(request, 'home.html')

# --- NEW ASYNC ENDPOINTS ---

@require_GET
def api_context(request):
    """
    Returns Market Context panel HTML (Trend, Volatility, Liquidity, HTF Bias).
    """
    symbol = request.GET.get('symbol', '').strip()
    if not symbol:
        return HttpResponse('')
        
    yf_symbol = normalize_symbol(symbol)
    cache_key = safe_cache_key("context_html", yf_symbol)
    
    cached = cache.get(cache_key)
    if cached:
        return HttpResponse(cached)
        
    try:
        bundle = get_analysis_bundle(yf_symbol)
        if bundle.error or not bundle.context_badges:
            return HttpResponse('')

        html = render(request, 'partials/context_badges.html', bundle.context_badges).content.decode('utf-8')
        cache.set(cache_key, html, 900) # 15 mins cache
        return HttpResponse(html)
    except Exception:
        logger.exception("Error api_context for %s", symbol)
        return HttpResponse('')

@require_GET
def api_insights(request):
    """
    Combines HTF context + trade signal logic into one rule-based paragraph.
    """
    symbol = request.GET.get('symbol', '').strip()
    if not symbol:
        return HttpResponse('')
        
    yf_symbol = normalize_symbol(symbol)
    cache_key = safe_cache_key("rule_insight_html", yf_symbol)
    
    cached = cache.get(cache_key)
    if cached:
        return HttpResponse(cached)
        
    try:
        bundle = get_analysis_bundle(yf_symbol)
        if bundle.error or not bundle.rule_insight:
            return HttpResponse('')

        html = render(request, 'partials/ai_insight.html', {'insight': bundle.rule_insight}).content.decode('utf-8')
        cache.set(cache_key, html, 900)
        return HttpResponse(html)
    except Exception:
        logger.exception("Error api_insights for %s", symbol)
        return HttpResponse('')
