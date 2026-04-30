import json
from datetime import datetime, timedelta
import pandas as pd
import yfinance as yf
from django.core.cache import cache
from django.http import HttpResponse, JsonResponse
from django.shortcuts import render
from django.views.decorators.http import require_GET
from concurrent.futures import ThreadPoolExecutor

from analysis.engine import IntradayAnalyzer, compute_pivot_points

# In-memory symbol cache for search
from stocks.models import Symbol
_symbol_cache = None

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

    try:
        # Download daily data (for previous day's pivot levels)
        df_daily = yf.download(
            tickers=tickers, period='5d', interval='1d', group_by='ticker', progress=False
        )
        # Download intraday 15m data
        df_intra = yf.download(
            tickers=tickers, period='5d', interval='15m', group_by='ticker', progress=False
        )
    except Exception as e:
        return []  # failed to fetch; show nothing

    picks = []

    for symbol in tickers:
        # Extract DataFrames per ticker; yfinance may return a simple DataFrame if only one ticker
        if len(tickers) == 1:
            # yfinance returns a single DataFrame (no MultiIndex) if only one ticker
            daily = df_daily.copy()
            intra = df_intra.copy()
        else:
            try:
                daily = df_daily[symbol]
                intra = df_intra[symbol]
            except KeyError:
                continue

        if daily.empty or intra.empty:
            continue

        try:
            # Previous day's OHLC for pivot points
            prev_day = daily.iloc[-2] if len(daily) >= 2 else daily.iloc[-1]
            info = {
                'regularMarketDayHigh': prev_day['High'],
                'regularMarketDayLow': prev_day['Low'],
                'previousClose': prev_day['Close']
            }
            analyzer = IntradayAnalyzer(intra, info)
            result = analyzer.analyze()
        except Exception:
            continue

        # Only include actionable picks (BUY/SELL)
        if result['recommendation'] != 'AVOID' and result['confidence'] >= 40:
            try:
                # Extract name from our local DB, fallback to ticker
                stock_obj = Symbol.objects.filter(ticker=symbol.replace('.NS', '')).first()
                name = stock_obj.name if stock_obj else symbol.replace('.NS', '')
            except:
                name = symbol.replace('.NS', '')
            picks.append({
                'symbol': symbol.replace('.NS', ''),
                'name': name,
                'ltp': result['entry'],
                'recommendation': result['recommendation'],
                'confidence': result['confidence'],
                'entry': result['entry'],
                'target1': result['target1'],
                'stop_loss': result['stop_loss'],
                'reason': result['reason'][:80] + '...',
            })

    # Sort by confidence descending and take top 6
    picks = sorted(picks, key=lambda x: x['confidence'], reverse=True)[:6]
    cache.set(cache_key, picks, 300)  # 5 minutes
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
        return HttpResponse('<div class="text-red-500">No symbol provided.</div>')
    if not symbol.endswith('.NS') and not symbol.endswith('.BO'):
        symbol += '.NS'

    cache_key = f'analysis_{symbol}'
    cached = cache.get(cache_key)
    if cached:
        return HttpResponse(cached)

    try:
        end = datetime.now()
        start_daily = end - timedelta(days=365)
        start_intra = end - timedelta(days=6)

        df_daily = yf.download(symbol, start=start_daily, end=end, interval='1d', progress=False)
        if df_daily.empty:
            return HttpResponse('<div class="text-red-500">No daily data found.</div>')

        df_intra = yf.download(symbol, start=start_intra, end=end, interval='15m', progress=False)
        if df_intra.empty:
            df_intra = yf.download(symbol, start=start_intra, end=end, interval='5m', progress=False)
            if df_intra.empty:
                return HttpResponse('<div class="text-red-500">No intraday data available.</div>')

        ticker = yf.Ticker(symbol)
        info = ticker.info
        ltp = info.get('regularMarketPrice', df_daily['Close'].iloc[-1])
        prev_close = info.get('previousClose', df_daily['Close'].iloc[-2] if len(df_daily)>1 else ltp)
        pct_change = ((ltp - prev_close) / prev_close) * 100 if prev_close else 0

        name = info.get('shortName', symbol)

        from analysis.engine import SwingAnalyzer
        swing_result = SwingAnalyzer(df_daily).analyze()
        intra_result = IntradayAnalyzer(df_intra, info).analyze()

        context = {
            'name': name,
            'symbol': symbol,
            'ltp': round(ltp, 2),
            'pct_change': round(pct_change, 2),
            'swing': swing_result,
            'intraday': intra_result,
        }
        rendered = render(request, 'partials/analysis_cards.html', context).content.decode('utf-8')
        cache.set(cache_key, rendered, 300)
        return HttpResponse(rendered)

    except Exception as e:
        return HttpResponse(f'<div class="text-red-500">Error: {str(e)}</div>')


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
        
    yf_symbol = symbol if (symbol.endswith('.NS') or symbol.endswith('.BO')) else f"{symbol}.NS"
    cache_key = f"context_{yf_symbol}"
    
    cached = cache.get(cache_key)
    if cached:
        return HttpResponse(cached)
        
    try:
        df = yf.download(yf_symbol, period='1mo', interval='1d', progress=False)
        if df.empty:
            return HttpResponse('')
            
        # Simplify if MultiIndex
        if isinstance(df.columns, pd.MultiIndex):
            if yf_symbol in df.columns.get_level_values(1):
                df = df.xs(yf_symbol, level=1, axis=1)
            else:
                df.columns = df.columns.droplevel(1)
                
        # Liquidity (Volume avg 10d vs cur)
        avg_vol = df['Volume'].rolling(10).mean().iloc[-2]
        cur_vol = df['Volume'].iloc[-1]
        liquidity = "HIGH" if cur_vol > avg_vol * 1.2 else "LOW" if cur_vol < avg_vol * 0.8 else "NORMAL"
        
        # Volatility (ATR roughly) -> High-Low rolling vs cur
        df['range'] = df['High'] - df['Low']
        avg_range = df['range'].rolling(14).mean().iloc[-2]
        cur_range = df['range'].iloc[-1]
        volatility = "HIGH" if cur_range > avg_range * 1.5 else "LOW" if cur_range < avg_range * 0.6 else "NORMAL"
        
        # Trend
        df['EMA21'] = df['Close'].ewm(span=21, adjust=False).mean()
        last_close = df['Close'].iloc[-1]
        ema21 = df['EMA21'].iloc[-1]
        trend = "BULLISH" if last_close > ema21 else "BEARISH"
        
        # HTF Bias
        from analysis.engine import SwingAnalyzer
        swing = SwingAnalyzer(df).analyze()
        htf_bias = swing.get('recommendation', 'AVOID')
        
        context_data = {
            'trend': trend,
            'volatility': volatility,
            'liquidity': liquidity,
            'htf_bias': htf_bias
        }
        
        html = render(request, 'partials/context_badges.html', context_data).content.decode('utf-8')
        cache.set(cache_key, html, 900) # 15 mins cache
        return HttpResponse(html)
    except Exception as e:
        print(f"Error api_context: {e}")
        return HttpResponse('')

@require_GET
def api_insights(request):
    """
    Combines HTF context + trade signal logic into one AI-like paragraph.
    """
    symbol = request.GET.get('symbol', '').strip()
    if not symbol:
        return HttpResponse('')
        
    yf_symbol = symbol if (symbol.endswith('.NS') or symbol.endswith('.BO')) else f"{symbol}.NS"
    cache_key = f"insights_{yf_symbol}"
    
    cached = cache.get(cache_key)
    if cached:
        return HttpResponse(cached)
        
    try:
        df = yf.download(yf_symbol, period='1mo', interval='1d', progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            if yf_symbol in df.columns.get_level_values(1):
                df = df.xs(yf_symbol, level=1, axis=1)
            else:
                df.columns = df.columns.droplevel(1)
        
        df['EMA21'] = df['Close'].ewm(span=21, adjust=False).mean()
        trend = "bullish trend" if df['Close'].iloc[-1] > df['EMA21'].iloc[-1] else "bearish trend"
        
        avg_vol = df['Volume'].rolling(10).mean().iloc[-2]
        vol = "strong volume" if df['Volume'].iloc[-1] > avg_vol * 1.2 else "average volume"
        
        # We can fetch the recent recommendation from analyzer
        from analysis.engine import SwingAnalyzer
        swing = SwingAnalyzer(df).analyze()
        htf_bias = swing.get('recommendation', 'AVOID')
        
        insight_text = f"Stock is currently in a {trend} showing {vol}. "
            
        if htf_bias == 'BUY':
            insight_text += "Higher timeframe bias supports long entries on pullbacks."
        elif htf_bias == 'SELL':
            insight_text += "Higher timeframe bias is weak, short structures are favorable."
        else:
            insight_text += "Awaiting clearer structural breakout before major moves."
            
        context_data = {'insight': insight_text}
        
        html = render(request, 'partials/ai_insight.html', context_data).content.decode('utf-8')
        cache.set(cache_key, html, 900)
        return HttpResponse(html)
    except Exception as e:
        print(f"Error api_insights: {e}")
        return HttpResponse('')