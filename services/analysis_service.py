from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from django.core.cache import cache

from analysis.engine import IntradayAnalyzer, SwingAnalyzer
from services.market_data import MarketSnapshot, get_market_snapshot, get_previous_day_info, normalize_symbol, safe_cache_key


ANALYSIS_CACHE_SECONDS = 300
BADGE_CACHE_SECONDS = 900
RULE_INSIGHT_CACHE_SECONDS = 900


@dataclass(frozen=True)
class AnalysisBundle:
    snapshot: MarketSnapshot
    swing: dict
    intraday: dict
    context_badges: dict
    rule_insight: str
    error: str = ""


def get_analysis_bundle(symbol: str, *, now: pd.Timestamp | None = None, use_cache: bool = True) -> AnalysisBundle:
    yf_symbol = normalize_symbol(symbol)
    if not yf_symbol:
        snapshot = MarketSnapshot("", "", 0.0, 0.0, {}, pd.DataFrame(), pd.DataFrame(), "No symbol provided.")
        return AnalysisBundle(snapshot, {}, {}, {}, "", snapshot.error)

    cache_key = safe_cache_key("analysis_bundle", yf_symbol)
    if use_cache:
        cached = cache.get(cache_key)
        if cached:
            return cached

    snapshot = get_market_snapshot(yf_symbol, now=now)
    if snapshot.error:
        bundle = AnalysisBundle(snapshot, {}, {}, {}, "", snapshot.error)
        cache.set(cache_key, bundle, 120)
        return bundle

    swing = SwingAnalyzer(snapshot.daily).analyze()
    intra_info = get_previous_day_info(snapshot.daily) | snapshot.info | {"symbol": yf_symbol}
    intraday = IntradayAnalyzer(snapshot.intraday, intra_info, htf_df=snapshot.daily, now=now).analyze()
    context_badges = build_context_badges(snapshot.daily, swing)
    rule_insight = build_rule_insight(snapshot.daily, swing)

    bundle = AnalysisBundle(snapshot, swing, intraday, context_badges, rule_insight)
    cache.set(cache_key, bundle, ANALYSIS_CACHE_SECONDS)
    return bundle


def build_context_badges(daily: pd.DataFrame, swing_result: dict) -> dict:
    if daily.empty:
        return {}

    df = daily.copy()
    avg_vol = df["Volume"].rolling(10).mean().iloc[-2] if len(df) > 10 else df["Volume"].mean()
    cur_vol = df["Volume"].iloc[-1]
    liquidity = "HIGH" if cur_vol > avg_vol * 1.2 else "LOW" if cur_vol < avg_vol * 0.8 else "NORMAL"

    ranges = df["High"] - df["Low"]
    avg_range = ranges.rolling(14).mean().iloc[-2] if len(df) > 14 else ranges.mean()
    cur_range = ranges.iloc[-1]
    volatility = "HIGH" if cur_range > avg_range * 1.5 else "LOW" if cur_range < avg_range * 0.6 else "NORMAL"

    ema21 = df["Close"].ewm(span=21, adjust=False).mean().iloc[-1]
    trend = "BULLISH" if df["Close"].iloc[-1] > ema21 else "BEARISH"

    return {
        "trend": trend,
        "volatility": volatility,
        "liquidity": liquidity,
        "htf_bias": swing_result.get("recommendation", "AVOID"),
    }


def build_rule_insight(daily: pd.DataFrame, swing_result: dict) -> str:
    if daily.empty:
        return ""

    df = daily.copy()
    ema21 = df["Close"].ewm(span=21, adjust=False).mean().iloc[-1]
    trend = "bullish trend" if df["Close"].iloc[-1] > ema21 else "bearish trend"

    avg_vol = df["Volume"].rolling(10).mean().iloc[-2] if len(df) > 10 else df["Volume"].mean()
    volume = "strong volume" if df["Volume"].iloc[-1] > avg_vol * 1.2 else "average volume"
    htf_bias = swing_result.get("recommendation", "AVOID")

    text = f"Rule-based context: stock is currently in a {trend} showing {volume}. "
    if htf_bias == "BUY":
        return text + "Higher timeframe bias supports long setups on pullbacks, subject to risk controls."
    if htf_bias == "SELL":
        return text + "Higher timeframe bias is weak, so short structures may be cleaner if confirmed."
    return text + "Awaiting clearer structural breakout before treating the setup as actionable."
