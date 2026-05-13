from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import timedelta
from enum import StrEnum
import re

import pandas as pd
import yfinance as yf
from django.core.cache import cache
from django.utils import timezone

from analysis.data import normalize_ohlcv

logger = logging.getLogger(__name__)
WATCHLIST_CACHE_SECONDS = 15
SNAPSHOT_CACHE_SECONDS = 300
ERROR_CACHE_SECONDS = 60


class MarketDataStatus(StrEnum):
    OK = "ok"
    EMPTY = "empty"
    STALE = "stale"
    INSUFFICIENT = "insufficient"
    DOWNLOAD_ERROR = "download_error"
    INVALID_SYMBOL = "invalid_symbol"


@dataclass
class MarketSnapshot:
    symbol: str
    name: str
    ltp: float
    pct_change: float
    info: dict
    daily: pd.DataFrame
    intraday: pd.DataFrame
    error: str = ""
    status: MarketDataStatus = MarketDataStatus.OK

    @property
    def ok(self) -> bool:
        return not self.error and self.status == MarketDataStatus.OK


def normalize_symbol(symbol: str) -> str:
    cleaned = (symbol or "").strip().upper()
    if not cleaned:
        return ""
    if not cleaned.endswith((".NS", ".BO")):
        cleaned += ".NS"
    return cleaned


def safe_cache_key(prefix: str, *parts: object) -> str:
    cleaned = []
    for part in parts:
        text = str(part or "").strip().upper()
        text = re.sub(r"[^A-Z0-9_.:-]+", "_", text)
        cleaned.append(text[:80])
    return ":".join([prefix, *cleaned])


def _download(symbol: str, *, start=None, end=None, period=None, interval="1d") -> pd.DataFrame:
    try:
        return yf.download(
            symbol,
            start=start,
            end=end,
            period=period,
            interval=interval,
            progress=False,
            auto_adjust=False,
        )
    except Exception as exc:
        logger.warning("Failed to download %s %s data: %s", symbol, interval, exc)
        return pd.DataFrame()


def _status_from_quality(reason: str, *, stale: bool = False) -> MarketDataStatus:
    if stale:
        return MarketDataStatus.STALE
    if reason.startswith("Insufficient candles"):
        return MarketDataStatus.INSUFFICIENT
    if reason.startswith("No market data"):
        return MarketDataStatus.EMPTY
    return MarketDataStatus.DOWNLOAD_ERROR


def get_market_snapshot(symbol: str, *, now=None) -> MarketSnapshot:
    yf_symbol = normalize_symbol(symbol)
    if not yf_symbol:
        return MarketSnapshot("", "", 0.0, 0.0, {}, pd.DataFrame(), pd.DataFrame(), "No symbol provided.", MarketDataStatus.INVALID_SYMBOL)

    cache_key = safe_cache_key("market_snapshot", yf_symbol)
    cached = cache.get(cache_key)
    if cached:
        return cached

    end = now or timezone.now()
    daily_raw = _download(yf_symbol, start=end - timedelta(days=420), end=end, interval="1d")
    daily, daily_quality = normalize_ohlcv(daily_raw, symbol=yf_symbol, min_candles=60, max_stale_days=7, now=end)
    if not daily_quality.ok:
        snapshot = MarketSnapshot(
            yf_symbol, yf_symbol, 0.0, 0.0, {}, daily, pd.DataFrame(), daily_quality.reason,
            _status_from_quality(daily_quality.reason, stale=daily_quality.stale),
        )
        cache.set(cache_key, snapshot, ERROR_CACHE_SECONDS)
        return snapshot

    intra_raw = _download(yf_symbol, start=end - timedelta(days=7), end=end, interval="15m")
    intraday, intra_quality = normalize_ohlcv(intra_raw, symbol=yf_symbol, min_candles=30, max_stale_days=7, now=end)
    if not intra_quality.ok:
        intra_raw = _download(yf_symbol, start=end - timedelta(days=7), end=end, interval="5m")
        intraday, intra_quality = normalize_ohlcv(intra_raw, symbol=yf_symbol, min_candles=30, max_stale_days=7, now=end)
    if not intra_quality.ok:
        snapshot = MarketSnapshot(
            yf_symbol, yf_symbol, 0.0, 0.0, {}, daily, intraday, intra_quality.reason,
            _status_from_quality(intra_quality.reason, stale=intra_quality.stale),
        )
        cache.set(cache_key, snapshot, ERROR_CACHE_SECONDS)
        return snapshot

    info = {}
    try:
        info = yf.Ticker(yf_symbol).info or {}
    except Exception:
        logger.exception("Failed to fetch ticker info for %s", yf_symbol)

    ltp = float(info.get("regularMarketPrice") or daily["Close"].iloc[-1])
    prev_close = float(info.get("previousClose") or (daily["Close"].iloc[-2] if len(daily) > 1 else ltp))
    pct_change = ((ltp - prev_close) / prev_close) * 100 if prev_close else 0.0
    name = info.get("shortName") or yf_symbol

    snapshot = MarketSnapshot(
        symbol=yf_symbol,
        name=name,
        ltp=round(ltp, 2),
        pct_change=round(pct_change, 2),
        info=info,
        daily=daily,
        intraday=intraday,
    )
    cache.set(cache_key, snapshot, SNAPSHOT_CACHE_SECONDS)
    return snapshot


def get_previous_day_info(daily: pd.DataFrame) -> dict:
    prev_day = daily.iloc[-2] if len(daily) >= 2 else daily.iloc[-1]
    return {
        "regularMarketDayHigh": float(prev_day["High"]),
        "regularMarketDayLow": float(prev_day["Low"]),
        "previousClose": float(prev_day["Close"]),
    }


def get_multi_ticker_data(tickers: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    normalized = sorted(normalize_symbol(ticker) for ticker in tickers if normalize_symbol(ticker))
    cache_key = safe_cache_key("watchlist_market_data", ",".join(normalized))
    cached = cache.get(cache_key)
    if cached:
        return cached

    try:
        daily = yf.download(tickers=normalized, period="7d", interval="1d", group_by="ticker", progress=False)
        intraday = yf.download(tickers=normalized, period="7d", interval="15m", group_by="ticker", progress=False)
    except Exception as exc:
        logger.warning("Failed to download watchlist data: %s", exc)
        return pd.DataFrame(), pd.DataFrame()

    if daily.empty or intraday.empty:
        return pd.DataFrame(), pd.DataFrame()

    cache.set(cache_key, (daily, intraday), WATCHLIST_CACHE_SECONDS)
    return daily, intraday


def extract_ticker_frame(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    if isinstance(df.columns, pd.MultiIndex):
        for level in range(df.columns.nlevels):
            if symbol in df.columns.get_level_values(level):
                return df.xs(symbol, level=level, axis=1)
    return df.copy()
