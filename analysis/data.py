from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


REQUIRED_OHLC = ("Open", "High", "Low", "Close")
OHLCV_COLUMNS = ("Open", "High", "Low", "Close", "Volume")


@dataclass(frozen=True)
class DataQuality:
    ok: bool
    reason: str = ""
    stale: bool = False
    candles: int = 0


def normalize_ohlcv(
    df: pd.DataFrame | None,
    *,
    symbol: str | None = None,
    min_candles: int = 50,
    max_stale_days: int | None = None,
    now: pd.Timestamp | None = None,
) -> tuple[pd.DataFrame, DataQuality]:
    """
    Normalize yfinance-style OHLCV data for deterministic analysis.

    Handles empty data, MultiIndex columns, adjusted-close extras, missing
    volume, duplicate rows, non-numeric columns, and stale daily candles.
    """
    if df is None or df.empty:
        return pd.DataFrame(columns=OHLCV_COLUMNS), DataQuality(False, "No market data returned.", candles=0)

    normalized = df.copy()

    if isinstance(normalized.columns, pd.MultiIndex):
        symbol_level = None
        if symbol:
            for level in range(normalized.columns.nlevels):
                if symbol in normalized.columns.get_level_values(level):
                    symbol_level = level
                    break
        if symbol_level is not None:
            normalized = normalized.xs(symbol, level=symbol_level, axis=1)
        else:
            flattened: list[str] = []
            for col in normalized.columns:
                parts = [str(part) for part in col if str(part) and str(part) != "nan"]
                flattened.append("_".join(parts))
            normalized.columns = flattened

    rename_map: dict[str, str] = {}
    for col in normalized.columns:
        name = str(col)
        base = name.split("_")[0]
        if base in OHLCV_COLUMNS and base not in rename_map.values():
            rename_map[col] = base
        elif base == "Adj Close" and "Close" not in normalized.columns:
            rename_map[col] = "Close"
    normalized = normalized.rename(columns=rename_map)

    missing = [col for col in REQUIRED_OHLC if col not in normalized.columns]
    if missing:
        return pd.DataFrame(columns=OHLCV_COLUMNS), DataQuality(
            False,
            f"Missing required OHLC columns: {', '.join(missing)}.",
            candles=len(normalized),
        )

    if "Volume" not in normalized.columns:
        normalized["Volume"] = 0

    normalized = normalized[list(OHLCV_COLUMNS)].copy()
    for col in OHLCV_COLUMNS:
        normalized[col] = pd.to_numeric(normalized[col], errors="coerce")

    normalized = normalized.replace([float("inf"), float("-inf")], pd.NA)
    normalized = normalized.dropna(subset=list(REQUIRED_OHLC))
    normalized["Volume"] = normalized["Volume"].fillna(0).clip(lower=0)
    normalized = normalized[normalized["Close"] > 0]
    normalized = normalized[normalized["High"] >= normalized["Low"]]
    normalized = normalized[~normalized.index.duplicated(keep="last")].sort_index()

    candles = len(normalized)
    if candles < min_candles:
        return normalized, DataQuality(
            False,
            f"Insufficient candles: {candles} available, {min_candles} required.",
            candles=candles,
        )

    stale = False
    if max_stale_days is not None and isinstance(normalized.index, pd.DatetimeIndex):
        last_ts = normalized.index[-1]
        if now is None:
            now = pd.Timestamp.now(tz=last_ts.tzinfo) if last_ts.tzinfo else pd.Timestamp.now()
        else:
            now = pd.Timestamp(now)
            if last_ts.tzinfo and now.tzinfo is None:
                now = now.tz_localize(last_ts.tzinfo)
            elif last_ts.tzinfo is None and now.tzinfo is not None:
                now = now.tz_localize(None)
            elif last_ts.tzinfo and now.tzinfo:
                now = now.tz_convert(last_ts.tzinfo)
        stale = (now.normalize() - last_ts.normalize()).days > max_stale_days
        if stale:
            return normalized, DataQuality(
                False,
                f"Stale market data: last candle is {last_ts.date()}.",
                stale=True,
                candles=candles,
            )

    return normalized, DataQuality(True, candles=candles)
