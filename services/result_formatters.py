from __future__ import annotations


def error_context(message: str) -> dict:
    return {
        "error": message,
        "disclaimer": "Educational decision-support only, not financial advice.",
    }


def analysis_context(snapshot, swing: dict, intraday: dict) -> dict:
    return {
        "name": snapshot.name,
        "symbol": snapshot.symbol,
        "ltp": snapshot.ltp,
        "pct_change": snapshot.pct_change,
        "swing": swing,
        "intraday": intraday,
        "disclaimer": "Educational decision-support only, not financial advice.",
    }


def pick_context(symbol: str, name: str, result: dict) -> dict:
    display_recommendation = result.get("display_recommendation") or (
        "SHORT SELL" if result.get("recommendation") == "SHORT_SELL" else result.get("recommendation")
    )
    return {
        "symbol": symbol.replace(".NS", ""),
        "name": name,
        "ltp": result["entry"],
        "recommendation": result["recommendation"],
        "display_recommendation": display_recommendation,
        "confidence": result["confidence"],
        "trade_quality": result.get("trade_quality", "SKIP"),
        "entry": result["entry"],
        "target1": result["target1"],
        "stop_loss": result["stop_loss"],
        "rr": result["rr"],
        "reason": (result.get("reason") or "")[:100] + "...",
    }
