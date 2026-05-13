from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Protocol

from django.conf import settings


DEFAULT_INTRADAY_SHORTABLE_SYMBOLS = {
    "RELIANCE",
    "TCS",
    "INFY",
    "HDFCBANK",
    "ICICIBANK",
    "SBIN",
    "AXISBANK",
    "KOTAKBANK",
    "LT",
    "ITC",
    "HINDUNILVR",
    "BHARTIARTL",
    "BAJFINANCE",
    "MARUTI",
    "TATAMOTORS",
    "SUNPHARMA",
    "ADANIENT",
    "ADANIPORTS",
    "WIPRO",
    "TECHM",
}


class ShortSellEligibilityService(Protocol):
    def is_shortable(self, symbol: str) -> bool:
        ...

    def reason(self, symbol: str) -> str:
        ...


@dataclass(frozen=True)
class StaticShortSellEligibility:
    """
    Local allow-list based short-sell eligibility.

    This intentionally keeps broker/exchange-specific logic behind a tiny
    service interface so a live broker or exchange ban-list adapter can replace
    the static list later.
    """

    symbols: frozenset[str]

    @classmethod
    def from_settings(cls, extra_symbols: Iterable[str] | None = None) -> "StaticShortSellEligibility":
        configured = getattr(settings, "INTRADAY_SHORT_SELL_SYMBOLS", DEFAULT_INTRADAY_SHORTABLE_SYMBOLS)
        symbols = set(configured or ())
        if extra_symbols:
            symbols.update(extra_symbols)
        return cls(frozenset(_normalize_symbol(symbol) for symbol in symbols if symbol))

    def is_shortable(self, symbol: str) -> bool:
        return _normalize_symbol(symbol) in self.symbols

    def reason(self, symbol: str) -> str:
        return f"{_normalize_symbol(symbol) or 'UNKNOWN'} is not configured as intraday short-sell eligible."


def _normalize_symbol(symbol: str | None) -> str:
    value = (symbol or "").upper().strip()
    for suffix in (".NS", ".BO"):
        if value.endswith(suffix):
            value = value[: -len(suffix)]
    return value
