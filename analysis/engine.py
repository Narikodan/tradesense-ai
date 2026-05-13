"""
TradeSense AI – Analysis Engine (v3 — strategic entry)
All original logic preserved; each version is strictly additive.

New capabilities (v2):
  1. Market-regime detector (ADX-14) + volatility filter + news-blackout flag
  2. Multi-timeframe confluence via optional htf_df
  3. Structure-based stop loss (last-3-candle swing) with ATR fallback
  4. Intraday POC (volume-profile approximation, 20 bins)
  5. 3-zone IST session weighting (opening / midday / closing)
  6. Sigmoid confidence calibration, capped at 95
  7. Risk-of-ruin guard → trade_quality + skip_trade
  8. Extra candlestick patterns: inside_bar, morning_star, evening_star
     + new consolidating output flag

New capabilities (v3) — strategic entry engine:
  entry is no longer the last close.  It is computed by _strategic_entry()
  which selects one of three entry styles based on context:

  'breakout'  — price just cleared a structural level (ORB, pivot, prior S/R).
                 Entry = that level +/- a 0.1×ATR confirmation buffer so we
                 don't chase a candle that is already extended.

  'pullback'  — price is trending but not at a breakout; wait for it to
                 retrace to the nearest confluence zone (VWAP / POC / EMA21).
                 Entry = that zone ± 0.05×ATR.  If price is already AT the
                 zone (within 0.15×ATR), entry = current close.

  'rejection' — a reversal candlestick pattern (hammer, engulf, star) fired
                 at a known S/R level.  Entry = high of signal candle + 0.05×ATR
                 for BUY, low of signal candle - 0.05×ATR for SELL.
                 This is a confirmation trigger, not a market-order entry.

  Priority order (highest confidence first):
    1. rejection  — if a qualifying pattern is present at a S/R level
    2. breakout   — if price is within 0.5×ATR of a structural breakout level
    3. pullback   — default for all other trending signals

  New output keys:  entry_type ('breakout' | 'pullback' | 'rejection')
                    entry_note  (human-readable rationale for the entry level)

  All risk/reward/stop calculations are rebased onto the strategic entry
  price, not the last close, so the R:R ratio reflects what the trader
  will actually experience when the order triggers.
"""

from __future__ import annotations

import math
import datetime
from typing import Optional

import numpy as np
import pandas as pd
from ta.momentum import RSIIndicator
from ta.trend import EMAIndicator, MACD
from ta.volatility import BollingerBands, AverageTrueRange

from analysis.data import normalize_ohlcv
from analysis.short_selling import StaticShortSellEligibility, ShortSellEligibilityService

# ──────────────────────────────────────────────────────────────────────────────
# Helper: ADX (no new pip packages – pure pandas/numpy)
# ──────────────────────────────────────────────────────────────────────────────

def compute_adx(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
    """
    Compute Average Directional Index (ADX) without external TA-Lib.

    Returns a Series aligned to the input index.
    """
    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low  - close.shift(1)).abs(),
    ], axis=1).max(axis=1)

    up_move   = high - high.shift(1)
    down_move = low.shift(1) - low

    plus_dm  = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    atr_s    = tr.ewm(span=window, adjust=False).mean()
    plus_di  = 100 * pd.Series(plus_dm,  index=high.index).ewm(span=window, adjust=False).mean() / atr_s
    minus_di = 100 * pd.Series(minus_dm, index=high.index).ewm(span=window, adjust=False).mean() / atr_s

    dx = (100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)).fillna(0)
    adx = dx.ewm(span=window, adjust=False).mean()
    return adx


# ──────────────────────────────────────────────────────────────────────────────
# Helper: sigmoid confidence (Upgrade 6)
# ──────────────────────────────────────────────────────────────────────────────

def sigmoid_confidence(score: float) -> int:
    """
    Map a raw score to [0, 95] using a sigmoid curve.

    confidence = int(100 / (1 + exp(-score * 0.9))), capped at 95.
    Markets are never 100 % certain.
    """
    raw = 100.0 / (1.0 + math.exp(-score * 0.9))
    return min(95, int(raw))


# ──────────────────────────────────────────────────────────────────────────────
# Helper: VWAP
# ──────────────────────────────────────────────────────────────────────────────

def compute_vwap(df: pd.DataFrame) -> pd.DataFrame:
    """Volume Weighted Average Price on the given DataFrame."""
    df = df.copy()
    cumulative_volume = df["Volume"].cumsum().replace(0, np.nan)
    df["VWAP"] = (
        df["Volume"] * ((df["High"] + df["Low"] + df["Close"]) / 3)
    ).cumsum() / cumulative_volume
    df["VWAP"] = df["VWAP"].ffill().fillna(df["Close"])
    return df


def _empty_result(reason: str, *, session: str = "n/a", price: float = 0.0) -> dict:
    price = round(float(price or 0), 2)
    return {
        "recommendation": "AVOID",
        "confidence": 0,
        "entry": price,
        "stop_loss": price,
        "sl_percent": 0.0,
        "target1": price,
        "target2": price,
        "rr": 0.0,
        "support1": price,
        "support2": price,
        "resistance1": price,
        "resistance2": price,
        "reason": reason,
        "why_trade": [],
        "why_avoid": [reason],
        "invalidation": "No trade is valid until data quality improves.",
        "position_size": {"shares": 0, "capital_at_risk": 0.0, "risk_per_share": 0.0, "note": reason},
        "trade_quality": "SKIP",
        "skip_trade": True,
        "skip_reason": reason,
        "market_regime": "unknown",
        "liquidity": "unknown",
        "session": session,
        "htf_bias": "n/a",
        "consolidating": False,
        "poc_price": None,
        "entry_type": "none",
        "entry_note": "No actionable signal.",
        "trade_side": "NONE",
        "order_type_hint": "NO_ORDER",
        "square_off_time": "15:15",
        "short_eligible": False,
        "execution_warning": "",
        "display_recommendation": "AVOID",
    }


# ──────────────────────────────────────────────────────────────────────────────
# Helper: pivot points
# ──────────────────────────────────────────────────────────────────────────────

def compute_pivot_points(high: float, low: float, close: float) -> dict:
    """Classic pivot points and full support/resistance set (S1–S3, R1–R3)."""
    pp = (high + low + close) / 3
    r1 = 2 * pp - low
    s1 = 2 * pp - high
    r2 = pp + (high - low)
    s2 = pp - (high - low)
    r3 = high + 2 * (pp - low)
    s3 = low  - 2 * (high - pp)
    return {
        "PP": round(pp, 2),
        "R1": round(r1, 2), "S1": round(s1, 2),
        "R2": round(r2, 2), "S2": round(s2, 2),
        "R3": round(r3, 2), "S3": round(s3, 2),
    }


# ──────────────────────────────────────────────────────────────────────────────
# Helper: candlestick patterns (Upgrade 8 – adds inside_bar, morning/evening star)
# ──────────────────────────────────────────────────────────────────────────────

def detect_candlestick_patterns(df: pd.DataFrame) -> dict:
    """
    Return a dict with last-candle pattern flags.

    Recognised patterns
    -------------------
    Original : doji, hammer, shooting_star, bull_engulf, bear_engulf
    New (v2) : inside_bar, morning_star, evening_star

    `consolidating` is True when inside_bar is detected.
    """
    base: dict = {
        "doji": False, "hammer": False, "shooting_star": False,
        "bull_engulf": False, "bear_engulf": False,
        "inside_bar": False, "morning_star": False, "evening_star": False,
        "consolidating": False,
    }

    if len(df) < 2:
        return base

    prev = df.iloc[-2]
    curr = df.iloc[-1]
    body         = abs(curr["Close"] - curr["Open"])
    total        = curr["High"] - curr["Low"]
    upper_shadow = curr["High"] - max(curr["Close"], curr["Open"])
    lower_shadow = min(curr["Close"], curr["Open"]) - curr["Low"]

    base["doji"]         = bool(total > 0 and body <= 0.05 * total)
    base["hammer"]       = bool(body > 0 and lower_shadow >= 2 * body and upper_shadow <= 0.1 * body)
    base["shooting_star"]= bool(body > 0 and upper_shadow >= 2 * body and lower_shadow <= 0.1 * body)

    prev_body      = abs(prev["Close"] - prev["Open"])
    prev_direction = 1 if prev["Close"] > prev["Open"] else -1
    curr_direction = 1 if curr["Close"] > curr["Open"] else -1

    base["bull_engulf"] = bool(
        prev_direction < 0 and curr_direction > 0
        and curr["Close"] > prev["Open"] and curr["Open"] < prev["Close"]
    )
    base["bear_engulf"] = bool(
        prev_direction > 0 and curr_direction < 0
        and curr["Open"] > prev["Close"] and curr["Close"] < prev["Open"]
    )

    # Inside bar: current candle completely inside previous candle's range
    inside = bool(curr["High"] < prev["High"] and curr["Low"] > prev["Low"])
    base["inside_bar"]    = inside
    base["consolidating"] = inside

    # 3-candle star patterns (need at least 3 bars)
    if len(df) >= 3:
        c0 = df.iloc[-3]   # first candle
        c1 = df.iloc[-2]   # middle (star body)
        c2 = df.iloc[-1]   # confirmation candle

        c0_bear = c0["Close"] < c0["Open"]
        c0_bull = c0["Close"] > c0["Open"]
        c1_small = abs(c1["Close"] - c1["Open"]) <= 0.3 * abs(c0["Close"] - c0["Open"])
        c2_bear = c2["Close"] < c2["Open"]
        c2_bull = c2["Close"] > c2["Open"]

        # Morning star: bearish → small middle → bullish
        base["morning_star"] = bool(
            c0_bear and c1_small
            and c1["High"] < min(c0["Open"], c0["Close"])   # gap down
            and c2_bull
            and c2["Close"] > (c0["Open"] + c0["Close"]) / 2
        )

        # Evening star: bullish → small middle → bearish
        base["evening_star"] = bool(
            c0_bull and c1_small
            and c1["Low"] > max(c0["Open"], c0["Close"])    # gap up
            and c2_bear
            and c2["Close"] < (c0["Open"] + c0["Close"]) / 2
        )

    return base


# ──────────────────────────────────────────────────────────────────────────────
# SwingAnalyzer  (daily timeframe) — original + v2 signal-quality filters
# ──────────────────────────────────────────────────────────────────────────────

class SwingAnalyzer:
    def __init__(
        self,
        df: pd.DataFrame,
        *,
        account_size: float | None = None,
        risk_percent: float = 1.0,
    ) -> None:
        self.account_size = account_size
        self.risk_percent = risk_percent
        self.df, self.data_quality = normalize_ohlcv(df, min_candles=60, max_stale_days=7)
        self.data_error = None if self.data_quality.ok else self.data_quality.reason
        if self.data_error:
            self.current = self.df.iloc[-1] if not self.df.empty else pd.Series(dtype=float)
            return

        self.current = self.df.iloc[-1]
        self.close   = self.df["Close"]
        self.volume  = self.df["Volume"]

        # ── original indicators ──────────────────────────────────────────────
        self.ema9   = EMAIndicator(self.close, window=9).ema_indicator()
        self.ema21  = EMAIndicator(self.close, window=21).ema_indicator()
        self.ema50  = EMAIndicator(self.close, window=50).ema_indicator()
        self.ema200 = EMAIndicator(self.close, window=200).ema_indicator()
        self.rsi    = RSIIndicator(self.close, window=14).rsi()
        self.macd   = MACD(self.close, window_slow=26, window_fast=12, window_sign=9)
        self.bb     = BollingerBands(self.close, window=20, window_dev=2)
        self.atr    = AverageTrueRange(
            self.df["High"], self.df["Low"], self.close, window=14
        ).average_true_range()
        self.vol_sma  = self.volume.rolling(window=20).mean()
        self.patterns = detect_candlestick_patterns(self.df)

        # ── v2: ADX ──────────────────────────────────────────────────────────
        self.adx = compute_adx(self.df["High"], self.df["Low"], self.close, window=14)

    # ── v2: market-regime / volatility helpers ───────────────────────────────

    def _market_regime(self) -> str:
        """
        Return 'trending' (ADX > 25), 'ranging' (ADX < 20), or 'neutral'.
        Used to suppress inappropriate signal types.
        """
        adx_val = self.adx.iloc[-1]
        atr_pct = (self.atr.iloc[-1] / self.close.iloc[-1]) * 100 if self.close.iloc[-1] else 0
        volume_avg = self.volume.rolling(20).mean().iloc[-1]
        low_liquidity = pd.notna(volume_avg) and self.volume.iloc[-1] < 0.5 * volume_avg
        if low_liquidity:
            return "low_liquidity"
        if atr_pct > 6:
            return "high_volatility"
        if adx_val > 25:
            return "trending"
        if adx_val < 20:
            return "ranging"
        return "neutral"

    def _liquidity_state(self) -> str:
        vol_avg = self.vol_sma.iloc[-1]
        vol_cur = self.current["Volume"]
        if vol_cur <= 0:
            return "missing"
        if pd.notna(vol_avg) and vol_cur < 0.5 * vol_avg:
            return "low"
        if pd.notna(vol_avg) and vol_cur >= 1.2 * vol_avg:
            return "strong"
        return "normal"

    def _atr_pct_filter(self) -> bool:
        """
        Return True if the current ATR% is above the 20th percentile of its
        50-bar rolling history (sufficient volatility to trade).
        Returns True (pass) when history is too short to compute.
        """
        atr_pct = self.atr / self.close
        if len(atr_pct.dropna()) < 50:
            return True
        p20 = atr_pct.rolling(50).quantile(0.20).iloc[-1]
        return atr_pct.iloc[-1] >= p20

    # ── original score methods (unchanged) ───────────────────────────────────

    def _trend_score(self) -> float:
        price = self.current["Close"]
        score = 0.0
        score += 2.0 if price > self.ema200.iloc[-1] else -2.0
        score += 1.5 if self.ema21.iloc[-1] > self.ema50.iloc[-1] else -1.5
        score += 1.0 if self.ema9.iloc[-1]  > self.ema21.iloc[-1] else -1.0
        return max(-4.0, min(4.0, score))

    def _momentum_score(self) -> float:
        rsi_val = self.rsi.iloc[-1]
        score   = 0.0
        if 40 < rsi_val < 70:
            score += 2.0 if rsi_val > self.rsi.iloc[-2] else -1.0
        elif rsi_val >= 70:
            score -= 2.0
        elif rsi_val <= 30:
            score += 2.0
        if len(self.close) >= 5:
            price_slice = self.close.iloc[-5:]
            rsi_slice   = self.rsi.iloc[-5:]
            if price_slice.min() == price_slice.iloc[-1] and rsi_slice.min() != rsi_slice.iloc[-1]:
                score += 1.5
            if price_slice.max() == price_slice.iloc[-1] and rsi_slice.max() != rsi_slice.iloc[-1]:
                score -= 1.5
        return max(-3.0, min(3.0, score))

    def _volume_confirmation(self) -> float:
        vol_avg = self.vol_sma.iloc[-1]
        vol_cur = self.current["Volume"]
        if pd.notna(vol_avg) and vol_cur >= 1.5 * vol_avg:
            return 2.0
        if pd.notna(vol_avg) and vol_cur >= 1.2 * vol_avg:
            return 1.0
        return 0.0

    def _pattern_score(self) -> float:
        p     = self.patterns
        score = 0.0
        if p["bull_engulf"]:
            score += 2.0
        if p["bear_engulf"]:
            score -= 2.0
        if p["morning_star"]:
            score += 2.0
        if p["evening_star"]:
            score -= 2.0
        if p["hammer"]:
            support = self.df["Low"].iloc[-10:].min()
            if self.current["Low"] <= support * 1.02:
                score += 1.5
        if p["shooting_star"]:
            resistance = self.df["High"].iloc[-10:].max()
            if self.current["High"] >= resistance * 0.98:
                score -= 1.5
        return score

    # ── main analyze ─────────────────────────────────────────────────────────

    def analyze(self) -> dict:
        if self.data_error:
            price = self.current.get("Close", 0.0) if not self.df.empty else 0.0
            return _empty_result(self.data_error, session="swing", price=price)

        trend    = self._trend_score()
        momentum = self._momentum_score()
        volume   = self._volume_confirmation()
        pattern  = self._pattern_score()

        total_score = trend * 0.4 + momentum * 0.25 + volume * 0.2 + pattern * 0.15

        # v2: regime filter
        regime = self._market_regime()
        liquidity = self._liquidity_state()
        if regime == "ranging" and abs(total_score) > 0.5:
            # suppress breakout signals in ranging market
            total_score *= 0.7
        if regime in ("high_volatility", "low_liquidity"):
            total_score *= 0.5

        # v2: sigmoid confidence (Upgrade 6)
        confidence = sigmoid_confidence(total_score)

        # v2: volatility filter
        atr_val  = self.atr.iloc[-1]
        price    = round(self.current["Close"], 2)
        atr_pct  = round((atr_val / price) * 100, 3) if price > 0 else 0.0
        low_vol  = not self._atr_pct_filter()
        if low_vol:
            confidence = min(confidence, 40)

        if total_score > 0.5:
            recommendation = "BUY"
        elif total_score < -0.5:
            recommendation = "SELL"
        else:
            recommendation = "AVOID"

        bb_upper = round(self.bb.bollinger_hband().iloc[-1], 2)
        bb_lower = round(self.bb.bollinger_lband().iloc[-1], 2)

        support1    = bb_lower
        support2    = round(bb_lower - atr_val, 2)
        resistance1 = bb_upper
        resistance2 = round(bb_upper + atr_val, 2)

        # ── v3: strategic entry ───────────────────────────────────────────────
        # Build the levels dict that _strategic_entry() needs.
        # For SwingAnalyzer the "structural level" for breakout is the BB band
        # that price is approaching; support/resistance come from BB bands too.
        levels_dict = {
            "breakout_long":  resistance1,   # bull breakout above upper BB
            "breakout_short": support1,       # bear breakdown below lower BB
            "support":        support1,
            "resistance":     resistance1,
            "candle_high":    round(self.current["High"], 2),
            "candle_low":     round(self.current["Low"],  2),
        }
        ema21_val = round(self.ema21.iloc[-1], 2)

        # ── Stop loss: structure-based with sanity guard ──────────────────────
        if recommendation == "BUY":
            stop_loss, _ = _structure_stop(self.df, "BUY", atr_val)
            stop_loss = round(stop_loss, 2)
            if stop_loss >= price:                       # must be below price
                stop_loss = round(price - 1.5 * atr_val, 2)
        elif recommendation == "SELL":
            stop_loss, _ = _structure_stop(self.df, "SELL", atr_val)
            stop_loss = round(stop_loss, 2)
            if stop_loss <= price:                       # must be above price
                stop_loss = round(price + 1.5 * atr_val, 2)
        else:
            stop_loss = round(price - atr_val, 2)

        # ── v3: strategic entry ───────────────────────────────────────────────
        if recommendation in ("BUY", "SELL"):
            entry, entry_type, entry_note = _strategic_entry(
                direction=recommendation,
                price=price,
                atr_val=atr_val,
                patterns=self.patterns,
                levels=levels_dict,
                vwap=None,   # SwingAnalyzer has no intraday VWAP
                poc=None,    # POC is intraday only
                ema21=ema21_val,
            )
        else:
            entry      = price
            entry_type = "pullback"
            entry_note = "No actionable signal – monitoring only."

        # ── Targets: always anchored to current PRICE, not entry ─────────────
        # This is the key fix: targets represent where the market CAN go.
        # Entry represents where the trader WAITS to get in.
        # If we based targets on entry, a pullback entry would shrink reward
        # to near-zero when entry < price on a BUY (entry is below market).
        # We then do a final guard: if entry already cleared the first target
        # (can happen on aggressive breakout entries), we project from entry.
        if recommendation == "BUY":
            raw_t1 = bb_upper if bb_upper > price else round(price + atr_val, 2)
            target1 = raw_t1
            target2 = round(target1 + atr_val, 2)
            if target1 <= entry:                         # entry leapfrogged T1
                target1 = round(entry + atr_val, 2)
                target2 = round(entry + 2 * atr_val, 2)
                entry_note += " (Targets rebased above entry.)"
        elif recommendation == "SELL":
            raw_t1 = bb_lower if bb_lower < price else round(price - atr_val, 2)
            target1 = raw_t1
            target2 = round(target1 - atr_val, 2)
            if target1 >= entry:                         # entry fell below T1
                target1 = round(entry - atr_val, 2)
                target2 = round(entry - 2 * atr_val, 2)
                entry_note += " (Targets rebased below entry.)"
        else:
            target1 = round(price + atr_val, 2)
            target2 = round(price + 2 * atr_val, 2)

        # ── R:R anchored to strategic entry ───────────────────────────────────
        risk   = abs(entry - stop_loss)
        reward = abs(target1 - entry)
        rr     = round(reward / risk, 2) if risk > 0 else 0.0

        why_trade = []
        why_avoid = []
        reasons = []
        reasons.append("Bullish trend" if trend > 0 else "Bearish trend")
        reasons.append("Positive RSI"  if momentum > 0 else "Weak momentum")
        if recommendation == "BUY" and trend > 0:
            why_trade.append("Price structure is aligned above key moving averages.")
        if recommendation == "SELL" and trend < 0:
            why_trade.append("Price structure is aligned below key moving averages.")
        if momentum > 0 and recommendation == "BUY":
            why_trade.append("Momentum supports a long setup.")
        if momentum < 0 and recommendation == "SELL":
            why_trade.append("Momentum supports a short setup.")
        if volume >= 1.5:
            reasons.append("High volume")
            why_trade.append("Volume is confirming participation.")
        if pattern > 0:
            reasons.append("Bullish pattern")
        elif pattern < 0:
            reasons.append("Bearish pattern")
        if low_vol:
            reasons.append("Low volatility – signal suppressed")
            why_avoid.append("Volatility is too low for a clean swing setup.")
        if regime == "ranging":
            why_avoid.append("Market is ranging, so breakout follow-through is less reliable.")
        if regime == "high_volatility":
            why_avoid.append("ATR is unusually high; stops may be too wide.")
        if liquidity in ("low", "missing"):
            why_avoid.append("Liquidity/volume is weak or unavailable.")
        reason = f"Score: {total_score:.2f}. " + ", ".join(reasons) + "."

        # v2: trade quality — now returns skip_reason too
        trade_quality, skip_trade, skip_reason = _grade_trade(
            rr,
            confidence,
            atr_pct,
            sl_percent=round((risk / entry) * 100, 2) if entry > 0 else 0.0,
            liquidity=liquidity,
            regime=regime,
        )
        if skip_reason:
            reason += f" Skip: {skip_reason}."
            why_avoid.extend(skip_reason.split("; "))
        if skip_trade:
            recommendation = "AVOID"
            confidence = min(confidence, 45)
            trade_quality = "SKIP"

        position_size = _position_size(entry, stop_loss, self.account_size, self.risk_percent)
        invalidation = _invalidation_reason(recommendation, entry, stop_loss, regime)

        return {
            # ── original keys ─────────────────────────────────────────────
            "recommendation": recommendation,
            "confidence":     confidence,
            "entry":          round(entry, 2),
            "stop_loss":      stop_loss,
            "sl_percent":     round((risk / entry) * 100, 2) if entry > 0 else 0.0,
            "target1":        target1,
            "target2":        target2,
            "rr":             rr,
            "support1":       support1,
            "support2":       support2,
            "resistance1":    resistance1,
            "resistance2":    resistance2,
            "reason":         reason,
            "why_trade":      why_trade,
            "why_avoid":      why_avoid or (["No clean confluence across trend, momentum, volume, and risk."] if recommendation == "AVOID" else []),
            "invalidation":   invalidation,
            "position_size":  position_size,
            # ── v2 new keys ───────────────────────────────────────────────
            "trade_quality":  trade_quality,
            "skip_trade":     skip_trade,
            "skip_reason":    skip_reason,
            "market_regime":  regime,
            "liquidity":      liquidity,
            "session":        "swing",
            "htf_bias":       "n/a",
            "consolidating":  self.patterns["consolidating"],
            "poc_price":      None,
            # ── v3 new keys ───────────────────────────────────────────────
            "entry_type":     entry_type,
            "entry_note":     entry_note,
            "trade_side":     "LONG" if recommendation == "BUY" else "SHORT" if recommendation == "SELL" else "NONE",
            "order_type_hint": "BUY_STOP_LIMIT" if recommendation == "BUY" else "SELL_STOP_LIMIT" if recommendation == "SELL" else "NO_ORDER",
            "square_off_time": None,
            "short_eligible": False,
            "execution_warning": "",
            "display_recommendation": recommendation,
        }


# ──────────────────────────────────────────────────────────────────────────────
# Shared helpers used by IntradayAnalyzer
# ──────────────────────────────────────────────────────────────────────────────

def _grade_trade(
    rr: float,
    confidence: int,
    atr_pct: float,
    *,
    sl_percent: float = 0.0,
    liquidity: str = "normal",
    regime: str = "neutral",
    htf_bias: str = "n/a",
    direction: str = "AVOID",
) -> tuple[str, bool, str]:
    """
    Assign a trade quality grade, skip flag, and human-readable skip reason.

    Grade rules
    -----------
    A : rr >= 2.5 AND confidence >= 65
    B : rr >= 1.7 AND confidence >= 55
    C : everything else

    skip_trade is True when grade is C, ATR% is poor, liquidity is weak,
    stop width is excessive, volatility is chaotic, or HTF bias conflicts.
    skip_reason is a non-empty string whenever skip_trade is True, so
    the UI can explain the contradiction (e.g. high confidence + C grade).
    """
    if rr >= 2.5 and confidence >= 65:
        grade = "A"
    elif rr >= 1.7 and confidence >= 55:
        grade = "B"
    else:
        grade = "C"

    skip_reasons: list[str] = []
    if grade == "C":
        if rr < 1.7:
            skip_reasons.append(f"R:R {rr:.2f} too low (min 1.7)")
        if confidence < 55:
            skip_reasons.append(f"confidence {confidence}% below threshold")
    if atr_pct < 0.3:
        skip_reasons.append(f"ATR% {atr_pct:.2f}% < 0.3% (dead market)")
    if atr_pct > 7:
        skip_reasons.append(f"ATR% {atr_pct:.2f}% too high for disciplined sizing")
    if sl_percent > 4.0:
        skip_reasons.append(f"stop width {sl_percent:.2f}% too wide")
    if liquidity in ("low", "missing"):
        skip_reasons.append(f"liquidity is {liquidity}")
    if regime == "high_volatility":
        skip_reasons.append("market regime is high volatility")
    if regime == "low_liquidity":
        skip_reasons.append("market regime is low liquidity")
    if htf_bias == "conflict" and direction in ("BUY", "SELL", "SHORT_SELL"):
        skip_reasons.append("higher-timeframe bias conflicts")

    skip        = bool(skip_reasons)
    skip_reason = "; ".join(skip_reasons)
    return grade, skip, skip_reason


def _position_size(
    entry: float,
    stop_loss: float,
    account_size: float | None,
    risk_percent: float,
) -> dict:
    risk_per_share = round(abs(entry - stop_loss), 2)
    if not account_size or account_size <= 0 or risk_percent <= 0 or risk_per_share <= 0:
        return {
            "shares": 0,
            "capital_at_risk": 0.0,
            "risk_per_share": risk_per_share,
            "note": "Provide account_size and risk_percent for position sizing.",
        }
    capital_at_risk = round(account_size * (risk_percent / 100), 2)
    shares = int(capital_at_risk // risk_per_share)
    return {
        "shares": max(shares, 0),
        "capital_at_risk": capital_at_risk,
        "risk_per_share": risk_per_share,
        "note": f"Risking {risk_percent:.2f}% of account.",
    }


def _invalidation_reason(direction: str, entry: float, stop_loss: float, regime: str) -> str:
    if direction == "BUY":
        return f"Long thesis invalidates below stop-loss {stop_loss}; reassess if regime turns {regime} with weak volume."
    if direction in ("SELL", "SHORT_SELL"):
        return f"Short thesis invalidates above stop-loss {stop_loss}; reassess if regime turns {regime} with bullish volume."
    return "No trade is valid until trend, momentum, volume, structure, and risk/reward align."


def _structure_stop(df: pd.DataFrame, direction: str, atr_val: float) -> tuple[float, bool]:
    """
    Compute a structure-based stop using the last 3 candle lows/highs.

    For BUY  : stop = min(last-3 lows)  – 0.1×ATR
    For SELL : stop = max(last-3 highs) + 0.1×ATR

    If the resulting risk exceeds 2×ATR the stop is too wide; returns
    (atr_stop, used_fallback=True) instead.
    price is df.iloc[-1]['Close'].
    """
    price = df.iloc[-1]["Close"]
    if direction == "BUY":
        structure_level = df["Low"].iloc[-3:].min()
        proposed_stop   = structure_level - 0.1 * atr_val
        fallback_stop   = price - 1.2 * atr_val
        if abs(price - proposed_stop) > 2 * atr_val:
            return fallback_stop, True
        return proposed_stop, False
    else:  # SELL
        structure_level = df["High"].iloc[-3:].max()
        proposed_stop   = structure_level + 0.1 * atr_val
        fallback_stop   = price + 1.2 * atr_val
        if abs(proposed_stop - price) > 2 * atr_val:
            return fallback_stop, True
        return proposed_stop, False


def _strategic_entry(
    direction: str,
    price: float,
    atr_val: float,
    patterns: dict,
    levels: dict,
    vwap: Optional[float] = None,
    poc: Optional[float] = None,
    ema21: Optional[float] = None,
) -> tuple[float, str, str]:
    """
    Compute a strategic limit-order entry price instead of the raw last close.

    Parameters
    ----------
    direction : 'BUY' or 'SELL'
    price     : last close (used as reference / fallback)
    atr_val   : current ATR(14) value
    patterns  : output of detect_candlestick_patterns()
    levels    : dict with keys 'breakout_long', 'breakout_short',
                'support', 'resistance' (all floats)
    vwap      : current VWAP (optional)
    poc       : intraday Point of Control (optional)
    ema21     : current EMA-21 value (optional)

    Returns
    -------
    (entry_price, entry_type, entry_note)

    entry_type is one of: 'rejection', 'breakout', 'pullback'

    Priority
    --------
    1. rejection  – reversal pattern fired at a S/R level  → confirmation trigger
    2. breakout   – price within 0.5×ATR of a structural level → level + buffer
    3. pullback   – trending but not at key level → nearest confluence zone
    """
    direction = "SELL" if direction == "SHORT_SELL" else direction
    buf  = round(0.05 * atr_val, 2)   # confirmation buffer (tight)
    half = round(0.15 * atr_val, 2)   # "already at zone" tolerance

    # ── 1. REJECTION entry ────────────────────────────────────────────────────
    #  Qualifying patterns: hammer / bull_engulf / morning_star (BUY)
    #                       shooting_star / bear_engulf / evening_star (SELL)
    if direction == "BUY":
        reversal_pattern = (
            patterns.get("hammer")
            or patterns.get("bull_engulf")
            or patterns.get("morning_star")
        )
        support_level = levels.get("support")
        if reversal_pattern and support_level is not None:
            # Only treat as rejection if candle is near the support level
            near_support = abs(price - support_level) <= 0.4 * atr_val
            if near_support:
                # Entry = high of the signal candle + tiny buffer
                # (levels dict carries current candle high via 'candle_high')
                candle_high = levels.get("candle_high", price)
                entry = round(candle_high + buf, 2)
                note  = (
                    f"Rejection BUY: enter above signal-candle high ({candle_high}) "
                    f"+ {buf} buffer at support {support_level}. "
                    f"Limit at {entry}."
                )
                return entry, "rejection", note

    elif direction == "SELL":
        reversal_pattern = (
            patterns.get("shooting_star")
            or patterns.get("bear_engulf")
            or patterns.get("evening_star")
        )
        resistance_level = levels.get("resistance")
        if reversal_pattern and resistance_level is not None:
            near_resistance = abs(price - resistance_level) <= 0.4 * atr_val
            if near_resistance:
                candle_low = levels.get("candle_low", price)
                entry = round(candle_low - buf, 2)
                note  = (
                    f"Rejection SELL: enter below signal-candle low ({candle_low}) "
                    f"- {buf} buffer at resistance {resistance_level}. "
                    f"Limit at {entry}."
                )
                return entry, "rejection", note

    # ── 2. BREAKOUT entry ─────────────────────────────────────────────────────
    #  Price has just broken (or is hugging) a structural level.
    #  We enter just BEYOND the level, not wherever price currently is,
    #  so the order only fills if the breakout continues.
    if direction == "BUY":
        brk = levels.get("breakout_long")
        if brk is not None and price <= brk + 0.5 * atr_val:
            entry = round(brk + buf, 2)
            note  = (
                f"Breakout BUY: limit just above structural level {brk} "
                f"(+{buf} buffer). Current price {price}."
            )
            return entry, "breakout", note

    elif direction == "SELL":
        brk = levels.get("breakout_short")
        if brk is not None and price >= brk - 0.5 * atr_val:
            entry = round(brk - buf, 2)
            note  = (
                f"Breakout SELL: limit just below structural level {brk} "
                f"(-{buf} buffer). Current price {price}."
            )
            return entry, "breakout", note

    # ── 3. PULLBACK entry (default) ───────────────────────────────────────────
    #  Find the nearest confluence zone: VWAP → POC → EMA21.
    #  If price is already within half-ATR of that zone, use close.
    #  Otherwise set a limit at the zone so we don't pay the extended price.
    candidates: list[tuple[float, str]] = []
    if vwap  is not None: candidates.append((vwap,  "VWAP"))
    if poc   is not None: candidates.append((poc,   "POC"))
    if ema21 is not None: candidates.append((ema21, "EMA21"))

    if candidates:
        # Pick zone closest to current price
        zone_price, zone_name = min(candidates, key=lambda c: abs(c[0] - price))
        at_zone = abs(price - zone_price) <= half

        if direction == "BUY":
            if at_zone:
                entry = price
                note  = (
                    f"Pullback BUY: price already at {zone_name} ({zone_price}). "
                    f"Enter at market ({price})."
                )
            else:
                # Place limit at the zone; price must pull back to it
                entry = round(zone_price + buf, 2)   # slightly above zone for fill certainty
                note  = (
                    f"Pullback BUY: wait for retracement to {zone_name} ({zone_price}). "
                    f"Limit at {entry}."
                )
        else:  # SELL
            if at_zone:
                entry = price
                note  = (
                    f"Pullback SELL: price already at {zone_name} ({zone_price}). "
                    f"Enter at market ({price})."
                )
            else:
                entry = round(zone_price - buf, 2)
                note  = (
                    f"Pullback SELL: wait for rally back to {zone_name} ({zone_price}). "
                    f"Limit at {entry}."
                )
        return entry, "pullback", note

    # Absolute fallback: no zones available → use last close (safe, explicit)
    return price, "pullback", f"Pullback entry at last close {price} (no confluence zones available)."


def _compute_poc(df: pd.DataFrame, n_bins: int = 20, now: Optional[pd.Timestamp] = None) -> Optional[float]:
    """
    Approximate intraday Point of Control via a 20-bin volume profile.

    Buckets the session's price range, sums volume per bin, and returns
    the midpoint of the highest-volume bin.  Returns None if insufficient data.
    """
    today = (now or pd.Timestamp.now(tz="Asia/Kolkata")).date()
    session = df[df.index.tz_convert("Asia/Kolkata").date == today] if getattr(df.index, "tz", None) else df
    if len(session) < 5:
        session = df  # fallback: use all available data

    lo, hi = session["Low"].min(), session["High"].max()
    if hi <= lo:
        return None

    bins = np.linspace(lo, hi, n_bins + 1)
    mid  = (bins[:-1] + bins[1:]) / 2
    vol  = np.zeros(n_bins)

    for _, row in session.iterrows():
        # distribute bar's volume evenly across bins that overlap its range
        overlap = (bins[1:] >= row["Low"]) & (bins[:-1] <= row["High"])
        count   = overlap.sum()
        if count:
            vol[overlap] += row["Volume"] / count

    return round(float(mid[np.argmax(vol)]), 2)


def _ist_session(now: Optional[pd.Timestamp] = None) -> tuple[str, float]:
    """
    Return (session_name, time_multiplier) for the current IST time.

    Zones
    -----
    opening : 09:15 – 10:00  → mult 1.2  (high opportunity, widen stops 20 %)
    midday  : 10:00 – 13:30  → mult 1.0
    closing : 13:30 – 15:15  → mult 0.7  (fade signals, prefer mean-reversion)
    """
    if now is None:
        now = pd.Timestamp.now(tz="Asia/Kolkata")
    t = now.time()
    import datetime
    if datetime.time(9, 15) <= t < datetime.time(10, 0):
        return "opening", 1.2
    if datetime.time(10, 0) <= t < datetime.time(13, 30):
        return "midday", 1.0
    return "closing", 0.7


def _news_blackout(confidence: int, now: Optional[pd.Timestamp] = None) -> int:
    """
    Cap confidence at 40 within 15 minutes of market open (09:15 IST)
    or market close (15:15 IST) to avoid news-driven false signals.
    """
    import datetime
    if now is None:
        now = pd.Timestamp.now(tz="Asia/Kolkata")
    t = now.time()
    blackout_windows = [
        (datetime.time(9,  0), datetime.time(9, 30)),
        (datetime.time(15, 0), datetime.time(15, 30)),
    ]
    for start, end in blackout_windows:
        if start <= t <= end:
            return min(confidence, 40)
    return confidence


# ──────────────────────────────────────────────────────────────────────────────
# IntradayAnalyzer  (15-minute) — all upgrades applied
# ──────────────────────────────────────────────────────────────────────────────

class IntradayAnalyzer:
    def __init__(
        self,
        df:     pd.DataFrame,
        info:   dict,
        htf_df: Optional[pd.DataFrame] = None,   # v2: higher-timeframe (e.g. 1H)
        account_size: float | None = None,
        risk_percent: float = 0.5,
        now: Optional[pd.Timestamp] = None,
        short_eligibility_service: ShortSellEligibilityService | None = None,
        late_short_cutoff: datetime.time = datetime.time(15, 0),
        square_off_time: str = "15:15",
    ) -> None:
        self.account_size = account_size
        self.risk_percent = risk_percent
        self.now = now
        self.late_short_cutoff = late_short_cutoff
        self.square_off_time = square_off_time
        self.df, self.data_quality = normalize_ohlcv(df, min_candles=30)
        self.data_error = None if self.data_quality.ok else self.data_quality.reason
        self.info = info
        self.symbol = self._resolve_symbol(info)
        extra_shortable = info.get("shortable_symbols") or info.get("intraday_shortable_symbols")
        self.short_eligibility_service = short_eligibility_service or StaticShortSellEligibility.from_settings(extra_shortable)
        self.is_pm = False
        if self.data_error:
            self.current = self.df.iloc[-1] if not self.df.empty else pd.Series(dtype=float)
            return

        if not isinstance(self.df.index, pd.DatetimeIndex):
            self.df.index = pd.to_datetime(self.df.index)

        self.current = self.df.iloc[-1]
        self.close   = self.df["Close"]
        self.volume  = self.df["Volume"]

        # ── original indicators ──────────────────────────────────────────────
        self.ema9    = EMAIndicator(self.close, window=9).ema_indicator()
        self.ema21   = EMAIndicator(self.close, window=21).ema_indicator()
        self.rsi     = RSIIndicator(self.close, window=14).rsi()
        self.atr     = AverageTrueRange(
            self.df["High"], self.df["Low"], self.close, window=14
        ).average_true_range()
        self.df      = compute_vwap(self.df)
        self.vol_sma = self.volume.rolling(window=20).mean()

        # ── v2: ADX ──────────────────────────────────────────────────────────
        self.adx = compute_adx(self.df["High"], self.df["Low"], self.close, window=14)

        # ── v2: HTF EMA ──────────────────────────────────────────────────────
        self._htf_df: Optional[pd.DataFrame] = None
        self._htf_ema9: Optional[pd.Series]  = None
        self._htf_ema21: Optional[pd.Series] = None
        if htf_df is not None:
            self._htf_df   = htf_df.copy()
            htf_close      = self._htf_df["Close"]
            self._htf_ema9  = EMAIndicator(htf_close, window=9).ema_indicator()
            self._htf_ema21 = EMAIndicator(htf_close, window=21).ema_indicator()

        self.patterns = detect_candlestick_patterns(self.df)

    def _resolve_symbol(self, info: dict) -> str:
        for key in ("symbol", "underlyingSymbol", "quoteTypeSymbol", "shortName"):
            value = info.get(key)
            if value:
                return str(value)
        return ""

    # ── original helpers (unchanged) ─────────────────────────────────────────

    def _pivot_points(self) -> dict:
        prev_high  = self.info.get("regularMarketDayHigh",  self.current["High"])
        prev_low   = self.info.get("regularMarketDayLow",   self.current["Low"])
        prev_close = self.info.get("previousClose",         self.current["Close"])
        return compute_pivot_points(prev_high, prev_low, prev_close)

    def _opening_range(self) -> tuple[float, float]:
        today      = (self.now or pd.Timestamp.now(tz="Asia/Kolkata")).date()
        today_data = self.df[self.df.index.date == today]
        first      = today_data.iloc[0] if len(today_data) >= 1 else self.df.iloc[0]
        return first["High"], first["Low"]

    # ── v2 helpers ───────────────────────────────────────────────────────────

    def _htf_confluence(self, lf_bull: bool) -> tuple[float, str]:
        """
        Compare 15-min trend direction with HTF EMA9/EMA21 trend.

        Returns (score_multiplier, bias_label).
        If no HTF data supplied, returns (1.0, 'n/a').
        """
        if self._htf_ema9 is None or self._htf_ema21 is None:
            return 1.0, "n/a"
        htf_bull = self._htf_ema9.iloc[-1] > self._htf_ema21.iloc[-1]
        if htf_bull == lf_bull:
            return 1.2, "bullish" if htf_bull else "bearish"
        return 0.6, "conflict"

    def _market_regime(self) -> str:
        """Return 'trending', 'ranging', or 'neutral' based on ADX(14)."""
        adx_val = self.adx.iloc[-1]
        atr_pct = (self.atr.iloc[-1] / self.close.iloc[-1]) * 100 if self.close.iloc[-1] else 0
        liquidity = self._liquidity_state()
        if liquidity in ("low", "missing"):
            return "low_liquidity"
        if atr_pct > 2.8:
            return "high_volatility"
        if adx_val > 25:
            return "trending"
        if adx_val < 20:
            return "ranging"
        return "neutral"

    def _liquidity_state(self) -> str:
        vol_avg = self.vol_sma.iloc[-1]
        vol_cur = self.current["Volume"]
        if vol_cur <= 0:
            return "missing"
        if pd.notna(vol_avg) and vol_cur < 0.5 * vol_avg:
            return "low"
        if pd.notna(vol_avg) and vol_cur >= 1.2 * vol_avg:
            return "strong"
        return "normal"

    def _atr_pct_filter(self) -> bool:
        """
        Return True when current ATR% is above the 20th percentile of its
        50-bar rolling history (adequate volatility to trade).
        """
        atr_pct = self.atr / self.close
        if len(atr_pct.dropna()) < 50:
            return True
        p20 = atr_pct.rolling(50).quantile(0.20).iloc[-1]
        return float(atr_pct.iloc[-1]) >= float(p20)

    def _late_for_new_short(self, now: pd.Timestamp) -> bool:
        return now.time() >= self.late_short_cutoff

    def _has_stale_intraday_data(self, now: pd.Timestamp) -> bool:
        if not isinstance(self.df.index, pd.DatetimeIndex) or self.df.empty:
            return True
        last_ts = pd.Timestamp(self.df.index[-1])
        if last_ts.tzinfo is None and now.tzinfo is not None:
            last_ts = last_ts.tz_localize(now.tzinfo)
        elif last_ts.tzinfo is not None and now.tzinfo is not None:
            last_ts = last_ts.tz_convert(now.tzinfo)
        elif last_ts.tzinfo is not None and now.tzinfo is None:
            last_ts = last_ts.tz_localize(None)
        return (now.normalize() - last_ts.normalize()).days > 1

    def _short_confluence(
        self,
        *,
        price: float,
        vwap: float,
        trend_bull: bool,
        rsi_val: float,
        volume_spike: bool,
        support1: float,
        resistance1: float,
        or_low: float,
        pivot: dict,
        atr_val: float,
        regime: str,
        liquidity: str,
        low_vol: bool,
        htf_bias: str,
        now: pd.Timestamp,
    ) -> tuple[bool, str, str]:
        checks: list[str] = []
        failures: list[str] = []

        if price < vwap:
            checks.append("price below VWAP")
        else:
            failures.append("price is not below VWAP")

        if not trend_bull:
            checks.append("EMA9 below EMA21")
        else:
            failures.append("EMA9 is not below EMA21")

        if rsi_val < 45:
            checks.append("RSI weakness")
        else:
            failures.append("RSI is not weak enough")

        breakdown = price < support1 or price < or_low or price < pivot["S1"]
        near_resistance = abs(price - resistance1) <= 0.35 * atr_val or abs(price - vwap) <= 0.25 * atr_val
        bearish_rejection = bool(
            self.patterns.get("shooting_star")
            or self.patterns.get("bear_engulf")
            or self.patterns.get("evening_star")
        )
        rally_to_sell = near_resistance and bearish_rejection and price <= max(resistance1, vwap)

        if breakdown and volume_spike:
            checks.append("breakdown below S1/opening-range low with volume")
        elif breakdown:
            failures.append("breakdown lacks volume confirmation")
        elif rally_to_sell:
            checks.append("bearish rejection near VWAP/EMA/resistance")
        else:
            failures.append("no breakdown or resistance rejection")

        if liquidity in ("low", "missing") or regime == "low_liquidity":
            failures.append("liquidity is too weak for intraday short execution")
        if regime == "high_volatility":
            failures.append("volatility is too extreme for a disciplined short")
        if low_vol:
            failures.append("intraday volatility is too low")
        if htf_bias == "conflict":
            failures.append("higher-timeframe trend is bullish")
        if self._has_stale_intraday_data(now):
            failures.append("intraday data is stale")
        if self._late_for_new_short(now):
            failures.append(f"new short entries are disabled after {self.late_short_cutoff.strftime('%H:%M')} IST")

        ok = len(checks) >= 4 and not failures
        setup = "breakdown" if breakdown else "rally_to_sell" if rally_to_sell else "none"
        reason = "; ".join(checks if ok else failures)
        return ok, setup, reason

    # ── main analyze ─────────────────────────────────────────────────────────

    def analyze(self) -> dict:
        if self.data_error:
            price = self.current.get("Close", 0.0) if not self.df.empty else 0.0
            return _empty_result(self.data_error, session="intraday", price=price)

        pivot         = self._pivot_points()
        or_high, or_low = self._opening_range()
        price         = round(self.current["Close"], 2)
        atr_val       = self.atr.iloc[-1]
        atr_pct       = round((atr_val / price) * 100, 3) if price > 0 else 0.0

        # ── v2: session (Upgrade 5) ───────────────────────────────────────
        now                  = self.now or pd.Timestamp.now(tz="Asia/Kolkata")
        session_name, time_mult = _ist_session(now)
        self.is_pm           = session_name == "closing"   # backward compat

        trend_bull       = self.ema9.iloc[-1] > self.ema21.iloc[-1]
        vwap             = self.df["VWAP"].iloc[-1]
        price_above_vwap = price > vwap
        rsi_val          = self.rsi.iloc[-1]
        vol_cur          = self.current["Volume"]
        vol_avg          = self.vol_sma.iloc[-1]
        volume_spike     = pd.notna(vol_avg) and vol_cur >= 1.2 * vol_avg
        liquidity        = self._liquidity_state()

        # ── original score logic ─────────────────────────────────────────
        score   = 0.0
        reasons = []

        if price > pivot["R1"]:
            score += 2; reasons.append("Above R1")
        elif price < pivot["S1"]:
            score -= 2; reasons.append("Below S1")

        if price > or_high and volume_spike:
            score += 2; reasons.append("ORB up")
        elif price < or_low and volume_spike:
            score -= 2; reasons.append("ORB down")

        if trend_bull:
            score += 1
        else:
            score -= 1

        if price_above_vwap:
            score += 1; reasons.append("Above VWAP")
        else:
            score -= 1; reasons.append("Below VWAP")

        if rsi_val > 60:
            score += 1
        elif rsi_val < 40:
            score -= 1

        # ── v2: POC score contribution (Upgrade 4) ────────────────────────
        poc_price = _compute_poc(self.df, now=now)
        if poc_price is not None and abs(price - poc_price) <= 0.3 * atr_val:
            score += 1.0
            reasons.append("Near POC")

        # ── v2: market-regime filter (Upgrade 1) ──────────────────────────
        regime = self._market_regime()
        if regime == "trending" and abs(score) < 1.5:
            # suppress mean-reversion signals in a trending market
            score *= 0.7
            reasons.append("Regime: trending – weak signal suppressed")
        elif regime == "ranging" and abs(score) >= 1.5:
            # suppress breakout signals in a ranging market
            score *= 0.7
            reasons.append("Regime: ranging – breakout suppressed")
        elif regime in ("high_volatility", "low_liquidity"):
            score *= 0.5
            reasons.append(f"Regime: {regime.replace('_', ' ')} – signal suppressed")

        # ── v2: HTF confluence (Upgrade 2) ────────────────────────────────
        htf_mult, htf_bias = self._htf_confluence(trend_bull)
        score *= htf_mult
        if htf_bias == "conflict":
            reasons.append("HTF conflict")
        elif htf_bias != "n/a":
            reasons.append("HTF confirmed")

        # ── v2: session time multiplier (Upgrade 5) ───────────────────────
        score *= time_mult

        # ── recommendation ───────────────────────────────────────────────
        if score > 1.5:
            recommendation = "BUY"
        elif score < -1.5:
            recommendation = "SHORT_SELL"
        else:
            recommendation = "AVOID"

        # ── v2: sigmoid confidence (Upgrade 6) ───────────────────────────
        confidence = sigmoid_confidence(abs(score) if recommendation == "SHORT_SELL" else score)

        # ── v2: volatility filter (Upgrade 1) ────────────────────────────
        low_vol = not self._atr_pct_filter()
        if low_vol:
            confidence = min(confidence, 40)
            reasons.append("Low volatility – signal suppressed")

        # ── v2: news blackout (Upgrade 1) ────────────────────────────────
        confidence = _news_blackout(confidence, now)

        support1    = round(min(or_low,  pivot["S1"]), 2)
        support2    = round(pivot["S2"], 2)
        resistance1 = round(max(or_high, pivot["R1"]), 2)
        resistance2 = round(pivot["R2"], 2)

        short_eligible = self.short_eligibility_service.is_shortable(self.symbol)
        short_setup = "none"
        short_reason = ""
        if recommendation == "SHORT_SELL":
            short_ok, short_setup, short_reason = self._short_confluence(
                price=price,
                vwap=vwap,
                trend_bull=trend_bull,
                rsi_val=rsi_val,
                volume_spike=volume_spike,
                support1=support1,
                resistance1=resistance1,
                or_low=or_low,
                pivot=pivot,
                atr_val=atr_val,
                regime=regime,
                liquidity=liquidity,
                low_vol=low_vol,
                htf_bias=htf_bias,
                now=now,
            )
            if not short_eligible:
                short_ok = False
                short_reason = self.short_eligibility_service.reason(self.symbol)
            if not short_ok:
                reasons.append(f"Short skipped: {short_reason}")
                recommendation = "AVOID"
                confidence = min(confidence, 45)

        # ── v2: structure-based stop loss with sanity guard ──────────────────
        if recommendation == "BUY":
            stop_loss, _ = _structure_stop(self.df, "BUY", atr_val)
            stop_loss = round(stop_loss, 2)
            if session_name == "opening":
                stop_loss = round(stop_loss - 0.2 * abs(price - stop_loss), 2)
            if stop_loss >= price:                       # must be below price
                stop_loss = round(price - 1.2 * atr_val, 2)
            # Targets anchored to PRICE (where market can go), not entry
            raw_t1 = pivot["R2"] if price > pivot["R1"] else pivot["R1"]
            target1 = round(raw_t1, 2)
            target2 = round(pivot["R3"], 2)
            if target1 <= price:                         # price already past T1
                target1 = round(price + atr_val, 2)
                target2 = round(price + 2 * atr_val, 2)

        elif recommendation == "SHORT_SELL":
            stop_loss, _ = _structure_stop(self.df, "SELL", atr_val)
            stop_loss = round(stop_loss, 2)
            if session_name == "opening":
                stop_loss = round(stop_loss + 0.2 * abs(stop_loss - price), 2)
            if stop_loss <= price:                       # must be above price
                stop_loss = round(price + 1.2 * atr_val, 2)
            raw_t1 = pivot["S2"] if price < pivot["S1"] else pivot["S1"]
            target1 = round(raw_t1, 2)
            target2 = round(pivot["S3"], 2)
            if target1 >= price:
                target1 = round(price - atr_val, 2)
                target2 = round(price - 2 * atr_val, 2)

        else:
            atr_sl    = atr_val * 1.2
            stop_loss = round(price - atr_sl, 2)
            target1   = round(price + atr_sl, 2)
            target2   = round(price + 2 * atr_sl, 2)

        # ── v3: strategic entry ────────────────────────────────────────────────
        levels_dict = {
            "breakout_long":  resistance1,
            "breakout_short": support1,
            "support":        support1,
            "resistance":     resistance1,
            "candle_high":    round(self.current["High"], 2),
            "candle_low":     round(self.current["Low"],  2),
        }
        ema21_val = round(self.ema21.iloc[-1], 2)

        if recommendation in ("BUY", "SHORT_SELL"):
            entry, entry_type, entry_note = _strategic_entry(
                direction=recommendation,
                price=price,
                atr_val=atr_val,
                patterns=self.patterns,
                levels=levels_dict,
                vwap=round(float(self.df["VWAP"].iloc[-1]), 2),
                poc=poc_price,
                ema21=ema21_val,
            )
            # Final guard: entry must not have leapfrogged its own target
            if recommendation == "BUY" and entry >= target1:
                target1 = round(entry + atr_val, 2)
                target2 = round(entry + 2 * atr_val, 2)
                entry_note += " (Targets rebased above entry.)"
            elif recommendation == "SHORT_SELL" and entry <= target1:
                target1 = round(entry - atr_val, 2)
                target2 = round(entry - 2 * atr_val, 2)
                entry_note += " (Targets rebased below entry.)"
            if recommendation == "SHORT_SELL":
                if short_setup == "breakdown" and entry >= support1:
                    entry = round(support1 - 0.05 * atr_val, 2)
                    entry_note += " (Short entry kept below current support.)"
                stop_loss = round(max(stop_loss, entry + 0.8 * atr_val), 2)
                if target1 >= entry:
                    target1 = round(entry - atr_val, 2)
                    target2 = round(entry - 2 * atr_val, 2)
                if target2 >= entry:
                    target2 = round(entry - 2 * atr_val, 2)
        else:
            entry      = price
            entry_type = "pullback"
            entry_note = "No actionable signal – monitoring only."

        # ── R:R anchored to strategic entry ───────────────────────────────────
        risk   = abs(entry - stop_loss)
        reward = abs(target1 - entry)
        rr     = round(reward / risk, 2) if risk > 0 else 0.0

        # ── reason text ──────────────────────────────────────────────────────
        why_trade = []
        why_avoid = []
        if recommendation == "BUY":
            if trend_bull:
                why_trade.append("EMA trend supports a long setup.")
            if price_above_vwap:
                why_trade.append("Price is holding above VWAP.")
            if rsi_val > 55:
                why_trade.append("RSI momentum is constructive.")
            if volume_spike:
                why_trade.append("Volume confirms participation.")
        elif recommendation == "SHORT_SELL":
            if not trend_bull:
                why_trade.append("EMA trend supports a short setup.")
            if not price_above_vwap:
                why_trade.append("Price is below VWAP.")
            if rsi_val < 45:
                why_trade.append("RSI momentum is weak.")
            if volume_spike:
                why_trade.append("Volume confirms participation.")
            if short_reason:
                why_trade.append(f"Short setup: {short_reason}.")
        if regime == "ranging":
            why_avoid.append("Ranging regime reduces breakout reliability.")
        if regime == "high_volatility":
            why_avoid.append("High volatility makes stops wide and fills less dependable.")
        if liquidity in ("low", "missing"):
            why_avoid.append("Liquidity is too weak for clean intraday execution.")
        if low_vol:
            why_avoid.append("Intraday volatility is too low.")
        if htf_bias == "conflict":
            why_avoid.append("Higher-timeframe trend conflicts with the intraday signal.")
        if short_reason and recommendation == "AVOID" and score < -1.5:
            why_avoid.append(short_reason)

        reason_text = f"Score: {score:.2f}. " + ", ".join(reasons) + f". Session: {session_name}."
        if session_name == "closing":
            reason_text += " Late session – tighter stops."

        # ── v2: trade quality — now with skip_reason ──────────────────────────
        trade_quality, skip_trade, skip_reason = _grade_trade(
            rr,
            confidence,
            atr_pct,
            sl_percent=round((risk / entry) * 100, 2) if entry > 0 else 0.0,
            liquidity=liquidity,
            regime=regime,
            htf_bias=htf_bias,
            direction=recommendation,
        )
        if skip_reason:
            reason_text += f" Skip: {skip_reason}."
            why_avoid.extend(skip_reason.split("; "))
        if skip_trade:
            recommendation = "AVOID"
            confidence = min(confidence, 45)
            trade_quality = "SKIP"

        position_size = _position_size(entry, stop_loss, self.account_size, self.risk_percent)
        invalidation = _invalidation_reason(recommendation, entry, stop_loss, regime)
        trade_side = "LONG" if recommendation == "BUY" else "SHORT" if recommendation == "SHORT_SELL" else "NONE"
        order_type_hint = "BUY_STOP_LIMIT" if recommendation == "BUY" else "SELL_STOP_LIMIT" if recommendation == "SHORT_SELL" else "NO_ORDER"
        execution_warning = ""
        if recommendation == "SHORT_SELL":
            execution_warning = f"Intraday short only; cover before {self.square_off_time} IST and verify broker margin/ban-list before execution."

        return {
            # ── original keys ─────────────────────────────────────────────
            "recommendation": recommendation,
            "confidence":     confidence,
            "entry":          round(entry, 2),
            "stop_loss":      stop_loss,
            "sl_percent":     round((risk / entry) * 100, 2) if entry > 0 else 0.0,
            "target1":        target1,
            "target2":        target2,
            "rr":             rr,
            "support1":       support1,
            "support2":       support2,
            "resistance1":    resistance1,
            "resistance2":    resistance2,
            "reason":         reason_text,
            "why_trade":      why_trade,
            "why_avoid":      why_avoid or (["No clean confluence across trend, VWAP, momentum, volume, and risk."] if recommendation == "AVOID" else []),
            "invalidation":   invalidation,
            "position_size":  position_size,
            "is_pm":          self.is_pm,
            # ── v2 new keys ───────────────────────────────────────────────
            "trade_quality":  trade_quality,
            "skip_trade":     skip_trade,
            "skip_reason":    skip_reason,
            "market_regime":  regime,
            "liquidity":      liquidity,
            "session":        session_name,
            "htf_bias":       htf_bias,
            "consolidating":  self.patterns["consolidating"],
            "poc_price":      poc_price,
            # ── v3 new keys ───────────────────────────────────────────────
            "entry_type":     entry_type,
            "entry_note":     entry_note,
            "trade_side":     trade_side,
            "order_type_hint": order_type_hint,
            "square_off_time": self.square_off_time,
            "short_eligible": short_eligible,
            "execution_warning": execution_warning,
            "display_recommendation": "SHORT SELL" if recommendation == "SHORT_SELL" else recommendation,
        }
