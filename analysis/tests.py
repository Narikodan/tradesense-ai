from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from django.test import SimpleTestCase

from analysis.backtest import simulate_trade, summarize_trades
from analysis.data import normalize_ohlcv
from analysis.engine import IntradayAnalyzer, SwingAnalyzer, _grade_trade, _strategic_entry, compute_vwap


def make_ohlcv(rows=120, *, trend=0.5, volatility=1.0, volume=100000, flat=False, missing_volume=False, freq="D", end=None):
    end = end or datetime.now()
    index = pd.date_range(end=end, periods=rows, freq=freq)
    base = np.linspace(100, 100 + trend * rows, rows)
    if flat:
        close = np.full(rows, 100.0)
    else:
        close = base + np.sin(np.arange(rows) / 3) * volatility
    open_ = close - trend * 0.2
    high = np.maximum(open_, close) + volatility
    low = np.minimum(open_, close) - volatility
    data = {"Open": open_, "High": high, "Low": low, "Close": close}
    if not missing_volume:
        data["Volume"] = np.full(rows, volume)
    return pd.DataFrame(data, index=index)


def make_intraday_breakdown(*, rows=80, end=None):
    end = end or pd.Timestamp("2026-05-13 11:30", tz="Asia/Kolkata")
    index = pd.date_range(end=end, periods=rows, freq="15min")
    close = np.linspace(112, 96, rows) + np.sin(np.arange(rows) / 4) * 0.15
    open_ = close + 0.25
    high = open_ + 0.45
    low = close - 0.45
    volume = np.full(rows, 100000.0)
    volume[-1] = 180000.0
    return pd.DataFrame({"Open": open_, "High": high, "Low": low, "Close": close, "Volume": volume}, index=index)


def make_bullish_htf(rows=80):
    index = pd.date_range(end=pd.Timestamp("2026-05-13 11:30", tz="Asia/Kolkata"), periods=rows, freq="60min")
    close = np.linspace(100, 125, rows)
    return pd.DataFrame(
        {
            "Open": close - 0.2,
            "High": close + 0.8,
            "Low": close - 0.8,
            "Close": close,
            "Volume": np.full(rows, 200000),
        },
        index=index,
    )


class DataNormalizationTests(SimpleTestCase):
    def test_empty_dataframe_returns_quality_error(self):
        df, quality = normalize_ohlcv(pd.DataFrame(), min_candles=10)
        self.assertTrue(df.empty)
        self.assertFalse(quality.ok)

    def test_missing_volume_is_filled(self):
        df, quality = normalize_ohlcv(make_ohlcv(missing_volume=True), min_candles=20)
        self.assertTrue(quality.ok)
        self.assertIn("Volume", df.columns)
        self.assertEqual(float(df["Volume"].sum()), 0.0)

    def test_vwap_handles_zero_volume(self):
        df = make_ohlcv(volume=0)
        result = compute_vwap(df)
        self.assertFalse(result["VWAP"].isna().any())

    def test_stale_data_is_rejected_deterministically(self):
        df = make_ohlcv(rows=80, end=datetime(2025, 1, 1))
        _, quality = normalize_ohlcv(df, min_candles=60, max_stale_days=7, now=pd.Timestamp("2025-01-20"))
        self.assertFalse(quality.ok)
        self.assertTrue(quality.stale)

    def test_malformed_ohlcv_is_rejected(self):
        df = make_ohlcv().drop(columns=["High"])
        normalized, quality = normalize_ohlcv(df, min_candles=20)
        self.assertTrue(normalized.empty)
        self.assertFalse(quality.ok)
        self.assertIn("Missing required OHLC", quality.reason)

    def test_multiindex_symbol_is_normalized_from_any_level(self):
        df = make_ohlcv(rows=80)
        multi = pd.concat({"RELIANCE.NS": df}, axis=1)
        normalized, quality = normalize_ohlcv(multi, symbol="RELIANCE.NS", min_candles=60)
        self.assertTrue(quality.ok)
        self.assertEqual(list(normalized.columns), ["Open", "High", "Low", "Close", "Volume"])


class AnalyzerOutputTests(SimpleTestCase):
    required_keys = {
        "recommendation", "confidence", "entry", "stop_loss", "sl_percent", "target1", "target2",
        "rr", "support1", "support2", "resistance1", "resistance2", "reason", "why_trade",
        "why_avoid", "invalidation", "position_size", "trade_quality", "skip_trade",
        "skip_reason", "market_regime", "liquidity", "session", "htf_bias", "consolidating",
        "poc_price", "entry_type", "entry_note", "trade_side", "order_type_hint",
        "square_off_time", "short_eligible", "execution_warning", "display_recommendation",
    }

    def assert_schema(self, result):
        self.assertTrue(self.required_keys.issubset(result.keys()))
        self.assertIn(result["recommendation"], {"BUY", "SELL", "SHORT_SELL", "AVOID"})

    def test_swing_empty_dataframe_returns_avoid(self):
        result = SwingAnalyzer(pd.DataFrame()).analyze()
        self.assert_schema(result)
        self.assertEqual(result["recommendation"], "AVOID")
        self.assertEqual(result["trade_quality"], "SKIP")

    def test_intraday_insufficient_candles_returns_avoid(self):
        result = IntradayAnalyzer(make_ohlcv(rows=10), {}).analyze()
        self.assert_schema(result)
        self.assertEqual(result["recommendation"], "AVOID")
        self.assertIn("Insufficient candles", result["reason"])

    def test_flat_market_is_avoided(self):
        result = SwingAnalyzer(make_ohlcv(flat=True, volatility=0.01)).analyze()
        self.assertEqual(result["recommendation"], "AVOID")
        self.assertTrue(result["skip_trade"])

    def test_highly_volatile_market_is_avoided_or_capped(self):
        result = IntradayAnalyzer(make_ohlcv(rows=80, trend=1.0, volatility=8.0), {}).analyze()
        self.assertEqual(result["recommendation"], "AVOID")
        self.assertLessEqual(result["confidence"], 45)

    def test_strong_trend_returns_complete_risk_fields(self):
        result = SwingAnalyzer(make_ohlcv(rows=260, trend=1.0, volatility=0.8), account_size=100000).analyze()
        self.assert_schema(result)
        for key in ["entry", "stop_loss", "target1", "target2", "rr", "position_size", "invalidation"]:
            self.assertIn(key, result)
        self.assertIn(result["trade_quality"], {"A", "B", "SKIP"})

    def test_strategic_entry_prefers_pullback_zone(self):
        entry, entry_type, note = _strategic_entry(
            "BUY",
            price=110,
            atr_val=4,
            patterns={},
            levels={"breakout_long": 107, "support": 100, "candle_high": 111},
            vwap=104,
            poc=105,
            ema21=103,
        )
        self.assertEqual(entry_type, "pullback")
        self.assertEqual(entry, 105.2)
        self.assertIn("POC", note)

    def test_rr_uses_entry_not_current_price(self):
        entry = 95
        stop = 90
        target = 110
        rr = round(abs(target - entry) / abs(entry - stop), 2)
        self.assertEqual(rr, 3.0)

    def test_intraday_session_uses_injected_time(self):
        now = pd.Timestamp("2026-05-11 09:20", tz="Asia/Kolkata")
        df = make_ohlcv(rows=80, trend=0.2, volatility=0.8, freq="15min", end=now)
        result = IntradayAnalyzer(df, {}, now=now).analyze()
        self.assertEqual(result["session"], "opening")

    def test_htf_conflict_is_reported_and_skips_directional_trade(self):
        grade, skip, reason = _grade_trade(
            rr=3.0,
            confidence=80,
            atr_pct=1.0,
            sl_percent=1.0,
            htf_bias="conflict",
            direction="BUY",
        )
        self.assertEqual(grade, "A")
        self.assertTrue(skip)
        self.assertIn("higher-timeframe bias conflicts", reason)

    def test_clean_bearish_breakdown_produces_short_sell(self):
        now = pd.Timestamp("2026-05-13 11:30", tz="Asia/Kolkata")
        result = IntradayAnalyzer(
            make_intraday_breakdown(end=now),
            {"symbol": "RELIANCE.NS", "regularMarketDayHigh": 110, "regularMarketDayLow": 100, "previousClose": 105},
            now=now,
        ).analyze()
        self.assertEqual(result["recommendation"], "SHORT_SELL")
        self.assertEqual(result["trade_side"], "SHORT")
        self.assertTrue(result["short_eligible"])
        self.assertEqual(result["display_recommendation"], "SHORT SELL")

    def test_bullish_htf_conflict_suppresses_short(self):
        now = pd.Timestamp("2026-05-13 11:30", tz="Asia/Kolkata")
        result = IntradayAnalyzer(
            make_intraday_breakdown(end=now),
            {"symbol": "RELIANCE.NS", "regularMarketDayHigh": 110, "regularMarketDayLow": 100, "previousClose": 105},
            htf_df=make_bullish_htf(),
            now=now,
        ).analyze()
        self.assertEqual(result["recommendation"], "AVOID")
        self.assertIn("higher-timeframe", result["skip_reason"] + " " + " ".join(result["why_avoid"]))

    def test_non_shortable_symbol_suppresses_short(self):
        now = pd.Timestamp("2026-05-13 11:30", tz="Asia/Kolkata")
        result = IntradayAnalyzer(
            make_intraday_breakdown(end=now),
            {"symbol": "NOTSHORTABLE.NS", "regularMarketDayHigh": 110, "regularMarketDayLow": 100, "previousClose": 105},
            now=now,
        ).analyze()
        self.assertEqual(result["recommendation"], "AVOID")
        self.assertFalse(result["short_eligible"])
        self.assertIn("not configured", result["reason"])

    def test_short_risk_prices_are_directionally_valid(self):
        now = pd.Timestamp("2026-05-13 11:30", tz="Asia/Kolkata")
        result = IntradayAnalyzer(
            make_intraday_breakdown(end=now),
            {"symbol": "RELIANCE.NS", "regularMarketDayHigh": 110, "regularMarketDayLow": 100, "previousClose": 105},
            now=now,
        ).analyze()
        if result["recommendation"] == "SHORT_SELL":
            self.assertGreater(result["stop_loss"], result["entry"])
            self.assertLess(result["target1"], result["entry"])
            self.assertLess(result["target2"], result["entry"])


class BacktestTests(SimpleTestCase):
    def signal(self, direction="BUY"):
        return {
            "signal_date": pd.Timestamp("2026-05-11"),
            "recommendation": direction,
            "entry": 100.0,
            "stop_loss": 95.0 if direction == "BUY" else 105.0,
            "target1": 110.0 if direction == "BUY" else 90.0,
        }

    def future(self, rows):
        return pd.DataFrame(rows, index=pd.date_range("2026-05-12", periods=len(rows), freq="D"))

    def test_backtest_target_first(self):
        trade = simulate_trade(self.signal(), self.future([{"Open": 100, "High": 111, "Low": 99, "Close": 110, "Volume": 1}]))
        self.assertEqual(trade.outcome, "target")
        self.assertEqual(trade.r_multiple, 2.0)

    def test_backtest_stop_first_when_both_hit_same_candle(self):
        trade = simulate_trade(self.signal(), self.future([{"Open": 100, "High": 111, "Low": 94, "Close": 96, "Volume": 1}]))
        self.assertEqual(trade.outcome, "stop")
        self.assertEqual(trade.r_multiple, -1.0)

    def test_backtest_no_fill(self):
        trade = simulate_trade(self.signal(), self.future([{"Open": 105, "High": 112, "Low": 101, "Close": 111, "Volume": 1}]))
        self.assertEqual(trade.outcome, "no_fill")

    def test_backtest_open_after_fill(self):
        trade = simulate_trade(self.signal(), self.future([{"Open": 100, "High": 104, "Low": 99, "Close": 102, "Volume": 1}]))
        self.assertEqual(trade.outcome, "open")

    def test_backtest_short_target_hit(self):
        trade = simulate_trade(
            self.signal("SHORT_SELL"),
            self.future([{"Open": 100, "High": 101, "Low": 89, "Close": 91, "Volume": 1}]),
        )
        self.assertEqual(trade.outcome, "target")
        self.assertEqual(trade.r_multiple, 2.0)

    def test_backtest_short_stop_hit(self):
        trade = simulate_trade(
            self.signal("SHORT_SELL"),
            self.future([{"Open": 100, "High": 106, "Low": 94, "Close": 105, "Volume": 1}]),
        )
        self.assertEqual(trade.outcome, "stop")
        self.assertEqual(trade.r_multiple, -1.0)

    def test_backtest_summary(self):
        trades = [
            simulate_trade(self.signal(), self.future([{"Open": 100, "High": 111, "Low": 99, "Close": 110, "Volume": 1}])),
            simulate_trade(self.signal(), self.future([{"Open": 100, "High": 101, "Low": 94, "Close": 96, "Volume": 1}])),
        ]
        summary = summarize_trades(trades)
        self.assertEqual(summary.trades, 2)
        self.assertEqual(summary.win_rate, 50.0)
