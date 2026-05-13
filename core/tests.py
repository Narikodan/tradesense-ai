from __future__ import annotations

from unittest.mock import patch

import pandas as pd
from django.core.cache import cache
from django.template.loader import render_to_string
from django.test import RequestFactory, TestCase, override_settings

from core.views import analyze_stock
from services.analysis_service import AnalysisBundle, get_analysis_bundle
from services.market_data import MarketSnapshot


@override_settings(CACHES={"default": {"BACKEND": "django.core.cache.backends.locmem.LocMemCache"}})
class AnalyzeViewTests(TestCase):
    def setUp(self):
        cache.clear()

    def test_analyze_view_renders_clean_error(self):
        request = RequestFactory().get("/analyze/?symbol=RELIANCE")
        snapshot = MarketSnapshot("RELIANCE.NS", "Reliance", 0, 0, {}, pd.DataFrame(), pd.DataFrame(), "No daily data found.")
        bundle = AnalysisBundle(snapshot, {}, {}, {}, "", snapshot.error)
        with patch("core.views.get_analysis_bundle", return_value=bundle):
            response = analyze_stock(request)
        self.assertContains(response, "Analysis unavailable")
        self.assertNotContains(response, "Traceback")

    def test_analyze_view_handles_missing_symbol(self):
        request = RequestFactory().get("/analyze/")
        response = analyze_stock(request)
        self.assertContains(response, "No symbol provided.")

    def test_analysis_bundle_caches_data_errors_consistently(self):
        snapshot = MarketSnapshot("RELIANCE.NS", "Reliance", 0, 0, {}, pd.DataFrame(), pd.DataFrame(), "No market data returned.")
        with patch("services.analysis_service.get_market_snapshot", return_value=snapshot) as mocked:
            first = get_analysis_bundle("RELIANCE")
            second = get_analysis_bundle("RELIANCE")
        self.assertEqual(first.error, "No market data returned.")
        self.assertEqual(second.error, "No market data returned.")
        self.assertEqual(mocked.call_count, 1)

    def test_analysis_cards_template_renders_existing_fields(self):
        setup = {
            "recommendation": "AVOID",
            "confidence": 40,
            "entry": 100,
            "stop_loss": 95,
            "sl_percent": 5,
            "target1": 105,
            "target2": 110,
            "rr": 1.0,
            "support1": 95,
            "support2": 90,
            "resistance1": 105,
            "resistance2": 110,
            "reason": "No clean setup.",
            "why_avoid": ["Risk/reward is weak."],
            "invalidation": "Wait for confirmation.",
            "position_size": {"shares": 0, "risk_per_share": 5, "note": "Educational only."},
            "trade_quality": "SKIP",
            "skip_trade": True,
            "market_regime": "neutral",
            "liquidity": "normal",
            "session": "swing",
            "htf_bias": "n/a",
            "consolidating": False,
            "poc_price": None,
            "entry_note": "Monitoring only.",
        }
        html = render_to_string(
            "partials/analysis_cards.html",
            {
                "name": "Reliance",
                "symbol": "RELIANCE.NS",
                "ltp": 100,
                "pct_change": 1.2,
                "swing": setup,
                "intraday": setup | {"session": "midday", "is_pm": False},
                "disclaimer": "Educational decision-support only, not financial advice.",
            },
        )
        self.assertIn("Swing Setup", html)
        self.assertIn("Intraday Setup", html)
        self.assertIn("Educational decision-support", html)
