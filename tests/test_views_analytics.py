"""Pruebas de los cálculos de las vistas del dashboard."""
import importlib.util
from pathlib import Path

import pandas as pd

spec = importlib.util.spec_from_file_location(
    "analytics", Path(__file__).resolve().parents[1] / "src" / "app" / "views" / "analytics.py")
analytics = importlib.util.module_from_spec(spec)
spec.loader.exec_module(analytics)

DF = pd.DataFrame({
    "channel": ["chat", "chat", "email", "email", "email"],
    "overall_sentiment": ["negative", "positive", "negative", "negative", "neutral"],
    "overall_urgency": ["high", "low", "low", "critical", "low"],
    "outcome": ["Resolved", "Escalated", "Resolved", "Resolved", "Pending"],
    "mes": ["2025-01", "2025-01", "2025-02", "2025-02", "2025-02"],
})


def test_share_table_rows_sum_to_one_and_respect_order():
    t = analytics.share_table(DF, "overall_urgency", "overall_sentiment", order=["low", "high", "critical"],
                              columns=["negative", "neutral", "positive"])
    assert list(t.index) == ["low", "high", "critical"]
    assert list(t.columns) == ["negative", "neutral", "positive"]
    assert t.sum(axis=1).round(6).eq(1).all()


def test_rate_by_and_monthly_rate():
    assert analytics.rate_by(DF, "channel", "outcome", "Resolved").to_dict() == {"email": 200 / 3, "chat": 50.0}
    assert analytics.monthly_rate(DF, "overall_sentiment", "negative").round(2).to_dict() == {"2025-01": 50.0, "2025-02": 66.67}


def test_resolution_counts():
    assert analytics.resolution_counts(DF, "email") == (2, 3)
