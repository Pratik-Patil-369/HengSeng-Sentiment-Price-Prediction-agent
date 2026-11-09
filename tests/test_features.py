"""Tests for Support & Resistance, Technical features, and Creative signals."""

import numpy as np
import pandas as pd
import pytest

from hengseng_sentiment_agent.features.creative import CreativeFeatureExtractor
from hengseng_sentiment_agent.features.levels import SupportResistanceCalculator
from hengseng_sentiment_agent.features.technical import TechnicalFeatureExtractor


def test_support_resistance_levels():
    dates = pd.date_range("2026-10-01", periods=25, freq="D", tz="UTC")
    closes = np.linspace(400, 440, 25)
    df = pd.DataFrame({
        "Open": closes - 2,
        "High": closes + 5,
        "Low": closes - 5,
        "Close": closes,
        "Volume": 1000000,
    }, index=dates)

    sr = SupportResistanceCalculator.extract_levels_from_df(df, window=20)
    assert sr.current_price == round(closes[-1], 2)
    assert sr.pivot > 0
    assert sr.support_1 < sr.resistance_1
    assert sr.nearest_support <= sr.current_price
    assert sr.nearest_resistance >= sr.current_price


def test_technical_and_creative_features():
    dates = pd.date_range("2026-10-01", periods=30, freq="D", tz="UTC")
    closes = np.sin(np.linspace(0, 10, 30)) * 20 + 200
    df = pd.DataFrame({
        "Open": closes - 1,
        "High": closes + 3,
        "Low": closes - 3,
        "Close": closes,
        "Volume": 500000,
        "fused_sentiment": np.cos(np.linspace(0, 10, 30)) * 0.5,
        "social_sentiment_weighted": np.cos(np.linspace(0, 10, 30)) * 0.5,
        "news_sentiment": np.zeros(30),
        "social_buzz_ratio": np.ones(30),
    }, index=dates)

    tech = TechnicalFeatureExtractor.add_technical_features(df)
    assert "rsi_14" in tech.columns
    assert "macd" in tech.columns
    assert "atr_14" in tech.columns
    assert (tech["rsi_14"] >= 0).all() and (tech["rsi_14"] <= 100).all()

    creative = CreativeFeatureExtractor.add_creative_features(tech)
    assert "sentiment_price_divergence" in creative.columns
    assert "retail_vs_news_gap" in creative.columns
    assert "social_hype_index" in creative.columns
