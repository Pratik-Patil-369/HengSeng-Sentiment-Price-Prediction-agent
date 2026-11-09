"""Tests for predictive modeling and walk-forward validation."""

import numpy as np
import pandas as pd
import pytest

from hengseng_sentiment_agent.features.creative import CreativeFeatureExtractor
from hengseng_sentiment_agent.features.technical import TechnicalFeatureExtractor
from hengseng_sentiment_agent.models.predictor import SentimentPricePredictor
from hengseng_sentiment_agent.models.validation import WalkForwardValidator


def test_predictor_n1_forecast():
    dates = pd.date_range("2026-10-01", periods=35, freq="D", tz="UTC")
    closes = np.cumsum(np.random.normal(0.5, 2.0, 35)) + 100
    df = pd.DataFrame({
        "Open": closes - 1,
        "High": closes + 2,
        "Low": closes - 2,
        "Close": closes,
        "Volume": 1000000,
        "fused_sentiment": np.random.uniform(-0.5, 0.5, 35),
        "social_sentiment_weighted": np.random.uniform(-0.5, 0.5, 35),
        "news_sentiment": np.random.uniform(-0.2, 0.2, 35),
        "social_buzz_ratio": np.random.uniform(0.8, 1.5, 35),
    }, index=dates)

    df = TechnicalFeatureExtractor.add_technical_features(df)
    df = CreativeFeatureExtractor.add_creative_features(df)

    predictor = SentimentPricePredictor()
    base_pred, sent_pred = predictor.predict_n1(df, "0700.HK")

    assert base_pred.current_price > 0
    assert sent_pred.predicted_price_n1 > 0
    assert sent_pred.direction in ["BULLISH", "BEARISH"]
    assert 0.5 <= sent_pred.direction_probability <= 1.0


def test_walk_forward_validator():
    dates = pd.date_range("2026-10-01", periods=30, freq="D", tz="UTC")
    closes = np.linspace(100, 130, 30) + np.random.normal(0, 1, 30)
    df = pd.DataFrame({
        "Open": closes,
        "High": closes + 1,
        "Low": closes - 1,
        "Close": closes,
        "Volume": 500000,
        "fused_sentiment": np.linspace(0.1, 0.6, 30),
        "social_sentiment_weighted": np.linspace(0.1, 0.6, 30),
        "news_sentiment": np.zeros(30),
        "social_buzz_ratio": np.ones(30),
    }, index=dates)

    df = TechnicalFeatureExtractor.add_technical_features(df)
    df = CreativeFeatureExtractor.add_creative_features(df)

    validator = WalkForwardValidator(n_splits=3)
    base_card, sent_card = validator.evaluate(df)

    assert base_card.total_eval_samples > 0
    assert sent_card.total_eval_samples > 0
    assert 0 <= base_card.directional_accuracy_pct <= 100
    assert 0 <= sent_card.directional_accuracy_pct <= 100
