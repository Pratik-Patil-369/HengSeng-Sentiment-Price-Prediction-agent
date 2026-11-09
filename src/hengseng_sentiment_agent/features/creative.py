"""Creative quantitative features: Sentiment-Price Divergence, Retail vs News Gap, and Hype Dynamics."""

from __future__ import annotations

import numpy as np
import pandas as pd


class CreativeFeatureExtractor:
    """Extracts alpha-generating interaction features between sentiment and price action."""

    @staticmethod
    def add_creative_features(df: pd.DataFrame) -> pd.DataFrame:
        """Add creative cross-asset and sentiment-price divergence signals."""
        if df.empty or len(df) < 3:
            return df.copy()

        data = df.copy()
        close = data["Close"]

        # Ensure required base columns exist
        if "fused_sentiment" not in data.columns:
            data["fused_sentiment"] = 0.0
        if "social_sentiment_weighted" not in data.columns:
            data["social_sentiment_weighted"] = 0.0
        if "news_sentiment" not in data.columns:
            data["news_sentiment"] = 0.0
        if "social_buzz_ratio" not in data.columns:
            data["social_buzz_ratio"] = 1.0

        # 1. Sentiment-Price Divergence Signal
        # Compute normalized 3-bar slope of price and sentiment
        price_norm = (close - close.rolling(5, min_periods=2).mean()) / (close.rolling(5, min_periods=2).std() + 1e-6)
        price_slope = price_norm.diff(2).fillna(0.0)
        sentiment_slope = data["fused_sentiment"].diff(2).fillna(0.0)

        # Divergence: When signs disagree
        # Positive divergence: price falling, sentiment rising (Bullish Divergence)
        # Negative divergence: price rising, sentiment falling (Bearish Divergence)
        divergence = np.where(
            (price_slope < -0.2) & (sentiment_slope > 0.1),
            1.0,  # Bullish Divergence
            np.where(
                (price_slope > 0.2) & (sentiment_slope < -0.1),
                -1.0,  # Bearish Divergence
                0.0,   # No divergence
            ),
        )
        data["sentiment_price_divergence"] = divergence

        # Continuous divergence magnitude
        data["divergence_continuous"] = sentiment_slope - price_slope

        # 2. Retail vs Institutional News Gap
        # Retail social sentiment minus institutional news sentiment
        data["retail_vs_news_gap"] = data["social_sentiment_weighted"] - data["news_sentiment"]

        # 3. Social Hype Acceleration
        # Volume spike multiplied by sentiment magnitude
        data["social_hype_index"] = data["social_buzz_ratio"] * data["social_sentiment_weighted"]

        # 4. Volatility-Sentiment Interaction (Breakout Catalyst)
        # Strong sentiment when realized volatility is compressed signals impending breakout
        if "realized_vol_10d" in data.columns:
            vol_safe = np.maximum(data["realized_vol_10d"], 0.05)
            data["sentiment_to_vol_ratio"] = data["fused_sentiment"] / vol_safe
        else:
            data["sentiment_to_vol_ratio"] = data["fused_sentiment"]

        # 5. Sentiment * Return Interaction
        if "returns" in data.columns:
            data["sentiment_return_momentum"] = data["fused_sentiment"] * data["returns"].shift(1).fillna(0.0)

        return data
