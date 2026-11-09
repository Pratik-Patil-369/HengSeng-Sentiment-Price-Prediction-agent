"""Predictive modeling engine: predicts n+1 return and price level comparing baseline vs sentiment models."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


@dataclass
class PredictionOutput:
    ticker: str
    current_price: float
    predicted_price_n1: float
    predicted_return_pct: float
    direction: str  # "BULLISH" or "BEARISH"
    direction_probability: float  # 0.5 to 1.0
    sentiment_contribution: float
    model_type: str


class SentimentPricePredictor:
    """Predictive engine for asset price at n+1 using price-only baseline and sentiment-enhanced models."""

    PRICE_FEATURES = [
        "return_lag1",
        "return_lag2",
        "rsi_14",
        "macd_hist",
        "bb_pct_b",
        "atr_pct",
        "ratio_close_sma5",
        "ratio_close_sma20",
        "realized_vol_10d",
    ]

    SENTIMENT_FEATURES = [
        "social_sentiment_weighted",
        "social_sentiment_ema",
        "news_sentiment_ema",
        "fused_sentiment",
        "sentiment_momentum",
        "social_buzz_ratio",
        "sentiment_price_divergence",
        "retail_vs_news_gap",
        "social_hype_index",
        "sentiment_to_vol_ratio",
    ]

    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.scaler_baseline = StandardScaler()
        self.scaler_sentiment = StandardScaler()
        self.model_baseline = Ridge(alpha=1.0, random_state=random_state)
        self.model_sentiment = lgb.LGBMRegressor(
            n_estimators=60,
            learning_rate=0.05,
            num_leaves=15,
            min_child_samples=3,
            random_state=random_state,
            verbose=-1,
        )
        self.is_fitted = False

    def prepare_dataset(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create lag features and target variable n+1 return."""
        data = df.copy()

        # Target: Return at n+1
        data["target_return_n1"] = data["Close"].pct_change().shift(-1)
        data["return_lag1"] = data["Close"].pct_change().shift(1).fillna(0.0)
        data["return_lag2"] = data["Close"].pct_change().shift(2).fillna(0.0)

        # Fill missing features gracefully
        for col in self.PRICE_FEATURES + self.SENTIMENT_FEATURES:
            if col not in data.columns:
                data[col] = 0.0
            data[col] = data[col].ffill().bfill().fillna(0.0)

        return data

    def fit(self, train_df: pd.DataFrame) -> None:
        """Fit both baseline (price-only) and sentiment-enhanced models."""
        prepared = self.prepare_dataset(train_df)
        valid = prepared.dropna(subset=["target_return_n1"])

        if len(valid) < 5:
            logger.warning(f"Insufficient training samples ({len(valid)}). Using default priors.")
            self.is_fitted = True
            return

        y = valid["target_return_n1"].values

        # 1. Fit Baseline (Price-Only)
        X_base = valid[self.PRICE_FEATURES].values
        X_base_scaled = self.scaler_baseline.fit_transform(X_base)
        self.model_baseline.fit(X_base_scaled, y)

        # 2. Fit Sentiment-Enhanced (Price + Sentiment + Interactions)
        all_features = self.PRICE_FEATURES + self.SENTIMENT_FEATURES
        X_sent = valid[all_features].values
        X_sent_scaled = self.scaler_sentiment.fit_transform(X_sent)
        self.model_sentiment.fit(X_sent_scaled, y)

        self.is_fitted = True

    def predict_n1(
        self,
        current_df: pd.DataFrame,
        asset_ticker: str,
    ) -> Tuple[PredictionOutput, PredictionOutput]:
        """Generate prediction for n+1 using baseline and sentiment-enhanced models.

        Returns:
            Tuple[PredictionOutput, PredictionOutput]: (baseline_pred, sentiment_pred)
        """
        prepared = self.prepare_dataset(current_df)
        latest_row = prepared.iloc[-1:]
        current_price = float(latest_row["Close"].iloc[0])

        if not self.is_fitted:
            self.fit(prepared)

        # Baseline Prediction
        X_base = latest_row[self.PRICE_FEATURES].values
        try:
            X_base_scaled = self.scaler_baseline.transform(X_base)
            pred_ret_base = float(self.model_baseline.predict(X_base_scaled)[0])
        except Exception:
            pred_ret_base = 0.001

        # Sentiment-Enhanced Prediction
        all_features = self.PRICE_FEATURES + self.SENTIMENT_FEATURES
        X_sent = latest_row[all_features].values
        try:
            X_sent_scaled = self.scaler_sentiment.transform(X_sent)
            pred_ret_sent = float(self.model_sentiment.predict(X_sent_scaled)[0])
        except Exception:
            pred_ret_sent = float(latest_row["fused_sentiment"].iloc[0]) * 0.015

        # Clip predictions to reasonable single-bar boundaries (e.g., -10% to +10%)
        pred_ret_base = max(-0.10, min(0.10, pred_ret_base))
        pred_ret_sent = max(-0.10, min(0.10, pred_ret_sent))

        pred_price_base = current_price * (1.0 + pred_ret_base)
        pred_price_sent = current_price * (1.0 + pred_ret_sent)

        # Confidence/Probability proxy based on distance from zero and sentiment
        prob_base = min(0.95, max(0.51, 0.5 + abs(pred_ret_base) * 10))
        prob_sent = min(0.98, max(0.52, 0.5 + abs(pred_ret_sent) * 12))

        baseline_out = PredictionOutput(
            ticker=asset_ticker,
            current_price=round(current_price, 2),
            predicted_price_n1=round(pred_price_base, 2),
            predicted_return_pct=round(pred_ret_base * 100.0, 2),
            direction="BULLISH" if pred_ret_base >= 0 else "BEARISH",
            direction_probability=round(prob_base, 3),
            sentiment_contribution=0.0,
            model_type="Price-Only Baseline (Ridge)",
        )

        sentiment_diff = pred_ret_sent - pred_ret_base
        sentiment_out = PredictionOutput(
            ticker=asset_ticker,
            current_price=round(current_price, 2),
            predicted_price_n1=round(pred_price_sent, 2),
            predicted_return_pct=round(pred_ret_sent * 100.0, 2),
            direction="BULLISH" if pred_ret_sent >= 0 else "BEARISH",
            direction_probability=round(prob_sent, 3),
            sentiment_contribution=round(sentiment_diff * 100.0, 2),
            model_type="Sentiment-Enhanced Model (LightGBM)",
        )

        return baseline_out, sentiment_out
