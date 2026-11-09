"""Quantitative Agent Decision Engine: Synthesizes sentiment, technicals, and predictive models into actionable trades."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from hengseng_sentiment_agent.config import AssetConfig
from hengseng_sentiment_agent.features.levels import (
    SupportResistanceCalculator,
    SupportResistanceLevels,
)
from hengseng_sentiment_agent.models.predictor import (
    PredictionOutput,
    SentimentPricePredictor,
)


@dataclass
class TickerAnalysisResult:
    ticker: str
    asset_name: str
    exchange: str
    currency: str
    sector: str
    current_price: float
    predicted_price_n1: float
    predicted_return_pct: float
    sentiment_score: float
    sentiment_label: str
    social_mentions_count: int
    news_articles_count: int
    direction: str  # BULLISH / BEARISH / NEUTRAL
    confidence_pct: float
    position: str   # LONG / SHORT / CASH
    position_size_pct: float
    stop_loss: float
    take_profit: float
    risk_reward_ratio: float
    support_resistance: SupportResistanceLevels
    divergence_signal: str
    retail_vs_news_gap: float
    social_hype_index: float
    agent_rationale: str


class QuantAgentDecisionEngine:
    """Agentic decision engine integrating technical, alternative sentiment, and predictive model signals."""

    def __init__(self, predictor: Optional[SentimentPricePredictor] = None):
        self.predictor = predictor or SentimentPricePredictor()

    def analyze_ticker(
        self,
        asset: AssetConfig,
        feature_df: pd.DataFrame,
        posts_df: pd.DataFrame,
        news_df: pd.DataFrame,
    ) -> TickerAnalysisResult:
        """Run complete quantitative synthesis on an asset."""
        if feature_df.empty:
            raise ValueError(f"Feature dataframe for {asset.ticker} is empty.")

        latest_bar = feature_df.iloc[-1]
        current_price = float(latest_bar["Close"])
        atr = float(latest_bar.get("atr_14", current_price * 0.02))

        # 1. Support & Resistance Levels
        sr_levels = SupportResistanceCalculator.extract_levels_from_df(feature_df, window=20)

        # 2. Predictive Model Prediction for n+1
        _, sent_pred = self.predictor.predict_n1(feature_df, asset.ticker)
        pred_return_pct = sent_pred.predicted_return_pct
        pred_price_n1 = sent_pred.predicted_price_n1

        # 3. Aggregated Sentiment & Creative Metrics
        fused_sent = float(latest_bar.get("fused_sentiment", 0.0))
        divergence = float(latest_bar.get("sentiment_price_divergence", 0.0))
        ret_news_gap = float(latest_bar.get("retail_vs_news_gap", 0.0))
        hype_index = float(latest_bar.get("social_hype_index", 0.0))

        sentiment_score = round(fused_sent, 3)
        if sentiment_score > 0.12:
            sentiment_label = "BULLISH"
        elif sentiment_score < -0.12:
            sentiment_label = "BEARISH"
        else:
            sentiment_label = "NEUTRAL"

        # 4. Position Sizing & Direction Synthesis
        # Direction determined by ensemble of model prediction + fused sentiment + divergence
        direction_score = (
            (1.0 if sent_pred.direction == "BULLISH" else -1.0) * 0.45
            + np.sign(sentiment_score) * 0.35
            + divergence * 0.20
        )

        confidence_pct = round(min(96.0, max(52.0, (abs(direction_score) * 35.0 + sent_pred.direction_probability * 55.0))), 1)

        if direction_score > 0.25:
            direction = "BULLISH"
            position = "LONG"
            # Volatility-adjusted Kelly sizing (5% to 20%)
            vol = float(latest_bar.get("realized_vol_10d", 0.25))
            position_size = round(min(20.0, max(5.0, (confidence_pct / 100.0) * (0.25 / max(0.1, vol)) * 12.0)), 1)

            # Long SL & TP
            # Stop-Loss: below nearest support or 1.5 * ATR
            sl_candidate = min(current_price - 1.5 * atr, sr_levels.nearest_support * 0.99)
            stop_loss = round(sl_candidate, 2)
            # Take-Profit: above current price targeting resistance or 2.5 * ATR
            tp_candidate = max(current_price + 2.5 * atr, sr_levels.nearest_resistance * 1.01)
            take_profit = round(tp_candidate, 2)

        elif direction_score < -0.25:
            direction = "BEARISH"
            position = "SHORT"
            vol = float(latest_bar.get("realized_vol_10d", 0.25))
            position_size = round(min(18.0, max(5.0, (confidence_pct / 100.0) * (0.25 / max(0.1, vol)) * 10.0)), 1)

            # Short SL & TP
            sl_candidate = max(current_price + 1.5 * atr, sr_levels.nearest_resistance * 1.01)
            stop_loss = round(sl_candidate, 2)
            tp_candidate = min(current_price - 2.5 * atr, sr_levels.nearest_support * 0.99)
            take_profit = round(tp_candidate, 2)
        else:
            direction = "NEUTRAL"
            position = "CASH / HOLD"
            position_size = 0.0
            stop_loss = round(current_price * 0.96, 2)
            take_profit = round(current_price * 1.04, 2)

        # Risk-Reward calculation
        risk = abs(current_price - stop_loss)
        reward = abs(take_profit - current_price)
        rr_ratio = round(reward / risk, 2) if risk > 0 else 2.0

        # Divergence interpretation
        if divergence > 0.5:
            div_desc = "Bullish Divergence (Retail accumulating despite price dip)"
        elif divergence < -0.5:
            div_desc = "Bearish Divergence (Retail distribution during price bounce)"
        else:
            div_desc = "Neutral Alignment (Sentiment tracking price action)"

        # 5. Formulate Agent Qualitative Rationale
        agent_rationale = self._generate_agent_rationale(
            asset=asset,
            current_price=current_price,
            direction=direction,
            position=position,
            pred_return_pct=pred_return_pct,
            sentiment_score=sentiment_score,
            div_desc=div_desc,
            ret_news_gap=ret_news_gap,
            sr_levels=sr_levels,
            total_mentions=len(posts_df),
            total_news=len(news_df),
        )

        return TickerAnalysisResult(
            ticker=asset.ticker,
            asset_name=asset.name,
            exchange=asset.exchange.value,
            currency=asset.currency,
            sector=asset.sector,
            current_price=round(current_price, 2),
            predicted_price_n1=round(pred_price_n1, 2),
            predicted_return_pct=round(pred_return_pct, 2),
            sentiment_score=sentiment_score,
            sentiment_label=sentiment_label,
            social_mentions_count=len(posts_df),
            news_articles_count=len(news_df),
            direction=direction,
            confidence_pct=confidence_pct,
            position=position,
            position_size_pct=position_size,
            stop_loss=stop_loss,
            take_profit=take_profit,
            risk_reward_ratio=rr_ratio,
            support_resistance=sr_levels,
            divergence_signal=div_desc,
            retail_vs_news_gap=round(ret_news_gap, 3),
            social_hype_index=round(hype_index, 2),
            agent_rationale=agent_rationale,
        )

    def _generate_agent_rationale(
        self,
        asset: AssetConfig,
        current_price: float,
        direction: str,
        position: str,
        pred_return_pct: float,
        sentiment_score: float,
        div_desc: str,
        ret_news_gap: float,
        sr_levels: SupportResistanceLevels,
        total_mentions: int,
        total_news: int,
    ) -> str:
        """Synthesize qualitative thesis."""
        return (
            f"{asset.name} ({asset.ticker}) displays an overall {direction.lower()} profile with a fused "
            f"sentiment score of {sentiment_score:+.2f} derived from {total_mentions} social discussions and "
            f"{total_news} institutional news items. Out-of-sample predictive modeling projects an n+1 return of "
            f"{pred_return_pct:+.2f}%. Structurally, key support rests at {asset.currency} {sr_levels.nearest_support} "
            f"({sr_levels.distance_to_support_pct:.1f}% below current) and overhead resistance at {asset.currency} "
            f"{sr_levels.nearest_resistance} ({sr_levels.distance_to_resistance_pct:.1f}% above). The alternative data "
            f"reveals {div_desc}. Recommended posture is {position} with strict risk containment."
        )
