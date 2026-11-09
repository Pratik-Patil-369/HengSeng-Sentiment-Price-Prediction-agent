"""Bar alignment engine: aligns asynchronous social & news sentiment with synchronous OHLCV bars.

Guarantees ZERO lookahead bias:
For any bar closing at time T, only sentiment data created at or before T is ingested.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

from hengseng_sentiment_agent.sentiment.aggregator import SentimentAggregator

logger = logging.getLogger(__name__)


class BarAlignmentEngine:
    """Aligns alternative sentiment streams to price bars."""

    def __init__(self, aggregator: Optional[SentimentAggregator] = None):
        self.aggregator = aggregator or SentimentAggregator()

    def align_sentiment_to_bars(
        self,
        ohlcv_df: pd.DataFrame,
        posts_df: pd.DataFrame,
        news_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """Align social and news data with price bars.

        Args:
            ohlcv_df: OHLCV DataFrame with UTC DatetimeIndex.
            posts_df: Scored social posts with 'timestamp_utc'.
            news_df: Scored news articles with 'published_utc'.

        Returns:
            pd.DataFrame: Augmented DataFrame with synchronized sentiment features.
        """
        if ohlcv_df.empty:
            return ohlcv_df.copy()

        df = ohlcv_df.copy().sort_index()

        # Score posts and news if not yet scored
        if not posts_df.empty and "weighted_polarity" not in posts_df.columns:
            posts_df = self.aggregator.process_and_score_posts(posts_df)
        if not news_df.empty and "news_polarity" not in news_df.columns:
            news_df = self.aggregator.process_and_score_news(news_df)

        bar_social_sent = []
        bar_social_raw = []
        bar_social_vol = []
        bar_bull_ratio = []
        bar_bear_ratio = []
        bar_total_views = []
        bar_total_likes = []

        bar_news_sent = []
        bar_news_vol = []

        # Iterate over bars to compute strictly backward-looking window aggregates
        bar_times = df.index
        for i, current_bar_time in enumerate(bar_times):
            prev_bar_time = bar_times[i - 1] if i > 0 else (current_bar_time - pd.Timedelta(days=1))

            # Filter posts: prev_bar_time < timestamp <= current_bar_time
            if not posts_df.empty:
                mask_posts = (posts_df["timestamp_utc"] > prev_bar_time) & (posts_df["timestamp_utc"] <= current_bar_time)
                window_posts = posts_df[mask_posts]
                agg = self.aggregator.aggregate_window_sentiment(window_posts)
            else:
                agg = self.aggregator.aggregate_window_sentiment(pd.DataFrame())

            bar_social_sent.append(agg["weighted_sentiment"])
            bar_social_raw.append(agg["raw_sentiment"])
            bar_social_vol.append(agg["total_mentions"])
            bar_bull_ratio.append(agg["bullish_ratio"])
            bar_bear_ratio.append(agg["bearish_ratio"])
            bar_total_views.append(agg["total_views"])
            bar_total_likes.append(agg["total_likes"])

            # Filter news: prev_bar_time < published <= current_bar_time
            if not news_df.empty:
                mask_news = (news_df["published_utc"] > prev_bar_time) & (news_df["published_utc"] <= current_bar_time)
                window_news = news_df[mask_news]
                if not window_news.empty:
                    bar_news_sent.append(float(window_news["news_polarity"].mean()))
                    bar_news_vol.append(len(window_news))
                else:
                    bar_news_sent.append(0.0)
                    bar_news_vol.append(0)
            else:
                bar_news_sent.append(0.0)
                bar_news_vol.append(0)

        # Assign raw aligned features
        df["social_sentiment_weighted"] = bar_social_sent
        df["social_sentiment_raw"] = bar_social_raw
        df["social_volume"] = bar_social_vol
        df["social_bull_ratio"] = bar_bull_ratio
        df["social_bear_ratio"] = bar_bear_ratio
        df["social_views"] = bar_total_views
        df["social_likes"] = bar_total_likes

        df["news_sentiment"] = bar_news_sent
        df["news_volume"] = bar_news_vol

        # Apply continuous exponential decay for periods with low/sparse social activity
        df["social_sentiment_ema"] = df["social_sentiment_weighted"].ewm(span=3, min_periods=1).mean()
        df["news_sentiment_ema"] = df["news_sentiment"].ewm(span=3, min_periods=1).mean()

        # Fused Sentiment (Combines Social + News)
        # 60% Social (retail pulse) + 40% News (fundamental institutional catalyst)
        df["fused_sentiment"] = 0.6 * df["social_sentiment_ema"] + 0.4 * df["news_sentiment_ema"]

        # Sentiment Momentum: rate of change of fused sentiment
        df["sentiment_momentum"] = df["fused_sentiment"].diff().fillna(0.0)

        # Social Buzz Ratio: current volume vs 5-period moving average
        vol_ma = df["social_volume"].rolling(window=5, min_periods=1).mean()
        df["social_buzz_ratio"] = (df["social_volume"] + 1.0) / (vol_ma + 1.0)

        return df
