"""Social engagement weighting and multi-source sentiment aggregation engine."""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd

from hengseng_sentiment_agent.config import DEFAULT_SETTINGS
from hengseng_sentiment_agent.sentiment.analyzer import FinancialSentimentAnalyzer


class SentimentAggregator:
    """Aggregates social and news sentiment using engagement-weighted metrics."""

    def __init__(
        self,
        like_weight: float = DEFAULT_SETTINGS.like_weight,
        repost_weight: float = DEFAULT_SETTINGS.repost_weight,
        analyzer: Optional[FinancialSentimentAnalyzer] = None,
    ):
        self.like_weight = like_weight
        self.repost_weight = repost_weight
        self.analyzer = analyzer or FinancialSentimentAnalyzer()

    def compute_post_weight(
        self,
        likes: int,
        reposts: int,
        views: int,
        author_followers: int,
    ) -> float:
        """Compute the non-linear engagement weight for a post.

        Weight formula:
        w = ln(1 + views) * (1 + like_w * likes + repost_w * reposts) * ln(10 + followers)
        """
        views_factor = math.log(1.0 + max(0, views))
        social_factor = 1.0 + self.like_weight * max(0, likes) + self.repost_weight * max(0, reposts)
        follower_factor = math.log(10.0 + max(0, author_followers))

        weight = views_factor * social_factor * follower_factor
        return max(0.1, weight)

    def process_and_score_posts(self, posts_df: pd.DataFrame) -> pd.DataFrame:
        """Score each post text and compute individual engagement weights."""
        if posts_df.empty:
            return posts_df.copy()

        df = posts_df.copy()
        polarities = []
        labels = []
        confidences = []
        weights = []

        for _, row in df.iterrows():
            text = str(row.get("text", ""))
            res = self.analyzer.analyze_text(text)
            polarities.append(res.polarity)
            labels.append(res.label)
            confidences.append(res.confidence)

            w = self.compute_post_weight(
                likes=int(row.get("likes", 0)),
                reposts=int(row.get("reposts", 0)),
                views=int(row.get("views", 0)),
                author_followers=int(row.get("author_followers", 0)),
            )
            weights.append(w)

        df["sentiment_polarity"] = polarities
        df["sentiment_label"] = labels
        df["confidence"] = confidences
        df["engagement_weight"] = weights
        df["weighted_polarity"] = df["sentiment_polarity"] * df["engagement_weight"]

        return df

    def process_and_score_news(self, news_df: pd.DataFrame) -> pd.DataFrame:
        """Score each news article title and snippet."""
        if news_df.empty:
            return news_df.copy()

        df = news_df.copy()
        polarities = []
        labels = []

        for _, row in df.iterrows():
            # If news already had a validated score, use it or combine with text analysis
            existing_score = row.get("news_sentiment_score")
            text = f"{row.get('title', '')} {row.get('snippet_text', '')}"
            res = self.analyzer.analyze_text(text)

            if existing_score is not None and not np.isnan(float(existing_score)):
                final_score = 0.5 * float(existing_score) + 0.5 * res.polarity
            else:
                final_score = res.polarity

            polarities.append(round(final_score, 4))
            labels.append("BULLISH" if final_score > 0.1 else ("BEARISH" if final_score < -0.1 else "NEUTRAL"))

        df["news_polarity"] = polarities
        df["news_label"] = labels
        return df

    def aggregate_window_sentiment(self, scored_posts_df: pd.DataFrame) -> Dict[str, Any]:
        """Aggregate summary metrics for a given window or collection of posts."""
        if scored_posts_df.empty:
            return {
                "total_mentions": 0,
                "weighted_sentiment": 0.0,
                "raw_sentiment": 0.0,
                "bullish_ratio": 0.0,
                "bearish_ratio": 0.0,
                "bull_bear_ratio": 1.0,
                "sentiment_dispersion": 0.0,
                "total_views": 0,
                "total_likes": 0,
                "total_reposts": 0,
            }

        total_weight = scored_posts_df["engagement_weight"].sum()
        weighted_sentiment = (
            (scored_posts_df["weighted_polarity"].sum() / total_weight)
            if total_weight > 0
            else scored_posts_df["sentiment_polarity"].mean()
        )
        raw_sentiment = scored_posts_df["sentiment_polarity"].mean()

        total_posts = len(scored_posts_df)
        bullish_count = (scored_posts_df["sentiment_label"] == "BULLISH").sum()
        bearish_count = (scored_posts_df["sentiment_label"] == "BEARISH").sum()

        bullish_ratio = bullish_count / total_posts if total_posts > 0 else 0.0
        bearish_ratio = bearish_count / total_posts if total_posts > 0 else 0.0
        bull_bear_ratio = (bullish_count + 1) / (bearish_count + 1)

        dispersion = scored_posts_df["sentiment_polarity"].std()
        if np.isnan(dispersion):
            dispersion = 0.0

        return {
            "total_mentions": total_posts,
            "weighted_sentiment": round(float(weighted_sentiment), 4),
            "raw_sentiment": round(float(raw_sentiment), 4),
            "bullish_ratio": round(float(bullish_ratio), 4),
            "bearish_ratio": round(float(bearish_ratio), 4),
            "bull_bear_ratio": round(float(bull_bear_ratio), 4),
            "sentiment_dispersion": round(float(dispersion), 4),
            "total_views": int(scored_posts_df["views"].sum()),
            "total_likes": int(scored_posts_df["likes"].sum()),
            "total_reposts": int(scored_posts_df["reposts"].sum()),
        }
