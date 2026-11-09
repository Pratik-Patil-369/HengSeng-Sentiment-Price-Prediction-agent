"""Tests for ingestion components: Market Data, Apify Social, and Exa News."""

import pandas as pd
import pytest

from hengseng_sentiment_agent.config import HKEX_ASSETS, KRX_ASSETS
from hengseng_sentiment_agent.ingestion.apify_client import ApifySocialIngestor
from hengseng_sentiment_agent.ingestion.exa_client import ExaNewsIngestor
from hengseng_sentiment_agent.ingestion.market_data import MarketDataIngestor


def test_market_data_ingestion():
    asset = HKEX_ASSETS["0700.HK"]
    ingestor = MarketDataIngestor()
    df = ingestor.fetch_ohlcv(asset=asset, lookback_days=15, interval="1d", use_cache=False)

    assert not df.empty
    assert "Close" in df.columns
    assert "Open" in df.columns
    assert "High" in df.columns
    assert "Low" in df.columns
    assert "Volume" in df.columns
    assert df.index.tz is not None  # UTC timezone confirmed
    assert str(df.index.tz) == "UTC"


def test_apify_social_ingestion():
    asset = HKEX_ASSETS["0700.HK"]
    ingestor = ApifySocialIngestor()
    posts_df = ingestor.scrape_posts_for_asset(asset=asset, lookback_days=10, use_cache=False)

    assert not posts_df.empty
    expected_cols = ["post_id", "ticker", "source", "timestamp_utc", "text", "likes", "reposts", "views", "author_followers"]
    for col in expected_cols:
        assert col in posts_df.columns

    # Verify UTC timestamp alignment
    assert str(posts_df["timestamp_utc"].dt.tz) == "UTC"
    assert (posts_df["views"] >= 0).all()


def test_exa_news_ingestion():
    asset = KRX_ASSETS["005930.KS"]
    ingestor = ExaNewsIngestor()
    news_df = ingestor.fetch_news_for_asset(asset=asset, lookback_days=10, use_cache=False)

    assert not news_df.empty
    assert "title" in news_df.columns
    assert "url" in news_df.columns
    assert "snippet_text" in news_df.columns
    assert "published_utc" in news_df.columns
