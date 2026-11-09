"""Tests for sentiment NLP analyzer, engagement weighting, and bar alignment."""

import pandas as pd
import pytest

from hengseng_sentiment_agent.sentiment.analyzer import FinancialSentimentAnalyzer
from hengseng_sentiment_agent.sentiment.aggregator import SentimentAggregator
from hengseng_sentiment_agent.sentiment.alignment import BarAlignmentEngine


def test_multilingual_sentiment_analyzer():
    analyzer = FinancialSentimentAnalyzer()

    # Cantonese / HK retail test
    res_hk_bull = analyzer.analyze_text("0700 騰訊股王回歸！大升突破阻力位，南向資金掃貨！")
    assert res_hk_bull.polarity > 0.3
    assert res_hk_bull.label == "BULLISH"

    res_hk_bear = analyzer.analyze_text("大盤暴跌破底，沽空壓力沉重，止蝕離場！")
    assert res_hk_bear.polarity < -0.3
    assert res_hk_bear.label == "BEARISH"

    # Korean retail test
    res_kr_bull = analyzer.analyze_text("삼성전자 실적 서프라이즈 대박! 외국인 순매수 가즈아 떡상!")
    assert res_kr_bull.polarity > 0.3
    assert res_kr_bull.label == "BULLISH"

    res_kr_bear = analyzer.analyze_text("악재 터졌네요... 지지선 이탈하고 떡락 중입니다. 패닉셀 손절 각")
    assert res_kr_bear.polarity < -0.3
    assert res_kr_bear.label == "BEARISH"


def test_engagement_weighting():
    aggregator = SentimentAggregator()
    # High viral engagement post should weigh more than a zero-engagement post
    w_viral = aggregator.compute_post_weight(likes=500, reposts=100, views=15000, author_followers=10000)
    w_low = aggregator.compute_post_weight(likes=1, reposts=0, views=20, author_followers=5)

    assert w_viral > w_low * 10.0


def test_bar_alignment_zero_lookahead():
    # Construct mock price bar DataFrame
    dates = pd.date_range("2026-11-01", periods=5, freq="D", tz="UTC")
    ohlcv = pd.DataFrame({
        "Open": [100.0, 102.0, 101.0, 105.0, 104.0],
        "High": [103.0, 104.0, 103.0, 107.0, 106.0],
        "Low": [99.0, 100.0, 99.5, 103.0, 103.0],
        "Close": [102.0, 101.0, 105.0, 104.0, 106.0],
        "Volume": [10000, 12000, 15000, 11000, 14000],
    }, index=dates)

    # Construct mock social posts
    posts = pd.DataFrame([
        {
            "post_id": "p1", "ticker": "0700.HK", "source": "lihkg",
            "timestamp_utc": pd.Timestamp("2026-11-02 12:00:00", tz="UTC"),
            "text": "騰訊突破！大升！", "likes": 50, "reposts": 10, "views": 1000, "author_followers": 200,
        },
        {
            "post_id": "p2", "ticker": "0700.HK", "source": "lihkg",
            "timestamp_utc": pd.Timestamp("2026-11-04 10:00:00", tz="UTC"),
            "text": "暴跌止損！", "likes": 30, "reposts": 5, "views": 800, "author_followers": 100,
        },
    ])

    engine = BarAlignmentEngine()
    aligned = engine.align_sentiment_to_bars(ohlcv, posts, pd.DataFrame())

    assert "social_sentiment_weighted" in aligned.columns
    assert "fused_sentiment" in aligned.columns
    # Post on Nov 2 should not affect bar on Nov 1
    assert aligned.loc[dates[0], "social_volume"] == 0
