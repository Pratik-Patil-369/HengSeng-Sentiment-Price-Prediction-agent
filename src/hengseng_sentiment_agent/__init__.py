"""Hang Seng & KRX Multi-Source Sentiment Price Prediction Agent."""

__version__ = "0.1.0"

from hengseng_sentiment_agent.main import main
from hengseng_sentiment_agent.pipeline import SentimentPredictionPipeline

__all__ = ["main", "SentimentPredictionPipeline", "__version__"]
