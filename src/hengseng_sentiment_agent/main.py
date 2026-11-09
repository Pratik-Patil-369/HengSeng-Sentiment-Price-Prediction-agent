"""Command-line interface for the Hang Seng & KRX Sentiment Price Prediction Agent."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from hengseng_sentiment_agent.config import ALL_ASSETS, DEFAULT_SETTINGS
from hengseng_sentiment_agent.pipeline import SentimentPredictionPipeline


def setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="hengseng-sentiment-agent",
        description="Institutional Multi-Source Sentiment Price Prediction Pipeline for HKEX & KRX Assets",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Enable verbose debug logging"
    )

    subparsers = parser.add_subparsers(dest="command", help="Available subcommands")

    # Command: run
    run_parser = subparsers.add_parser("run", help="Run full pipeline: ingest, predict, validate, and report")
    run_parser.add_argument(
        "--exchange",
        choices=["ALL", "HKEX", "KRX"],
        default="ALL",
        help="Target exchange universe (default: ALL)",
    )
    run_parser.add_argument(
        "--lookback",
        type=int,
        default=DEFAULT_SETTINGS.lookback_days,
        help="Lookback days for social & news sentiment (default: 10)",
    )
    run_parser.add_argument(
        "--sort-by",
        choices=["social_mentions_count", "sentiment_score", "predicted_return_pct", "direction"],
        default="social_mentions_count",
        help="Column to sort terminal leaderboard by",
    )
    run_parser.add_argument(
        "--no-cache", action="store_true", help="Bypass local parquet cache and refetch fresh data"
    )
    run_parser.add_argument(
        "--output-dir", type=str, default=str(DEFAULT_SETTINGS.reports_dir), help="Directory to save reports"
    )

    # Command: predict
    predict_parser = subparsers.add_parser("predict", help="Predict n+1 price for a single ticker")
    predict_parser.add_argument(
        "--ticker",
        type=str,
        required=True,
        help="Ticker symbol (e.g. 0700.HK, 9988.HK, 005930.KS)",
    )
    predict_parser.add_argument(
        "--no-cache", action="store_true", help="Bypass local cache"
    )

    # Command: report
    report_parser = subparsers.add_parser("report", help="Regenerate visual HTML report from cached pipeline data")
    report_parser.add_argument(
        "--output-dir", type=str, default=str(DEFAULT_SETTINGS.reports_dir), help="Output directory"
    )

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    setup_logging(args.verbose if hasattr(args, "verbose") else False)

    if not args.command or args.command == "run":
        exchange = getattr(args, "exchange", "ALL")
        lookback = getattr(args, "lookback", 10)
        use_cache = not getattr(args, "no_cache", False)
        output_dir = Path(getattr(args, "output_dir", str(DEFAULT_SETTINGS.reports_dir)))

        pipeline = SentimentPredictionPipeline(lookback_days=lookback, reports_dir=output_dir)
        pipeline.run_pipeline(exchange=exchange, use_cache=use_cache, generate_html=True)

    elif args.command == "predict":
        ticker = args.ticker.upper()
        if ticker not in ALL_ASSETS:
            print(f"Error: Ticker '{ticker}' not found in asset universe. Available: {list(ALL_ASSETS.keys())}")
            sys.exit(1)

        pipeline = SentimentPredictionPipeline(reports_dir=DEFAULT_SETTINGS.reports_dir)
        pipeline.run_pipeline(tickers=[ticker], use_cache=not args.no_cache, generate_html=False)

    elif args.command == "report":
        pipeline = SentimentPredictionPipeline(reports_dir=Path(args.output_dir))
        pipeline.run_pipeline(exchange="ALL", use_cache=True, generate_html=True)


if __name__ == "__main__":
    main()
