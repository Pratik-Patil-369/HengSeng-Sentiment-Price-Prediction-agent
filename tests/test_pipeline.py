"""End-to-end integration test for the full pipeline."""

import tempfile
from pathlib import Path
import pytest

from hengseng_sentiment_agent.pipeline import SentimentPredictionPipeline


def test_full_pipeline_execution():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        pipeline = SentimentPredictionPipeline(lookback_days=10, reports_dir=tmp_path)

        # Run on a representative asset from each exchange
        tickers = ["0700.HK", "005930.KS"]
        results, base_card, sent_card = pipeline.run_pipeline(
            tickers=tickers,
            generate_html=True,
            use_cache=False,
        )

        assert len(results) == 2
        assert results[0].ticker in tickers
        assert results[1].ticker in tickers

        # Verify HTML report generated
        html_file = tmp_path / "sentiment_price_prediction_report.html"
        assert html_file.exists()
        assert html_file.stat().st_size > 1000

        # Verify CSV export
        csv_file = tmp_path / "predictions_summary.csv"
        assert csv_file.exists()

        # Verify JSON export
        json_file = tmp_path / "predictions_summary.json"
        assert json_file.exists()
