"""Full end-to-end pipeline orchestrator for Hang Seng & KRX Sentiment Price Prediction Agent."""

from __future__ import annotations

import json
import logging
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

from hengseng_sentiment_agent.agent.decision import (
    QuantAgentDecisionEngine,
    TickerAnalysisResult,
)
from hengseng_sentiment_agent.config import (
    ALL_ASSETS,
    HKEX_ASSETS,
    KRX_ASSETS,
    AssetConfig,
    Exchange,
    DEFAULT_SETTINGS,
)
from hengseng_sentiment_agent.features.creative import CreativeFeatureExtractor
from hengseng_sentiment_agent.features.technical import TechnicalFeatureExtractor
from hengseng_sentiment_agent.ingestion.apify_client import ApifySocialIngestor
from hengseng_sentiment_agent.ingestion.exa_client import ExaNewsIngestor
from hengseng_sentiment_agent.ingestion.market_data import MarketDataIngestor
from hengseng_sentiment_agent.models.predictor import SentimentPricePredictor
from hengseng_sentiment_agent.models.validation import (
    ValidationScorecard,
    WalkForwardValidator,
)
from hengseng_sentiment_agent.reporting.cli_summary import CLISummaryReporter
from hengseng_sentiment_agent.reporting.html_report import VisualReportGenerator
from hengseng_sentiment_agent.sentiment.alignment import BarAlignmentEngine

logger = logging.getLogger(__name__)


class SentimentPredictionPipeline:
    """Orchestrates end-to-end ingestion, feature alignment, prediction, validation, and reporting."""

    def __init__(
        self,
        lookback_days: int = DEFAULT_SETTINGS.lookback_days,
        reports_dir: Optional[Path] = None,
    ):
        self.lookback_days = lookback_days
        self.reports_dir = reports_dir or DEFAULT_SETTINGS.reports_dir
        self.reports_dir.mkdir(parents=True, exist_ok=True)

        # Ingestion components
        self.market_ingestor = MarketDataIngestor()
        self.social_ingestor = ApifySocialIngestor()
        self.news_ingestor = ExaNewsIngestor()

        # Processing & modeling
        self.alignment_engine = BarAlignmentEngine()
        self.agent_engine = QuantAgentDecisionEngine()
        self.validator = WalkForwardValidator(n_splits=DEFAULT_SETTINGS.n_splits_cv)
        self.report_generator = VisualReportGenerator(output_dir=self.reports_dir)

    def run_pipeline(
        self,
        exchange: Optional[str] = "ALL",
        tickers: Optional[List[str]] = None,
        generate_html: bool = True,
        use_cache: bool = True,
    ) -> Tuple[List[TickerAnalysisResult], ValidationScorecard, ValidationScorecard]:
        """Execute full end-to-end workflow across selected assets."""
        # 1. Resolve Target Assets
        target_assets = self._resolve_target_assets(exchange, tickers)
        logger.info(f"Running pipeline for {len(target_assets)} assets across {exchange}...")

        analysis_results: List[TickerAnalysisResult] = []
        all_aligned_dfs: Dict[str, pd.DataFrame] = {}

        for asset in target_assets:
            logger.info(f"Processing asset: {asset.ticker} ({asset.name}) [{asset.exchange.value}]...")

            # Ingest Market Data
            ohlcv_df = self.market_ingestor.fetch_ohlcv(
                asset=asset,
                lookback_days=max(45, self.lookback_days + 30),
                use_cache=use_cache,
            )

            # Ingest Social Posts (Apify / LIHKG / Futu / Naver / Telegram)
            posts_df = self.social_ingestor.scrape_posts_for_asset(
                asset=asset,
                lookback_days=self.lookback_days,
                use_cache=use_cache,
            )

            # Ingest News Articles (Exa.ai / Financial Press)
            news_df = self.news_ingestor.fetch_news_for_asset(
                asset=asset,
                lookback_days=self.lookback_days,
                use_cache=use_cache,
            )

            # Bar Alignment (Zero Lookahead Bias)
            aligned_df = self.alignment_engine.align_sentiment_to_bars(
                ohlcv_df=ohlcv_df,
                posts_df=posts_df,
                news_df=news_df,
            )

            # Feature Engineering: Technical + Creative Alpha Signals
            tech_df = TechnicalFeatureExtractor.add_technical_features(aligned_df)
            feature_df = CreativeFeatureExtractor.add_creative_features(tech_df)
            all_aligned_dfs[asset.ticker] = feature_df

            # Agent Decision & Prediction Synthesis
            result = self.agent_engine.analyze_ticker(
                asset=asset,
                feature_df=feature_df,
                posts_df=posts_df,
                news_df=news_df,
            )
            analysis_results.append(result)

        # 2. Out-of-Sample Walk-Forward Validation
        logger.info("Executing rigorous out-of-sample walk-forward validation...")
        base_card, sent_card = self._run_aggregate_validation(all_aligned_dfs)

        # 3. Export Data (JSON & CSV)
        self._export_results(analysis_results, base_card, sent_card)

        # 4. Generate Visual HTML Report
        if generate_html:
            html_path = self.report_generator.generate_html_report(
                analysis_results=analysis_results,
                baseline_scorecard=base_card,
                sentiment_scorecard=sent_card,
            )
            logger.info(f"Interactive Visual Report generated at: {html_path}")

        # 5. Print CLI Summary Tables
        CLISummaryReporter.print_leaderboard(analysis_results, sort_by="social_mentions_count")
        CLISummaryReporter.print_validation_scorecards(base_card, sent_card)

        return analysis_results, base_card, sent_card

    def _resolve_target_assets(
        self,
        exchange: Optional[str],
        tickers: Optional[List[str]],
    ) -> List[AssetConfig]:
        """Resolve list of AssetConfig to process."""
        if tickers:
            return [ALL_ASSETS[t] for t in tickers if t in ALL_ASSETS]

        if exchange == "HKEX":
            return list(HKEX_ASSETS.values())
        elif exchange == "KRX":
            return list(KRX_ASSETS.values())
        else:
            return list(ALL_ASSETS.values())

    def _run_aggregate_validation(
        self,
        aligned_dfs: Dict[str, pd.DataFrame],
    ) -> Tuple[ValidationScorecard, ValidationScorecard]:
        """Run walk-forward validation across pooled or representative ticker series."""
        base_scores = []
        sent_scores = []

        for ticker, df in aligned_dfs.items():
            b, s = self.validator.evaluate(df)
            base_scores.append(b)
            sent_scores.append(s)

        if not base_scores:
            return self.validator._mock_scorecards()

        # Compute average metrics across universe
        avg_base = ValidationScorecard(
            model_name="Price-Only Baseline (Ridge)",
            directional_accuracy_pct=round(float(sum(x.directional_accuracy_pct for x in base_scores) / len(base_scores)), 1),
            mae=round(float(sum(x.mae for x in base_scores) / len(base_scores)), 4),
            rmse=round(float(sum(x.rmse for x in base_scores) / len(base_scores)), 4),
            information_coefficient=round(float(sum(x.information_coefficient for x in base_scores) / len(base_scores)), 3),
            cumulative_pnl_pct=round(float(sum(x.cumulative_pnl_pct for x in base_scores) / len(base_scores)), 2),
            sharpe_ratio=round(float(sum(x.sharpe_ratio for x in base_scores) / len(base_scores)), 2),
            total_eval_samples=sum(x.total_eval_samples for x in base_scores),
        )

        avg_sent = ValidationScorecard(
            model_name="Sentiment-Enhanced Model (LightGBM)",
            directional_accuracy_pct=round(float(sum(x.directional_accuracy_pct for x in sent_scores) / len(sent_scores)), 1),
            mae=round(float(sum(x.mae for x in sent_scores) / len(sent_scores)), 4),
            rmse=round(float(sum(x.rmse for x in sent_scores) / len(sent_scores)), 4),
            information_coefficient=round(float(sum(x.information_coefficient for x in sent_scores) / len(sent_scores)), 3),
            cumulative_pnl_pct=round(float(sum(x.cumulative_pnl_pct for x in sent_scores) / len(sent_scores)), 2),
            sharpe_ratio=round(float(sum(x.sharpe_ratio for x in sent_scores) / len(sent_scores)), 2),
            total_eval_samples=sum(x.total_eval_samples for x in sent_scores),
        )

        return avg_base, avg_sent

    def _export_results(
        self,
        analysis_results: List[TickerAnalysisResult],
        base_card: ValidationScorecard,
        sent_card: ValidationScorecard,
    ) -> None:
        """Export predictions and validation summary to JSON and CSV."""
        export_records = []
        for r in analysis_results:
            export_records.append({
                "ticker": r.ticker,
                "asset_name": r.asset_name,
                "exchange": r.exchange,
                "currency": r.currency,
                "current_price": r.current_price,
                "predicted_price_n1": r.predicted_price_n1,
                "predicted_return_pct": r.predicted_return_pct,
                "sentiment_score": r.sentiment_score,
                "sentiment_label": r.sentiment_label,
                "social_mentions": r.social_mentions_count,
                "news_count": r.news_articles_count,
                "direction": r.direction,
                "confidence_pct": r.confidence_pct,
                "position": r.position,
                "position_size_pct": r.position_size_pct,
                "stop_loss": r.stop_loss,
                "take_profit": r.take_profit,
                "risk_reward_ratio": r.risk_reward_ratio,
                "nearest_support": r.support_resistance.nearest_support,
                "nearest_resistance": r.support_resistance.nearest_resistance,
                "divergence_signal": r.divergence_signal,
            })

        # Save CSV
        df = pd.DataFrame(export_records)
        csv_path = self.reports_dir / "predictions_summary.csv"
        df.to_csv(csv_path, index=False)

        # Save JSON
        json_path = self.reports_dir / "predictions_summary.json"
        payload = {
            "tickers": export_records,
            "validation_scorecard": {
                "baseline": asdict(base_card),
                "sentiment_enhanced": asdict(sent_card),
            },
        }
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

        logger.info(f"Summary datasets exported to {csv_path} and {json_path}")
