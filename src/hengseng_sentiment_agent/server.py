"""FastAPI Web Server serving both the Frontend Dashboard and Backend REST API."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel

from hengseng_sentiment_agent.config import ALL_ASSETS, DEFAULT_SETTINGS
from hengseng_sentiment_agent.pipeline import SentimentPredictionPipeline

app = FastAPI(
    title="Hang Seng & KRX Sentiment Price Prediction Agent",
    description="Real-Time Multi-Source Alternative Data & Price Prediction API",
    version="1.0.0",
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

SUMMARY_JSON_PATH = DEFAULT_SETTINGS.reports_dir / "predictions_summary.json"
REPORT_HTML_PATH = DEFAULT_SETTINGS.reports_dir / "sentiment_price_prediction_report.html"


def load_cached_data() -> Dict[str, Any]:
    """Helper to load latest cached predictions and validation metrics."""
    if not SUMMARY_JSON_PATH.exists():
        # Run pipeline once if cache is missing
        pipeline = SentimentPredictionPipeline()
        pipeline.run_pipeline(exchange="ALL", use_cache=True, generate_html=True)

    if SUMMARY_JSON_PATH.exists():
        with open(SUMMARY_JSON_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"tickers": [], "validation_scorecard": {}}


# -------------------------------------------------------------
# FRONTEND ROUTE
# -------------------------------------------------------------
@app.get("/", response_class=HTMLResponse, tags=["Frontend"])
def get_frontend_dashboard():
    """Serves the interactive Visual Dashboard UI."""
    if REPORT_HTML_PATH.exists():
        return FileResponse(REPORT_HTML_PATH, media_type="text/html")
    # Generate report if missing
    pipeline = SentimentPredictionPipeline()
    pipeline.run_pipeline(exchange="ALL", use_cache=True, generate_html=True)
    return FileResponse(REPORT_HTML_PATH, media_type="text/html")


# -------------------------------------------------------------
# BACKEND API ROUTES
# -------------------------------------------------------------
@app.get("/api/health", tags=["System"])
def health_check():
    """System health check and universe status."""
    return {
        "status": "healthy",
        "service": "hengseng-sentiment-prediction-agent",
        "universe_count": len(ALL_ASSETS),
        "exchanges": ["HKEX (Hang Seng)", "KRX (South Korea)"],
        "data_sources": ["Apify (LIHKG, Futu, Naver, Telegram)", "Exa.ai News", "yfinance OHLCV"],
    }


@app.get("/api/tickers", tags=["Predictions"])
def get_all_tickers(
    exchange: Optional[str] = Query(None, description="Filter by HKEX or KRX"),
    sort_by: Optional[str] = Query("social_mentions", description="Sort field: social_mentions, sentiment_score, predicted_return_pct"),
    order: Optional[str] = Query("desc", description="asc or desc"),
):
    """Retrieve predictions, sentiment, direction, and technical levels for all monitored assets."""
    data = load_cached_data()
    tickers = data.get("tickers", [])

    if exchange:
        ex_upper = exchange.upper()
        tickers = [t for t in tickers if t.get("exchange") == ex_upper]

    reverse = (order.lower() == "desc")
    if sort_by:
        tickers = sorted(tickers, key=lambda x: x.get(sort_by, 0), reverse=reverse)

    return {
        "count": len(tickers),
        "filter": exchange or "ALL",
        "sort_by": sort_by,
        "results": tickers,
    }


@app.get("/api/tickers/{ticker}", tags=["Predictions"])
def get_ticker_detail(ticker: str):
    """Retrieve detailed prediction and quantitative analysis for a specific ticker."""
    t_upper = ticker.upper()
    data = load_cached_data()
    for item in data.get("tickers", []):
        if item.get("ticker", "").upper() == t_upper:
            return item

    raise HTTPException(status_code=404, detail=f"Ticker '{ticker}' not found. Available: {list(ALL_ASSETS.keys())}")


@app.get("/api/scorecard", tags=["Validation"])
def get_validation_scorecard():
    """Retrieve out-of-sample walk-forward validation scorecard comparing baseline vs sentiment models."""
    data = load_cached_data()
    return data.get("validation_scorecard", {})


class RunPipelineRequest(BaseModel):
    exchange: Optional[str] = "ALL"
    bypass_cache: Optional[bool] = False


@app.post("/api/pipeline/run", tags=["Pipeline"])
def trigger_pipeline_run(req: RunPipelineRequest):
    """Trigger an on-demand re-run of the ingestion, prediction, and validation pipeline."""
    pipeline = SentimentPredictionPipeline()
    results, base_card, sent_card = pipeline.run_pipeline(
        exchange=req.exchange or "ALL",
        use_cache=not req.bypass_cache,
        generate_html=True,
    )
    return {
        "status": "success",
        "processed_assets": len(results),
        "exchange": req.exchange,
        "sentiment_hit_rate": f"{sent_card.directional_accuracy_pct}%",
        "baseline_hit_rate": f"{base_card.directional_accuracy_pct}%",
        "alpha_gain": f"+{(sent_card.directional_accuracy_pct - base_card.directional_accuracy_pct):.1f}%",
    }
