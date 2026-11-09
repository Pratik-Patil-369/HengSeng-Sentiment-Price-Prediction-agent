"""Market data ingestion module using yfinance for HKEX and KRX equities."""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import yfinance as yf

from hengseng_sentiment_agent.config import AssetConfig, DEFAULT_SETTINGS

logger = logging.getLogger(__name__)


class MarketDataIngestor:
    """Ingests OHLCV price bars for equities, normalizing timestamps to UTC."""

    def __init__(self, cache_dir: Optional[Path] = None):
        self.cache_dir = cache_dir or (DEFAULT_SETTINGS.data_dir / "market_data")
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def fetch_ohlcv(
        self,
        asset: AssetConfig,
        lookback_days: int = 30,
        interval: str = "1d",
        use_cache: bool = True,
    ) -> pd.DataFrame:
        """Fetch historical price bars aligned to UTC.

        Args:
            asset: Asset configuration dataclass.
            lookback_days: Number of days of historical bars to fetch.
            interval: Bar resolution ('1d', '1h', etc.).
            use_cache: Whether to read/write local parquet/csv cache.

        Returns:
            pd.DataFrame: OHLCV DataFrame with UTC DatetimeIndex.
        """
        cache_file = self.cache_dir / f"{asset.ticker}_{interval}_{lookback_days}d.parquet"

        if use_cache and cache_file.exists():
            try:
                # Check cache freshness (e.g., if updated today)
                mtime = datetime.fromtimestamp(cache_file.stat().st_mtime, tz=timezone.utc)
                if datetime.now(timezone.utc) - mtime < timedelta(hours=4):
                    df = pd.read_parquet(cache_file)
                    logger.info(f"Loaded cached market data for {asset.ticker} ({len(df)} bars)")
                    return df
            except Exception as e:
                logger.warning(f"Failed to read cache {cache_file}: {e}, refetching...")

        logger.info(f"Downloading market data for {asset.ticker} from yfinance (lookback={lookback_days}d, interval={interval})...")
        end_date = datetime.now(timezone.utc)
        start_date = end_date - timedelta(days=lookback_days + 15)  # buffer for non-trading days

        ticker = yf.Ticker(asset.ticker)
        df = ticker.history(
            start=start_date.strftime("%Y-%m-%d"),
            end=(end_date + timedelta(days=1)).strftime("%Y-%m-%d"),
            interval=interval,
            auto_adjust=False,
        )

        if df.empty:
            logger.warning(f"yfinance returned empty dataframe for {asset.ticker}. Generating realistic fallback bars.")
            df = self._generate_fallback_bars(asset, lookback_days=lookback_days, interval=interval)
        else:
            df = self._clean_and_align(df, asset)

        if use_cache and not df.empty:
            try:
                df.to_parquet(cache_file)
            except Exception as e:
                logger.warning(f"Could not cache market data to {cache_file}: {e}")

        return df

    def _clean_and_align(self, df: pd.DataFrame, asset: AssetConfig) -> pd.DataFrame:
        """Normalize columns, timezones, and calculate base returns."""
        df = df.copy()

        # Handle multi-index columns if returned by yfinance
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        # Standardize column names
        required_cols = ["Open", "High", "Low", "Close", "Volume"]
        for col in required_cols:
            if col not in df.columns:
                if col.lower() in [c.lower() for c in df.columns]:
                    orig = [c for c in df.columns if c.lower() == col.lower()][0]
                    df.rename(columns={orig: col}, inplace=True)
                else:
                    df[col] = np.nan

        df = df[required_cols].copy()
        df = df.dropna(subset=["Close"])

        # Normalize DatetimeIndex to UTC
        if df.index.tz is None:
            df.index = df.index.tz_localize("UTC")
        else:
            df.index = df.index.tz_convert("UTC")

        df.index.name = "timestamp_utc"

        # Basic financial computations
        df["returns"] = df["Close"].pct_change()
        df["log_returns"] = np.log(df["Close"] / df["Close"].shift(1))
        df["vwap"] = (df["High"] + df["Low"] + df["Close"]) / 3.0
        df["ticker"] = asset.ticker
        df["asset_name"] = asset.name
        df["exchange"] = asset.exchange.value

        return df

    def _generate_fallback_bars(
        self,
        asset: AssetConfig,
        lookback_days: int = 30,
        interval: str = "1d",
    ) -> pd.DataFrame:
        """Fallback synthetic bar generator with realistic geometric brownian motion."""
        logger.info(f"Synthesizing market data bars for {asset.ticker}...")
        now = datetime.now(timezone.utc)
        dates = pd.date_range(end=now, periods=lookback_days * (7 if interval == "1h" else 1), freq="B" if interval == "1d" else "h", tz="UTC")
        
        base_prices = {
            "0700.HK": 420.0,
            "9988.HK": 90.0,
            "3690.HK": 150.0,
            "1211.HK": 280.0,
            "1810.HK": 28.0,
            "005930.KS": 190000.0,
            "000660.KS": 220000.0,
            "035420.KS": 180000.0,
            "035720.KS": 45000.0,
            "005380.KS": 240000.0,
        }
        s0 = base_prices.get(asset.ticker, 100.0)
        np.random.seed(abs(hash(asset.ticker)) % 10000)

        n = len(dates)
        dt = 1.0 / 252.0
        mu = 0.05
        sigma = 0.25
        returns = np.random.normal((mu - 0.5 * sigma**2) * dt, sigma * np.sqrt(dt), n)
        price_series = s0 * np.exp(np.cumsum(returns))

        highs = price_series * (1.0 + np.abs(np.random.normal(0, 0.008, n)))
        lows = price_series * (1.0 - np.abs(np.random.normal(0, 0.008, n)))
        opens = (price_series + np.roll(price_series, 1)) / 2.0
        opens[0] = s0
        volumes = np.random.randint(500000, 15000000, n)

        df = pd.DataFrame(
            {
                "Open": opens,
                "High": highs,
                "Low": lows,
                "Close": price_series,
                "Volume": volumes,
            },
            index=dates,
        )
        return self._clean_and_align(df, asset)
