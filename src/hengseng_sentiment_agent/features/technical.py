"""Technical indicators module: RSI, MACD, Bollinger Bands, ATR, Volatility."""

from __future__ import annotations

import numpy as np
import pandas as pd


class TechnicalFeatureExtractor:
    """Calculates standardized technical indicator features from price bars."""

    @staticmethod
    def add_technical_features(df: pd.DataFrame) -> pd.DataFrame:
        """Add standard quantitative technical features to bar DataFrame."""
        if df.empty or len(df) < 5:
            return df.copy()

        data = df.copy()
        close = data["Close"]
        high = data["High"]
        low = data["Low"]

        # 1. RSI (Relative Strength Index)
        delta = close.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14, min_periods=3).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14, min_periods=3).mean()
        rs = gain / (loss + 1e-9)
        data["rsi_14"] = 100 - (100 / (1 + rs))
        data["rsi_14"] = data["rsi_14"].fillna(50.0)

        # 2. MACD (12, 26, 9)
        ema_12 = close.ewm(span=12, adjust=False).mean()
        ema_26 = close.ewm(span=26, adjust=False).mean()
        data["macd"] = ema_12 - ema_26
        data["macd_signal"] = data["macd"].ewm(span=9, adjust=False).mean()
        data["macd_hist"] = data["macd"] - data["macd_signal"]

        # 3. Bollinger Bands (20, 2)
        sma_20 = close.rolling(window=20, min_periods=3).mean()
        std_20 = close.rolling(window=20, min_periods=3).std().fillna(0.0)
        data["bb_upper"] = sma_20 + 2 * std_20
        data["bb_lower"] = sma_20 - 2 * std_20
        bb_range = data["bb_upper"] - data["bb_lower"]
        data["bb_pct_b"] = np.where(bb_range > 0, (close - data["bb_lower"]) / bb_range, 0.5)

        # 4. ATR (Average True Range)
        prev_close = close.shift(1)
        tr1 = high - low
        tr2 = (high - prev_close).abs()
        tr3 = (low - prev_close).abs()
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        data["atr_14"] = true_range.rolling(window=14, min_periods=3).mean().bfill()
        data["atr_pct"] = data["atr_14"] / close

        # 5. Moving Average Ratios
        sma_5 = close.rolling(window=5, min_periods=1).mean()
        data["ratio_close_sma5"] = close / sma_5
        data["ratio_close_sma20"] = close / sma_20.bfill()

        # 6. Realized Volatility (rolling annualized)
        log_ret = np.log(close / close.shift(1)).fillna(0.0)
        data["realized_vol_10d"] = log_ret.rolling(window=10, min_periods=3).std().fillna(0.0) * np.sqrt(252)

        return data
