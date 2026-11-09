"""Support and Resistance level calculations: Pivot points, Fibonacci, and local extrema clustering."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd


@dataclass
class SupportResistanceLevels:
    current_price: float
    pivot: float
    support_1: float
    support_2: float
    support_3: float
    resistance_1: float
    resistance_2: float
    resistance_3: float
    nearest_support: float
    nearest_resistance: float
    distance_to_support_pct: float
    distance_to_resistance_pct: float
    cluster_levels: List[float]


class SupportResistanceCalculator:
    """Calculates multi-method Support and Resistance levels for equities."""

    @staticmethod
    def calculate_pivot_points(
        high: float,
        low: float,
        close: float,
        method: str = "classic",
    ) -> Dict[str, float]:
        """Compute Standard, Fibonacci, or Camarilla Pivot Points."""
        pivot = (high + low + close) / 3.0
        hl_range = high - low

        if method == "fibonacci":
            r1 = pivot + 0.382 * hl_range
            r2 = pivot + 0.618 * hl_range
            r3 = pivot + 1.000 * hl_range
            s1 = pivot - 0.382 * hl_range
            s2 = pivot - 0.618 * hl_range
            s3 = pivot - 1.000 * hl_range
        elif method == "camarilla":
            r4 = close + hl_range * 1.1 / 2.0
            r3 = close + hl_range * 1.1 / 4.0
            r2 = close + hl_range * 1.1 / 6.0
            r1 = close + hl_range * 1.1 / 12.0
            s1 = close - hl_range * 1.1 / 12.0
            s2 = close - hl_range * 1.1 / 6.0
            s3 = close - hl_range * 1.1 / 4.0
            s4 = close - hl_range * 1.1 / 2.0
            return {
                "pivot": pivot,
                "r1": r1, "r2": r2, "r3": r3, "r4": r4,
                "s1": s1, "s2": s2, "s3": s3, "s4": s4,
            }
        else:  # classic
            r1 = 2 * pivot - low
            s1 = 2 * pivot - high
            r2 = pivot + hl_range
            s2 = pivot - hl_range
            r3 = high + 2 * (pivot - low)
            s3 = low - 2 * (high - pivot)

        return {
            "pivot": pivot,
            "r1": r1,
            "r2": r2,
            "r3": r3,
            "s1": s1,
            "s2": s2,
            "s3": s3,
        }

    @classmethod
    def extract_levels_from_df(
        cls,
        df: pd.DataFrame,
        window: int = 20,
    ) -> SupportResistanceLevels:
        """Extract multi-tier support and resistance levels from recent bar history."""
        if df.empty:
            return SupportResistanceLevels(
                current_price=0.0, pivot=0.0,
                support_1=0.0, support_2=0.0, support_3=0.0,
                resistance_1=0.0, resistance_2=0.0, resistance_3=0.0,
                nearest_support=0.0, nearest_resistance=0.0,
                distance_to_support_pct=0.0, distance_to_resistance_pct=0.0,
                cluster_levels=[],
            )

        recent = df.tail(window)
        high = float(recent["High"].max())
        low = float(recent["Low"].min())
        close = float(df["Close"].iloc[-1])

        pivots = cls.calculate_pivot_points(high, low, close, method="classic")

        # Cluster local extrema (peaks and troughs)
        highs = recent["High"].values
        lows = recent["Low"].values
        extrema = np.unique(np.concatenate([highs, lows]))

        supports = [s for s in [pivots["s1"], pivots["s2"], pivots["s3"]] if s < close]
        resistances = [r for r in [pivots["r1"], pivots["r2"], pivots["r3"]] if r > close]

        nearest_s = max(supports) if supports else low
        nearest_r = min(resistances) if resistances else high

        dist_s_pct = round(((close - nearest_s) / close) * 100.0, 2)
        dist_r_pct = round(((nearest_r - close) / close) * 100.0, 2)

        return SupportResistanceLevels(
            current_price=round(close, 2),
            pivot=round(pivots["pivot"], 2),
            support_1=round(pivots["s1"], 2),
            support_2=round(pivots["s2"], 2),
            support_3=round(pivots["s3"], 2),
            resistance_1=round(pivots["r1"], 2),
            resistance_2=round(pivots["r2"], 2),
            resistance_3=round(pivots["r3"], 2),
            nearest_support=round(nearest_s, 2),
            nearest_resistance=round(nearest_r, 2),
            distance_to_support_pct=dist_s_pct,
            distance_to_resistance_pct=dist_r_pct,
            cluster_levels=[round(float(x), 2) for x in sorted(extrema)[::max(1, len(extrema) // 5)]],
        )
