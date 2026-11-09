"""Multilingual financial sentiment analyzer supporting Cantonese/Chinese, Korean, and English."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, Tuple


@dataclass
class SentimentResult:
    polarity: float       # -1.0 (very negative) to +1.0 (very positive)
    label: str           # "BULLISH", "BEARISH", "NEUTRAL"
    confidence: float    # 0.0 to 1.0
    subjectivity: float  # 0.0 to 1.0


class FinancialSentimentAnalyzer:
    """Specialized financial NLP analyzer with domain lexicons for HK & KR retail finance."""

    def __init__(self):
        # Traditional & Simplified Chinese Financial Lexicon
        self.cn_bullish = {
            "升": 0.6, "大升": 0.9, "爆升": 1.0, "突破": 0.8, "破頂": 0.9, "超預期": 0.85,
            "超预期": 0.85, "淨流入": 0.7, "净流入": 0.7, "牛掣": 0.8, "牛證": 0.7,
            "加注": 0.65, "掃貨": 0.8, "扫货": 0.8, "睇好": 0.65, "看好": 0.65,
            "黃金交叉": 0.85, "黄金交叉": 0.85, "反彈": 0.6, "反弹": 0.6, "穩": 0.4,
            "股王": 0.75, "買入": 0.8, "买入": 0.8, "抄底": 0.75, "加倉": 0.7,
            "利好": 0.8, "開車": 0.75, "衝": 0.7, "支持位穩": 0.8, "勝率高": 0.7,
        }
        self.cn_bearish = {
            "跌": -0.6, "大跌": -0.9, "暴跌": -1.0, "沽空": -0.85, "破底": -0.9,
            "不及預期": -0.85, "不及预期": -0.85, "淨流出": -0.7, "净流出": -0.7,
            "熊掣": -0.8, "熊證": -0.7, "減倉": -0.65, "减仓": -0.65, "拋售": -0.8,
            "抛售": -0.8, "睇淡": -0.65, "看空": -0.7, "死亡交叉": -0.85, "接飛刀": -0.85,
            "接飞刀": -0.85, "走壞": -0.7, "走坏": -0.7, "止蝕": -0.7, "止损": -0.7,
            "利空": -0.8, "套牢": -0.75, "恐慌": -0.85, "爛": -0.6, "莊家壓盤": -0.7,
        }

        # Korean Retail Financial Lexicon
        self.kr_bullish = {
            "상한가": 1.0, "떡상": 1.0, "가즈아": 0.85, "풀매수": 0.9, "호재": 0.8,
            "돌파": 0.8, "반등": 0.65, "서프라이즈": 0.85, "쌍끌이": 0.8, "순매수": 0.75,
            "신고가": 0.9, "저평가": 0.7, "지지선": 0.5, "상향": 0.65, "대박": 0.8,
            "급등": 0.85, "탈출": 0.5, "매수": 0.6, "상방": 0.75, "숏스퀴즈": 0.85,
        }
        self.kr_bearish = {
            "떡락": -1.0, "폭락": -1.0, "손절": -0.85, "악재": -0.85, "이탈": -0.75,
            "공매도": -0.85, "물타기": -0.6, "신저가": -0.9, "순매도": -0.75, "하향": -0.65,
            "급락": -0.85, "물림": -0.75, "매도": -0.6, "하방": -0.75, "개미털기": -0.6,
            "패닉셀": -0.9, "털림": -0.7, "조정": -0.4, "위험": -0.6, "실망": -0.7,
        }

        # English Financial Lexicon
        self.en_bullish = {
            "surge": 0.8, "rally": 0.75, "outperform": 0.8, "beat": 0.75, "record": 0.7,
            "bullish": 0.85, "buy": 0.7, "upgrade": 0.8, "gain": 0.6, "breakout": 0.75,
            "growth": 0.5, "soar": 0.85, "dividend": 0.5, "synergy": 0.5, "profit": 0.6,
        }
        self.en_bearish = {
            "plunge": -0.85, "crash": -1.0, "underperform": -0.8, "miss": -0.75, "bearish": -0.85,
            "sell": -0.7, "downgrade": -0.8, "loss": -0.65, "breakdown": -0.75, "decline": -0.6,
            "slump": -0.8, "drop": -0.6, "regulatory": -0.4, "scrutiny": -0.5, "headwind": -0.5,
        }

    def analyze_text(self, text: str) -> SentimentResult:
        """Calculate normalized sentiment score from multilingual financial text."""
        if not text or not text.strip():
            return SentimentResult(polarity=0.0, label="NEUTRAL", confidence=0.0, subjectivity=0.0)

        cleaned = text.strip()
        scores = []
        intensities = []

        # Check Chinese terms
        for word, val in self.cn_bullish.items():
            if word in cleaned:
                scores.append(val)
                intensities.append(abs(val))
        for word, val in self.cn_bearish.items():
            if word in cleaned:
                scores.append(val)
                intensities.append(abs(val))

        # Check Korean terms
        for word, val in self.kr_bullish.items():
            if word in cleaned:
                scores.append(val)
                intensities.append(abs(val))
        for word, val in self.kr_bearish.items():
            if word in cleaned:
                scores.append(val)
                intensities.append(abs(val))

        # Check English terms
        lower_en = cleaned.lower()
        for word, val in self.en_bullish.items():
            if re.search(r"\b" + re.escape(word) + r"\b", lower_en):
                scores.append(val)
                intensities.append(abs(val))
        for word, val in self.en_bearish.items():
            if re.search(r"\b" + re.escape(word) + r"\b", lower_en):
                scores.append(val)
                intensities.append(abs(val))

        if not scores:
            return SentimentResult(polarity=0.0, label="NEUTRAL", confidence=0.1, subjectivity=0.1)

        # Compute bounded average polarity [-1.0, 1.0]
        avg_polarity = sum(scores) / len(scores)
        # Apply gentle tanh bounding
        polarity = max(-1.0, min(1.0, avg_polarity))

        confidence = min(1.0, len(scores) * 0.2 + sum(intensities) / (len(intensities) * 2.0))
        subjectivity = min(1.0, len(scores) * 0.25)

        if polarity > 0.15:
            label = "BULLISH"
        elif polarity < -0.15:
            label = "BEARISH"
        else:
            label = "NEUTRAL"

        return SentimentResult(
            polarity=round(polarity, 4),
            label=label,
            confidence=round(confidence, 4),
            subjectivity=round(subjectivity, 4),
        )
