"""Configuration for Hang Seng & KRX Sentiment Price Prediction Agent."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()


class Exchange(str, Enum):
    HKEX = "HKEX"  # Hong Kong Exchange (Hang Seng)
    KRX = "KRX"    # Korea Exchange


@dataclass
class AssetConfig:
    ticker: str
    symbol: str
    name: str
    local_name: str
    exchange: Exchange
    cashtags: List[str]
    keywords: List[str]
    sector: str
    currency: str


# 5 Selected Assets for HKEX (Hang Seng)
HKEX_ASSETS: Dict[str, AssetConfig] = {
    "0700.HK": AssetConfig(
        ticker="0700.HK",
        symbol="0700",
        name="Tencent Holdings",
        local_name="騰訊控股",
        exchange=Exchange.HKEX,
        cashtags=["#0700", "#Tencent", "$0700", "$TCEHY"],
        keywords=["騰訊", "腾讯", "Tencent", "0700", "企鵝"],
        sector="Technology / Gaming",
        currency="HKD",
    ),
    "9988.HK": AssetConfig(
        ticker="9988.HK",
        symbol="9988",
        name="Alibaba Group",
        local_name="阿里巴巴",
        exchange=Exchange.HKEX,
        cashtags=["#9988", "#Alibaba", "$9988", "$BABA"],
        keywords=["阿里巴巴", "阿里", "Alibaba", "9988", "淘寶"],
        sector="E-commerce / Cloud",
        currency="HKD",
    ),
    "3690.HK": AssetConfig(
        ticker="3690.HK",
        symbol="3690",
        name="Meituan",
        local_name="美團",
        exchange=Exchange.HKEX,
        cashtags=["#3690", "#Meituan", "$3690"],
        keywords=["美團", "美团", "Meituan", "3690", "外賣"],
        sector="Consumer Internet",
        currency="HKD",
    ),
    "1211.HK": AssetConfig(
        ticker="1211.HK",
        symbol="1211",
        name="BYD Company",
        local_name="比亞迪股份",
        exchange=Exchange.HKEX,
        cashtags=["#1211", "#BYD", "$1211", "$BYDDY"],
        keywords=["比亞迪", "比亚迪", "BYD", "1211", "電車"],
        sector="Automotive / EV",
        currency="HKD",
    ),
    "1810.HK": AssetConfig(
        ticker="1810.HK",
        symbol="1810",
        name="Xiaomi Corporation",
        local_name="小米集團",
        exchange=Exchange.HKEX,
        cashtags=["#1810", "#Xiaomi", "$1810", "$XIACY"],
        keywords=["小米", "Xiaomi", "1810", "雷軍", "SU7"],
        sector="Consumer Electronics / EV",
        currency="HKD",
    ),
}

# 5 Selected Assets for KRX (Korea)
KRX_ASSETS: Dict[str, AssetConfig] = {
    "005930.KS": AssetConfig(
        ticker="005930.KS",
        symbol="005930",
        name="Samsung Electronics",
        local_name="삼성전자",
        exchange=Exchange.KRX,
        cashtags=["#005930", "#Samsung", "$005930"],
        keywords=["삼성전자", "삼전", "Samsung", "005930", "반도체"],
        sector="Semiconductors / Hardware",
        currency="KRW",
    ),
    "000660.KS": AssetConfig(
        ticker="000660.KS",
        symbol="000660",
        name="SK Hynix",
        local_name="SK하이닉스",
        exchange=Exchange.KRX,
        cashtags=["#000660", "#SKHynix", "$000660"],
        keywords=["SK하이닉스", "하이닉스", "Hynix", "000660", "HBM"],
        sector="Semiconductors / Memory",
        currency="KRW",
    ),
    "035420.KS": AssetConfig(
        ticker="035420.KS",
        symbol="035420",
        name="Naver Corporation",
        local_name="네이버",
        exchange=Exchange.KRX,
        cashtags=["#035420", "#Naver", "$035420"],
        keywords=["네이버", "NAVER", "035420", "라인", "하이퍼클로바"],
        sector="Internet / AI",
        currency="KRW",
    ),
    "035720.KS": AssetConfig(
        ticker="035720.KS",
        symbol="035720",
        name="Kakao Corp",
        local_name="카카오",
        exchange=Exchange.KRX,
        cashtags=["#035720", "#Kakao", "$035720"],
        keywords=["카카오", "Kakao", "035720", "카카오톡", "카카오페이"],
        sector="Social / Fintech",
        currency="KRW",
    ),
    "005380.KS": AssetConfig(
        ticker="005380.KS",
        symbol="005380",
        name="Hyundai Motor",
        local_name="현대자동차",
        exchange=Exchange.KRX,
        cashtags=["#005380", "#Hyundai", "$005380"],
        keywords=["현대자동차", "현대차", "Hyundai", "005380", "아이오닉"],
        sector="Automotive",
        currency="KRW",
    ),
}

ALL_ASSETS: Dict[str, AssetConfig] = {**HKEX_ASSETS, **KRX_ASSETS}


@dataclass
class PipelineSettings:
    lookback_days: int = 10
    prediction_horizon: int = 1
    bar_interval: str = "1d"  # 1d or 1h
    
    # Social sources to use per exchange (at least 2 required)
    hkex_sources: List[str] = field(default_factory=lambda: ["lihkg", "futu_moomoo", "telegram"])
    krx_sources: List[str] = field(default_factory=lambda: ["naver_finance", "daum_finance", "blind"])
    
    # Engagement weights
    like_weight: float = 0.5
    repost_weight: float = 1.0
    view_weight: float = 0.1
    follower_weight: float = 0.2
    
    # Model parameters
    test_size_ratio: float = 0.25
    n_splits_cv: int = 5
    random_state: int = 42
    
    # API Keys
    apify_api_key: Optional[str] = field(
        default_factory=lambda: os.getenv("APIFY_API_KEY")
    )
    exa_api_key: Optional[str] = field(
        default_factory=lambda: os.getenv("EXA_API_KEY") or os.getenv("EXA_SEARCH_API_KEY")
    )
    
    # Paths
    project_root: Path = field(
        default_factory=lambda: Path(__file__).resolve().parent.parent.parent
    )
    data_dir: Path = field(init=False)
    reports_dir: Path = field(init=False)

    def __post_init__(self):
        self.data_dir = self.project_root / "data"
        self.reports_dir = self.project_root / "reports"
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.reports_dir.mkdir(parents=True, exist_ok=True)


DEFAULT_SETTINGS = PipelineSettings()
