"""Schémas pour les features"""

from pydantic import BaseModel, Field
from typing import List, Optional, Dict
from datetime import datetime
from api.schemas.common import MetaInfo, ResponseBase


class FeatureRequest(BaseModel):
    """Requête pour calculer des features"""
    year: int = Field(..., description="Année (2022, 2023, 2024)")
    features: Optional[List[str]] = Field(
        None, 
        description="Liste des features à calculer (si None, toutes)"
    )


class CandleWithFeatures(BaseModel):
    """Bougie M15 avec features calculées"""
    timestamp_15m: datetime
    open_15m: float
    high_15m: float
    low_15m: float
    close_15m: float
    volume_15m: int
    
    # Features (optionnelles)
    return_1: Optional[float] = None
    return_4: Optional[float] = None
    ema_20: Optional[float] = None
    ema_50: Optional[float] = None
    ema_200: Optional[float] = None
    ema_diff: Optional[float] = None
    rsi_14: Optional[float] = None
    rolling_std_20: Optional[float] = None
    rolling_std_100: Optional[float] = None
    range_15m: Optional[float] = None
    body: Optional[float] = None
    upper_wick: Optional[float] = None
    lower_wick: Optional[float] = None
    distance_to_ema200: Optional[float] = None
    slope_ema50: Optional[float] = None
    atr_14: Optional[float] = None
    volatility_ratio: Optional[float] = None
    adx_14: Optional[float] = None
    macd: Optional[float] = None
    macd_signal: Optional[float] = None


class FeaturesResponse(ResponseBase):
    """Réponse avec features calculées"""
    year: int
    n_candles: int
    features_computed: List[str]
    data: List[CandleWithFeatures]


class FeatureInfoResponse(ResponseBase):
    """Informations sur les features disponibles"""
    available_features: Dict[str, List[str]]
    total_features: int