"""Schémas pour les données"""

from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime
from api.schemas.common import MetaInfo, ResponseBase


class CandleM15(BaseModel):
    """Une bougie M15"""
    timestamp_15m: datetime
    open_15m: float
    high_15m: float
    low_15m: float
    close_15m: float
    volume_15m: int
    n_candles_m1: Optional[int] = None


class DataResponse(ResponseBase):
    """Réponse avec données"""
    data: List[CandleM15]
    total_candles: int
    year: int


class DataStatsResponse(ResponseBase):
    """Statistiques sur les données"""
    year: int
    n_candles: int
    period_start: datetime
    period_end: datetime
    price_mean: float
    price_std: float
    price_min: float
    price_max: float