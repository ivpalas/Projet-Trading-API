"""Schémas pour les stratégies de trading"""

from pydantic import BaseModel, Field
from typing import List, Optional, Dict
from datetime import datetime
from api.schemas.common import MetaInfo, ResponseBase


class StrategyRequest(BaseModel):
    """Requête pour backtester une stratégie"""
    strategy: str = Field(..., description="Nom de la stratégie (random, buy_hold, sma_crossover)")
    year: int = Field(..., description="Année (2022, 2023, 2024)")
    initial_capital: float = Field(10000.0, description="Capital initial")


class Trade(BaseModel):
    """Un trade exécuté"""
    timestamp: datetime
    action: str  # 'buy' ou 'sell'
    price: float
    quantity: float
    capital_after: float


class BacktestResult(BaseModel):
    """Résultat d'un backtest"""
    strategy: str
    year: int
    initial_capital: float
    final_capital: float
    total_return: float  # En %
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float  # En %
    trades: List[Trade]


class BacktestResponse(ResponseBase):
    """Réponse du backtest"""
    result: BacktestResult


class StrategyListResponse(ResponseBase):
    """Liste des stratégies disponibles"""
    strategies: List[Dict[str, str]]