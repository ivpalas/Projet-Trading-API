"""Router pour les stratégies de trading"""

from fastapi import APIRouter, HTTPException
from api.schemas.strategies import (
    BacktestResponse,
    StrategyListResponse,
    StrategyRequest
)
from api.schemas.common import MetaInfo
from api.services.strategy_service import StrategyService
from datetime import datetime

router = APIRouter(prefix="/strategies", tags=["Strategies"])
strategy_service = StrategyService()


@router.get("/available", response_model=StrategyListResponse)
def get_available_strategies():
    """Liste des stratégies disponibles"""
    
    strategies = strategy_service.get_available_strategies()
    
    return StrategyListResponse(
        meta=MetaInfo(timestamp=datetime.now()),
        strategies=strategies
    )


@router.post("/backtest", response_model=BacktestResponse)
def backtest_strategy(request: StrategyRequest):
    """Backteste une stratégie"""
    
    try:
        result = strategy_service.backtest(
            strategy=request.strategy,
            year=request.year,
            initial_capital=request.initial_capital
        )
        
        return BacktestResponse(
            meta=MetaInfo(year=request.year, timestamp=datetime.now()),
            result=result
        )
    
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))