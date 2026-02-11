"""Router pour les données"""

from fastapi import APIRouter, HTTPException
from api.schemas.data import DataResponse, DataStatsResponse
from api.schemas.common import MetaInfo
from api.services.data_service import DataService
from datetime import datetime

router = APIRouter(prefix="/data", tags=["Données M15"])
data_service = DataService()


@router.get("/m15/{year}", response_model=DataResponse)
def get_m15_data(year: int, limit: int = 100, offset: int = 0):
    """Récupère les données M15 d'une année"""
    
    try:
        result = data_service.get_candles(year, limit, offset)
        
        return DataResponse(
            meta=MetaInfo(year=year, timestamp=datetime.now()),
            data=result['candles'],
            total_candles=result['total'],
            year=year
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/stats/{year}", response_model=DataStatsResponse)
def get_m15_stats(year: int):
    """Statistiques sur les données M15"""
    
    try:
        stats = data_service.get_stats(year)
        return DataStatsResponse(
            meta=MetaInfo(year=year, timestamp=datetime.now()),
            **stats
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/years")
def get_available_years():
    """Liste des années disponibles"""
    return data_service.get_available_years()