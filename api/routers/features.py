"""Router pour les features"""

from fastapi import APIRouter, HTTPException, Query
from typing import Optional, List
import pandas as pd
import numpy as np
from api.schemas.features import (
    FeaturesResponse,
    FeatureInfoResponse
)
from api.schemas.common import MetaInfo
from api.services.feature_service import FeatureService
from datetime import datetime

router = APIRouter(prefix="/features", tags=["Features"])
feature_service = FeatureService()


@router.get("/available", response_model=FeatureInfoResponse)
def get_available_features():
    """Liste des features disponibles"""
    
    info = feature_service.get_available_features()
    
    return FeatureInfoResponse(
        meta=MetaInfo(timestamp=datetime.now()),
        **info
    )


@router.post("/compute/{year}", response_model=FeaturesResponse)
def compute_features(
    year: int,
    features: Optional[List[str]] = Query(None, description="Features à calculer"),
    limit: int = Query(100, description="Nombre de lignes à retourner")
):
    """Calcule les features pour une année"""
    
    try:
        # Si features est None ou liste vide, calculer toutes les features
        if not features:
            features = None
        
        df = feature_service.compute_features(year, features)
        
        # NETTOYER LES VALEURS INVALIDES (NaN, Inf)
        df = df.replace([np.inf, -np.inf], np.nan)
        
        # Pagination
        df_page = df.head(limit)
        
        # Features calculées
        base_cols = ['timestamp_15m', 'open_15m', 'high_15m', 'low_15m', 
                     'close_15m', 'volume_15m', 'n_candles_m1']
        computed = [col for col in df.columns if col not in base_cols]
        
        # Convertir en dict avec gestion NaN
        records = []
        for _, row in df_page.iterrows():
            record = {}
            for col in df_page.columns:
                val = row[col]
                # Convertir NaN en None pour JSON
                if pd.isna(val):
                    record[col] = None
                else:
                    record[col] = val
            records.append(record)
        
        return FeaturesResponse(
            meta=MetaInfo(year=year, timestamp=datetime.now()),
            year=year,
            n_candles=len(df),
            features_computed=computed,
            data=records
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))