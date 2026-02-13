"""
Schémas Pydantic pour l'API de prédiction
Contrats API pour T10
"""

from pydantic import BaseModel, Field, validator
from typing import List, Optional, Dict, Any
from datetime import datetime
from enum import Enum


# ============================================================================
# ENUMS
# ============================================================================

class ModelType(str, Enum):
    """Types de modèles disponibles"""
    LOGISTIC_REGRESSION = "logistic_regression"
    RANDOM_FOREST = "random_forest"
    XGBOOST = "xgboost"
    DQN_AGENT = "dqn_agent"


class PredictionClass(str, Enum):
    """Classes de prédiction"""
    DOWN = "DOWN"
    HOLD = "HOLD"
    UP = "UP"


class SignalType(str, Enum):
    """Types de signaux de trading"""
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"


# ============================================================================
# REQUEST SCHEMAS
# ============================================================================

class PredictionRequest(BaseModel):
    """Requête de prédiction"""
    
    model_type: ModelType = Field(
        ..., 
        description="Type de modèle à utiliser"
    )
    
    features: Dict[str, float] = Field(
        ..., 
        description="Features pour la prédiction (dict feature_name -> value)"
    )
    
    timestamp: Optional[datetime] = Field(
        None,
        description="Timestamp de la prédiction (optionnel)"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "model_type": "logistic_regression",
                "features": {
                    "close_15m": 1.2500,
                    "rsi_14": 55.5,
                    "macd": 0.0012,
                    "ema_20": 1.2495
                },
                "timestamp": "2024-01-15T10:30:00"
            }
        }


class BatchPredictionRequest(BaseModel):
    """Requête de prédiction batch"""
    
    model_type: ModelType = Field(
        ..., 
        description="Type de modèle à utiliser"
    )
    
    data: List[Dict[str, float]] = Field(
        ..., 
        description="Liste de features pour prédictions multiples"
    )
    
    @validator('data')
    def validate_batch_size(cls, v):
        if len(v) > 1000:
            raise ValueError("Maximum 1000 prédictions par batch")
        if len(v) == 0:
            raise ValueError("Au moins 1 prédiction requise")
        return v
    
    class Config:
        json_schema_extra = {
            "example": {
                "model_type": "xgboost",
                "data": [
                    {"close_15m": 1.2500, "rsi_14": 55.5},
                    {"close_15m": 1.2505, "rsi_14": 56.2}
                ]
            }
        }


# ============================================================================
# RESPONSE SCHEMAS
# ============================================================================

class PredictionResponse(BaseModel):
    """Réponse de prédiction"""
    
    prediction: PredictionClass = Field(
        ..., 
        description="Classe prédite (UP, DOWN, HOLD)"
    )
    
    probabilities: Dict[str, float] = Field(
        ..., 
        description="Probabilités pour chaque classe"
    )
    
    signal: SignalType = Field(
        ..., 
        description="Signal de trading recommandé"
    )
    
    confidence: float = Field(
        ..., 
        ge=0.0, 
        le=1.0,
        description="Confiance de la prédiction (max probability)"
    )
    
    model_type: ModelType = Field(
        ..., 
        description="Modèle utilisé"
    )
    
    model_version: str = Field(
        ..., 
        description="Version du modèle"
    )
    
    timestamp: datetime = Field(
        default_factory=datetime.now,
        description="Timestamp de la prédiction"
    )
    
    processing_time_ms: float = Field(
        ..., 
        description="Temps de traitement en millisecondes"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "prediction": "UP",
                "probabilities": {
                    "DOWN": 0.15,
                    "HOLD": 0.25,
                    "UP": 0.60
                },
                "signal": "BUY",
                "confidence": 0.60,
                "model_type": "logistic_regression",
                "model_version": "v1.0",
                "timestamp": "2024-01-15T10:30:00.123",
                "processing_time_ms": 12.5
            }
        }


class BatchPredictionResponse(BaseModel):
    """Réponse de prédiction batch"""
    
    predictions: List[PredictionResponse] = Field(
        ..., 
        description="Liste des prédictions"
    )
    
    total_predictions: int = Field(
        ..., 
        description="Nombre total de prédictions"
    )
    
    model_type: ModelType = Field(
        ..., 
        description="Modèle utilisé"
    )
    
    total_processing_time_ms: float = Field(
        ..., 
        description="Temps total de traitement"
    )


# ============================================================================
# MODEL INFO SCHEMAS
# ============================================================================

class ModelInfo(BaseModel):
    """Informations sur un modèle"""
    
    model_type: ModelType = Field(
        ..., 
        description="Type de modèle"
    )
    
    version: str = Field(
        ..., 
        description="Version du modèle"
    )
    
    description: str = Field(
        ..., 
        description="Description du modèle"
    )
    
    training_date: Optional[datetime] = Field(
        None,
        description="Date d'entraînement"
    )
    
    metrics: Dict[str, float] = Field(
        default_factory=dict,
        description="Métriques du modèle (accuracy, f1, etc.)"
    )
    
    features_required: List[str] = Field(
        ..., 
        description="Liste des features requises"
    )
    
    is_loaded: bool = Field(
        ..., 
        description="Le modèle est-il chargé en mémoire ?"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "model_type": "logistic_regression",
                "version": "v1.0",
                "description": "Logistic Regression trained on 2022-2023",
                "training_date": "2024-01-10T12:00:00",
                "metrics": {
                    "accuracy": 0.8978,
                    "f1_macro": 0.3478,
                    "precision": 0.5061
                },
                "features_required": ["close_15m", "rsi_14", "macd"],
                "is_loaded": True
            }
        }


class ModelsListResponse(BaseModel):
    """Liste des modèles disponibles"""
    
    models: List[ModelInfo] = Field(
        ..., 
        description="Liste des modèles"
    )
    
    total_models: int = Field(
        ..., 
        description="Nombre total de modèles"
    )
    
    loaded_models: int = Field(
        ..., 
        description="Nombre de modèles chargés en mémoire"
    )


# ============================================================================
# HEALTH CHECK SCHEMAS
# ============================================================================

class HealthCheck(BaseModel):
    """Health check response"""
    
    status: str = Field(
        ..., 
        description="Statut du service (healthy, degraded, unhealthy)"
    )
    
    timestamp: datetime = Field(
        default_factory=datetime.now,
        description="Timestamp du check"
    )
    
    version: str = Field(
        ..., 
        description="Version de l'API"
    )
    
    models_loaded: int = Field(
        ..., 
        description="Nombre de modèles chargés"
    )
    
    uptime_seconds: float = Field(
        ..., 
        description="Temps de fonctionnement en secondes"
    )
    
    details: Optional[Dict[str, Any]] = Field(
        None,
        description="Détails additionnels"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "status": "healthy",
                "timestamp": "2024-01-15T10:30:00",
                "version": "1.0.0",
                "models_loaded": 4,
                "uptime_seconds": 3600.5,
                "details": {
                    "logistic_regression": "loaded",
                    "random_forest": "loaded",
                    "xgboost": "loaded",
                    "dqn_agent": "loaded"
                }
            }
        }


# ============================================================================
# ERROR SCHEMAS
# ============================================================================

class ErrorResponse(BaseModel):
    """Réponse d'erreur standardisée"""
    
    error: str = Field(
        ..., 
        description="Type d'erreur"
    )
    
    message: str = Field(
        ..., 
        description="Message d'erreur détaillé"
    )
    
    timestamp: datetime = Field(
        default_factory=datetime.now,
        description="Timestamp de l'erreur"
    )
    
    details: Optional[Dict[str, Any]] = Field(
        None,
        description="Détails additionnels de l'erreur"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "error": "ModelNotLoaded",
                "message": "Le modèle xgboost n'est pas chargé en mémoire",
                "timestamp": "2024-01-15T10:30:00",
                "details": {
                    "model_type": "xgboost",
                    "available_models": ["logistic_regression", "random_forest"]
                }
            }
        }
