"""
Router API pour les prédictions (T10)
Endpoints de prédiction ML et RL
"""

from fastapi import APIRouter, HTTPException, status
from fastapi.responses import JSONResponse
import numpy as np
import pandas as pd
from datetime import datetime
import time
from typing import Dict

from api.predictions_schemas import (
    PredictionRequest, 
    PredictionResponse,
    BatchPredictionRequest,
    BatchPredictionResponse,
    ModelInfo,
    ModelsListResponse,
    HealthCheck,
    ErrorResponse,
    ModelType,
    PredictionClass,
    SignalType
)
from api.services.model_loader import get_model_loader  # ← Et ici

router = APIRouter()


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def convert_prediction_to_class(prediction: int) -> PredictionClass:
    """Convertir prédiction numérique en classe"""
    mapping = {
        -1: PredictionClass.DOWN,
        0: PredictionClass.HOLD,
        1: PredictionClass.UP
    }
    return mapping.get(prediction, PredictionClass.HOLD)


def convert_class_to_signal(pred_class: PredictionClass) -> SignalType:
    """Convertir classe de prédiction en signal de trading"""
    mapping = {
        PredictionClass.UP: SignalType.BUY,
        PredictionClass.DOWN: SignalType.SELL,
        PredictionClass.HOLD: SignalType.HOLD
    }
    return mapping[pred_class]


def prepare_features_ml(
    features: Dict[str, float],
    feature_names: list,
    scaler
) -> np.ndarray:
    """
    Préparer les features pour un modèle ML
    
    Args:
        features: Dictionnaire de features
        feature_names: Noms des features attendues
        scaler: Scaler pour normalisation
        
    Returns:
        Array numpy des features normalisées
    """
    # Créer DataFrame avec les features dans le bon ordre
    feature_values = []
    missing_features = []
    
    for feature_name in feature_names:
        if feature_name in features:
            feature_values.append(features[feature_name])
        else:
            missing_features.append(feature_name)
            feature_values.append(0.0)  # Valeur par défaut
    
    if missing_features:
        print(f"⚠️ Features manquantes: {missing_features[:5]}...")
    
    # Convertir en array et normaliser
    X = np.array([feature_values])
    X_scaled = scaler.transform(X)
    
    return X_scaled


# ============================================================================
# ENDPOINTS
# ============================================================================

@router.post(
    "/predict",
    response_model=PredictionResponse,
    summary="Prédiction unique",
    description="Obtenir une prédiction de trading pour un ensemble de features"
)
async def predict(request: PredictionRequest):
    """
    Endpoint de prédiction unique
    
    Returns:
        PredictionResponse avec la prédiction et les probabilités
    """
    start_time = time.time()
    
    try:
        # Récupérer le loader
        loader = get_model_loader()
        
        # Vérifier que le modèle est chargé
        if not loader.is_loaded(request.model_type.value):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Modèle {request.model_type.value} non chargé. Modèles disponibles: {loader.get_loaded_models()}"
            )
        
        # Récupérer le modèle
        model = loader.get_model(request.model_type.value)
        
        # Prédiction selon le type de modèle
        if request.model_type == ModelType.DQN_AGENT:
            # RL Agent
            state = np.array(list(request.features.values()), dtype=np.float32)
            action = model.select_action(state, training=False)
            
            # Convertir action en prédiction
            prediction_num = action - 1  # 0,1,2 -> -1,0,1
            prediction_class = convert_prediction_to_class(prediction_num)
            
            # Pas de probabilités pour RL (utiliser Q-values si besoin)
            probabilities = {
                "DOWN": 0.33,
                "HOLD": 0.34,
                "UP": 0.33
            }
            confidence = 0.5
            
        else:
            # ML Models
            scaler = loader.get_scaler(request.model_type.value)
            feature_names = loader.get_feature_names(request.model_type.value)
            
            if not scaler or not feature_names:
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=f"Scaler ou feature_names manquant pour {request.model_type.value}"
                )
            
            # Préparer features
            X_scaled = prepare_features_ml(request.features, feature_names, scaler)
            
            # Prédiction
            prediction_num = model.predict(X_scaled)[0]
            prediction_class = convert_prediction_to_class(prediction_num)
            
            # Probabilités
            if hasattr(model, 'predict_proba'):
                proba = model.predict_proba(X_scaled)[0]
                probabilities = {
                    "DOWN": float(proba[0]),
                    "HOLD": float(proba[1]),
                    "UP": float(proba[2])
                }
                confidence = float(max(proba))
            else:
                probabilities = {
                    "DOWN": 0.33,
                    "HOLD": 0.34,
                    "UP": 0.33
                }
                confidence = 0.5
        
        # Signal de trading
        signal = convert_class_to_signal(prediction_class)
        
        # Temps de traitement
        processing_time = (time.time() - start_time) * 1000
        
        # Metadata du modèle
        model_info = loader.get_model_info(request.model_type.value)
        model_version = model_info.get('version', 'v1.0') if model_info else 'v1.0'
        
        return PredictionResponse(
            prediction=prediction_class,
            probabilities=probabilities,
            signal=signal,
            confidence=confidence,
            model_type=request.model_type,
            model_version=model_version,
            timestamp=request.timestamp or datetime.now(),
            processing_time_ms=processing_time
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Erreur lors de la prédiction: {str(e)}"
        )


@router.post(
    "/predict/batch",
    response_model=BatchPredictionResponse,
    summary="Prédictions batch",
    description="Obtenir plusieurs prédictions en une seule requête (max 1000)"
)
async def predict_batch(request: BatchPredictionRequest):
    """
    Endpoint de prédiction batch
    
    Returns:
        BatchPredictionResponse avec la liste des prédictions
    """
    start_time = time.time()
    
    predictions = []
    
    for features_dict in request.data:
        # Créer une requête individuelle
        single_request = PredictionRequest(
            model_type=request.model_type,
            features=features_dict
        )
        
        # Faire la prédiction
        pred_response = await predict(single_request)
        predictions.append(pred_response)
    
    total_time = (time.time() - start_time) * 1000
    
    return BatchPredictionResponse(
        predictions=predictions,
        total_predictions=len(predictions),
        model_type=request.model_type,
        total_processing_time_ms=total_time
    )


@router.get(
    "/models",
    response_model=ModelsListResponse,
    summary="Liste des modèles",
    description="Obtenir la liste de tous les modèles chargés"
)
async def list_models():
    """
    Liste tous les modèles disponibles
    
    Returns:
        ModelsListResponse avec les infos de tous les modèles
    """
    loader = get_model_loader()
    
    models_info = []
    for model_type in loader.get_loaded_models():
        info = loader.get_model_info(model_type)
        
        if info:
            model_info = ModelInfo(
                model_type=ModelType(model_type),
                version=info.get('version', 'v1.0'),
                description=f"{model_type.replace('_', ' ').title()} model",
                training_date=None,  # TODO: ajouter dans metadata
                metrics=info.get('metrics', {}),
                features_required=info.get('features_required', []),
                is_loaded=True
            )
            models_info.append(model_info)
    
    return ModelsListResponse(
        models=models_info,
        total_models=len(models_info),
        loaded_models=len(models_info)
    )


@router.get(
    "/models/{model_type}",
    response_model=ModelInfo,
    summary="Info d'un modèle",
    description="Obtenir les informations détaillées d'un modèle spécifique"
)
async def get_model_info_endpoint(model_type: ModelType):
    """
    Récupérer les infos d'un modèle spécifique
    
    Args:
        model_type: Type de modèle
        
    Returns:
        ModelInfo avec les détails du modèle
    """
    loader = get_model_loader()
    
    if not loader.is_loaded(model_type.value):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Modèle {model_type.value} non trouvé. Modèles disponibles: {loader.get_loaded_models()}"
        )
    
    info = loader.get_model_info(model_type.value)
    
    return ModelInfo(
        model_type=model_type,
        version=info.get('version', 'v1.0'),
        description=f"{model_type.value.replace('_', ' ').title()} model",
        training_date=None,
        metrics=info.get('metrics', {}),
        features_required=info.get('features_required', []),
        is_loaded=True
    )


@router.get(
    "/health",
    response_model=HealthCheck,
    summary="Health check",
    description="Vérifier l'état de santé de l'API et des modèles"
)
async def health_check():
    """
    Health check de l'API
    
    Returns:
        HealthCheck avec le statut du système
    """
    loader = get_model_loader()
    health = loader.get_health_status()
    
    return HealthCheck(
        status=health['status'],
        timestamp=datetime.now(),
        version="1.0.0",
        models_loaded=health['models_loaded'],
        uptime_seconds=health['uptime_seconds'],
        details={
            model: "loaded" 
            for model in health['loaded_models']
        }
    )
