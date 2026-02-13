"""
Router API pour le versioning des modèles (T11)
Endpoints pour gérer les versions de modèles
"""

from fastapi import APIRouter, HTTPException, status
from typing import List, Optional
from pydantic import BaseModel, Field
from datetime import datetime

from api.services.model_registry import get_registry


router = APIRouter()


# ============================================================================
# SCHEMAS
# ============================================================================

class ModelVersionInfo(BaseModel):
    """Informations sur une version de modèle"""
    version: str
    file_path: str
    metrics: dict
    description: str
    author: str
    registered_at: str
    status: str
    is_latest: bool = False
    is_production: bool = False


class RegisterModelRequest(BaseModel):
    """Requête pour enregistrer un modèle"""
    model_type: str = Field(..., description="Type de modèle")
    version: str = Field(..., description="Version (ex: v1.0)")
    file_path: str = Field(..., description="Chemin vers le fichier")
    metrics: Optional[dict] = Field(None, description="Métriques du modèle")
    description: str = Field("", description="Description")
    author: str = Field("system", description="Auteur")
    
    class Config:
        json_schema_extra = {
            "example": {
                "model_type": "logistic_regression",
                "version": "v1.0",
                "file_path": "models/saved/logistic_regression_v1.pkl",
                "metrics": {"accuracy": 0.89, "f1": 0.34},
                "description": "Premier modèle LogReg",
                "author": "Ivin"
            }
        }


class SetProductionRequest(BaseModel):
    """Requête pour définir une version en production"""
    version: str = Field(..., description="Version à mettre en production")


class ModelSummary(BaseModel):
    """Résumé d'un modèle"""
    model_type: str
    total_versions: int
    latest_version: Optional[str]
    production_version: Optional[str]
    versions: List[str]


# ============================================================================
# ENDPOINTS
# ============================================================================

@router.post(
    "/register",
    summary="Enregistrer une version de modèle",
    description="Enregistre une nouvelle version d'un modèle dans le registry"
)
async def register_model(request: RegisterModelRequest):
    """Enregistrer un modèle"""
    registry = get_registry()
    
    success = registry.register_model(
        model_type=request.model_type,
        version=request.version,
        file_path=request.file_path,
        metrics=request.metrics,
        description=request.description,
        author=request.author
    )
    
    if not success:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Version {request.version} existe déjà pour {request.model_type}"
        )
    
    return {
        "message": f"Modèle {request.model_type} v{request.version} enregistré",
        "model_type": request.model_type,
        "version": request.version,
        "is_production": registry.get_production_version(request.model_type) == request.version
    }


@router.get(
    "/models",
    response_model=List[ModelSummary],
    summary="Lister tous les modèles",
    description="Liste tous les modèles enregistrés dans le registry"
)
async def list_all_models():
    """Liste tous les modèles"""
    registry = get_registry()
    models = registry.list_all_models()
    return models


@router.get(
    "/models/{model_type}/versions",
    response_model=List[ModelVersionInfo],
    summary="Lister les versions d'un modèle",
    description="Liste toutes les versions d'un modèle spécifique"
)
async def list_model_versions(model_type: str):
    """Liste les versions d'un modèle"""
    registry = get_registry()
    versions = registry.list_versions(model_type)
    
    if not versions:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Modèle {model_type} non trouvé dans le registry"
        )
    
    return versions


@router.get(
    "/models/{model_type}/versions/{version}",
    response_model=ModelVersionInfo,
    summary="Info d'une version spécifique",
    description="Récupère les infos détaillées d'une version de modèle"
)
async def get_model_version(model_type: str, version: str):
    """Récupère les infos d'une version"""
    registry = get_registry()
    info = registry.get_model_version(model_type, version)
    
    if not info:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Version {version} non trouvée pour {model_type}"
        )
    
    # Ajouter les flags is_latest et is_production
    info['is_latest'] = version == registry.get_latest_version(model_type)
    info['is_production'] = version == registry.get_production_version(model_type)
    
    return info


@router.get(
    "/models/{model_type}/production",
    summary="Récupérer la version en production",
    description="Retourne la version actuellement en production"
)
async def get_production_version(model_type: str):
    """Récupère la version en production"""
    registry = get_registry()
    version = registry.get_production_version(model_type)
    
    if not version:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Modèle {model_type} non trouvé ou aucune version en production"
        )
    
    info = registry.get_model_version(model_type, version)
    
    return {
        "model_type": model_type,
        "production_version": version,
        **info
    }


@router.get(
    "/models/{model_type}/latest",
    summary="Récupérer la dernière version",
    description="Retourne la dernière version enregistrée"
)
async def get_latest_version(model_type: str):
    """Récupère la dernière version"""
    registry = get_registry()
    version = registry.get_latest_version(model_type)
    
    if not version:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Modèle {model_type} non trouvé"
        )
    
    info = registry.get_model_version(model_type, version)
    
    return {
        "model_type": model_type,
        "latest_version": version,
        **info
    }


@router.post(
    "/models/{model_type}/production",
    summary="Définir version en production",
    description="Définit une version spécifique comme version de production"
)
async def set_production_version(model_type: str, request: SetProductionRequest):
    """Définit la version en production"""
    registry = get_registry()
    
    success = registry.set_production(model_type, request.version)
    
    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Modèle {model_type} ou version {request.version} non trouvé"
        )
    
    return {
        "message": f"{model_type} v{request.version} est maintenant en production",
        "model_type": model_type,
        "production_version": request.version
    }


@router.post(
    "/models/{model_type}/rollback",
    summary="Rollback vers version précédente",
    description="Effectue un rollback vers la version de production précédente"
)
async def rollback_model(model_type: str):
    """Rollback vers la version précédente"""
    registry = get_registry()
    
    old_version = registry.get_production_version(model_type)
    new_version = registry.rollback(model_type)
    
    if not new_version:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Impossible de rollback {model_type} (pas de version antérieure)"
        )
    
    return {
        "message": f"Rollback effectué pour {model_type}",
        "model_type": model_type,
        "old_production": old_version,
        "new_production": new_version
    }


@router.delete(
    "/models/{model_type}/versions/{version}/deprecate",
    summary="Déprécier une version",
    description="Marque une version comme deprecated (ne peut pas être la version en production)"
)
async def deprecate_version(model_type: str, version: str):
    """Déprécier une version"""
    registry = get_registry()
    
    success = registry.deprecate_version(model_type, version)
    
    if not success:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Impossible de déprécier {model_type} v{version}"
        )
    
    return {
        "message": f"{model_type} v{version} marquée comme deprecated",
        "model_type": model_type,
        "version": version,
        "status": "deprecated"
    }


@router.get(
    "/export",
    summary="Exporter le registry",
    description="Exporte le registry complet en JSON (backup)"
)
async def export_registry():
    """Exporte le registry"""
    registry = get_registry()
    
    file_path = registry.export_registry()
    
    return {
        "message": "Registry exporté",
        "file_path": file_path,
        "timestamp": datetime.now().isoformat()
    }


@router.get(
    "/health",
    summary="Health check du registry",
    description="Vérifie l'état du registry"
)
async def registry_health():
    """Health check du registry"""
    registry = get_registry()
    models = registry.list_all_models()
    
    total_versions = sum(m['total_versions'] for m in models)
    
    return {
        "status": "healthy",
        "total_models": len(models),
        "total_versions": total_versions,
        "models": [m['model_type'] for m in models],
        "timestamp": datetime.now().isoformat()
    }