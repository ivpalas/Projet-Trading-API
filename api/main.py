"""Point d'entrée FastAPI"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from api.config import API_TITLE, API_DESCRIPTION, API_VERSION
from api.routers.data import router as data_router
from api.routers.features import router as features_router
from api.routers.strategies import router as strategies_router
from api.routers.predictions import router as predictions_router

app = FastAPI(
    title=API_TITLE,
    description=API_DESCRIPTION,
    version=API_VERSION
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Routers
# Routers
app.include_router(data_router, prefix="/data", tags=["Data"])
app.include_router(features_router, prefix="/features", tags=["Features"])
app.include_router(strategies_router, prefix="/strategies", tags=["Strategies"])
app.include_router(predictions_router, prefix="/predictions", tags=["Predictions"])



@app.get("/")
def root():
    """Page d'accueil"""
    return {
        "message": "GBP/USD Trading System API",
        "version": API_VERSION,
        "docs": "/docs",
        "status": "operational"
    }


@app.get("/health")
def health_check():
    """Health check"""
    return {"status": "ok"}


@app.on_event("startup")
async def startup_event():
    """Événement au démarrage de l'API - Charge les modèles"""
    from api.services.model_loader import get_model_loader
    
    print("\n" + "="*80)
    print("🚀 DÉMARRAGE DE L'API")
    print("="*80)
    
    # Charger les modèles
    loader = get_model_loader()
    
    print(f"\n✓ API prête avec {len(loader.get_loaded_models())} modèles chargés")
    print("="*80 + "\n")