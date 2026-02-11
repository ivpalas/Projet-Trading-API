"""Point d'entrée FastAPI"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from api.config import API_TITLE, API_DESCRIPTION, API_VERSION
from api.routers.data import router as data_router  # Import corrigé

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
app.include_router(data_router)


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