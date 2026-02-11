"""Schémas Pydantic communs"""

from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime


class MetaInfo(BaseModel):
    """Métadonnées communes"""
    dataset_id: Optional[str] = Field(None, description="ID du dataset")
    year: Optional[int] = Field(None, description="Année (2022, 2023, 2024)")
    schema_version: str = Field(default="1.0", description="Version du schéma")
    timestamp: Optional[datetime] = Field(default_factory=datetime.now)


class ResponseBase(BaseModel):
    """Réponse de base"""
    meta: MetaInfo
    success: bool = True
    message: Optional[str] = None