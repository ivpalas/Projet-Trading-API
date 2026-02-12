"""Service pour gérer les données"""

import pandas as pd
from typing import Dict
from api.config import DATA_PROCESSED, YEARS


class DataService:
    """Service de gestion des données M15"""
    
    def __init__(self):
        self.data_path = DATA_PROCESSED
        self._cache = {}
    
    def _validate_year(self, year: int):
        """Valide l'année"""
        if year not in YEARS:
            raise ValueError(f"Année doit être dans {YEARS}")
    
    def load_m15(self, year: int) -> pd.DataFrame:
        """Charge les données M15"""
        
        self._validate_year(year)
        
        if year in self._cache:
            return self._cache[year]
        
        file_path = self.data_path / f"m15_{year}.csv"
        
        if not file_path.exists():
            raise FileNotFoundError(f"Fichier m15_{year}.csv introuvable")
        
        df = pd.read_csv(file_path)
        df['timestamp_15m'] = pd.to_datetime(df['timestamp_15m'])
        
        self._cache[year] = df
        return df
    
    def get_candles(self, year: int, limit: int = 100, offset: int = 0) -> Dict:
        """Récupère des bougies avec pagination"""
        
        if limit > 1000:
            raise ValueError("Limite maximum: 1000 bougies")
        
        df = self.load_m15(year)
        df_page = df.iloc[offset:offset+limit]
        
        return {
            'candles': df_page.to_dict(orient='records'),
            'total': len(df),
            'returned': len(df_page),
            'offset': offset
        }
    
    def get_stats(self, year: int) -> Dict:
        """Calcule les statistiques"""
        
        df = self.load_m15(year)
        
        return {
            'year': year,
            'n_candles': len(df),
            'period_start': df['timestamp_15m'].min(),
            'period_end': df['timestamp_15m'].max(),
            'price_mean': float(df['close_15m'].mean()),
            'price_std': float(df['close_15m'].std()),
            'price_min': float(df['close_15m'].min()),
            'price_max': float(df['close_15m'].max())
        }
    
    def get_available_years(self) -> Dict:
        """Liste des années disponibles"""
        
        years_info = []
        
        for year in YEARS:
            try:
                stats = self.get_stats(year)
                years_info.append({
                    'year': year,
                    'n_candles': stats['n_candles'],
                    'usage': 'train' if year == 2022 else 'valid' if year == 2023 else 'test'
                })
            except:
                pass
        
        return {
            'years': years_info,
            'split': {
                'train': 2022,
                'valid': 2023,
                'test': 2024
            }
        }
    
    def get_all_data(self, year: int) -> pd.DataFrame:
        """Récupère toutes les données (usage interne)"""
        return self.load_m15(year)