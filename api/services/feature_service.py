"""Service pour calculer les features techniques"""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional
from api.services.data_service import DataService


class FeatureService:
    """Service de calcul des features techniques"""
    
    def __init__(self):
        self.data_service = DataService()
        
        # Définition des groupes de features
        self.feature_groups = {
            'short_term': [
                'return_1', 'return_4', 
                'ema_20', 'ema_50', 'ema_diff',
                'rsi_14', 'rolling_std_20',
                'range_15m', 'body', 'upper_wick', 'lower_wick'
            ],
            'regime': [
                'ema_200', 'distance_to_ema200', 'slope_ema50',
                'atr_14', 'rolling_std_100', 'volatility_ratio',
                'adx_14', 'macd', 'macd_signal'
            ]
        }
    
    def get_available_features(self) -> Dict:
        """Liste des features disponibles"""
        all_features = []
        for features in self.feature_groups.values():
            all_features.extend(features)
        
        return {
            'available_features': self.feature_groups,
            'total_features': len(all_features)
        }
    
    def compute_features(
        self, 
        year: int, 
        features: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Calcule les features pour une année
        
        Args:
            year: Année
            features: Liste des features à calculer (None = toutes)
        
        Returns:
            DataFrame avec features calculées
        """
        
        # Charger les données
        df = self.data_service.load_m15(year).copy()
        
        # Si aucune feature spécifiée, calculer toutes
        if features is None:
            features = []
            for group_features in self.feature_groups.values():
                features.extend(group_features)
        
        # Calculer chaque feature
        for feature in features:
            if feature in self.feature_groups['short_term']:
                df = self._compute_short_term_feature(df, feature)
            elif feature in self.feature_groups['regime']:
                df = self._compute_regime_feature(df, feature)
        
        return df
    
    def _compute_short_term_feature(self, df: pd.DataFrame, feature: str) -> pd.DataFrame:
        """Calcule une feature court terme"""
        
        if feature == 'return_1':
            df['return_1'] = df['close_15m'].pct_change(1)
        
        elif feature == 'return_4':
            df['return_4'] = df['close_15m'].pct_change(4)
        
        elif feature == 'ema_20':
            df['ema_20'] = df['close_15m'].ewm(span=20, adjust=False).mean()
        
        elif feature == 'ema_50':
            df['ema_50'] = df['close_15m'].ewm(span=50, adjust=False).mean()
        
        elif feature == 'ema_diff':
            if 'ema_20' not in df.columns:
                df['ema_20'] = df['close_15m'].ewm(span=20, adjust=False).mean()
            if 'ema_50' not in df.columns:
                df['ema_50'] = df['close_15m'].ewm(span=50, adjust=False).mean()
            df['ema_diff'] = df['ema_20'] - df['ema_50']
        
        elif feature == 'rsi_14':
            df['rsi_14'] = self._calculate_rsi(df['close_15m'], 14)
        
        elif feature == 'rolling_std_20':
            df['rolling_std_20'] = df['close_15m'].rolling(window=20).std()
        
        elif feature == 'range_15m':
            df['range_15m'] = df['high_15m'] - df['low_15m']
        
        elif feature == 'body':
            df['body'] = abs(df['close_15m'] - df['open_15m'])
        
        elif feature == 'upper_wick':
            df['upper_wick'] = df['high_15m'] - df[['close_15m', 'open_15m']].max(axis=1)
        
        elif feature == 'lower_wick':
            df['lower_wick'] = df[['close_15m', 'open_15m']].min(axis=1) - df['low_15m']
        
        return df
    
    def _compute_regime_feature(self, df: pd.DataFrame, feature: str) -> pd.DataFrame:
        """Calcule une feature de régime"""
        
        if feature == 'ema_200':
            df['ema_200'] = df['close_15m'].ewm(span=200, adjust=False).mean()
        
        elif feature == 'distance_to_ema200':
            if 'ema_200' not in df.columns:
                df['ema_200'] = df['close_15m'].ewm(span=200, adjust=False).mean()
            df['distance_to_ema200'] = (df['close_15m'] - df['ema_200']) / df['ema_200']
        
        elif feature == 'slope_ema50':
            if 'ema_50' not in df.columns:
                df['ema_50'] = df['close_15m'].ewm(span=50, adjust=False).mean()
            df['slope_ema50'] = df['ema_50'].diff(1)
        
        elif feature == 'atr_14':
            df['atr_14'] = self._calculate_atr(df, 14)
        
        elif feature == 'rolling_std_100':
            df['rolling_std_100'] = df['close_15m'].rolling(window=100).std()
        
        elif feature == 'volatility_ratio':
            if 'rolling_std_20' not in df.columns:
                df['rolling_std_20'] = df['close_15m'].rolling(window=20).std()
            if 'rolling_std_100' not in df.columns:
                df['rolling_std_100'] = df['close_15m'].rolling(window=100).std()
            df['volatility_ratio'] = df['rolling_std_20'] / df['rolling_std_100']
        
        elif feature == 'adx_14':
            df['adx_14'] = self._calculate_adx(df, 14)
        
        elif feature == 'macd':
            macd, signal = self._calculate_macd(df['close_15m'])
            df['macd'] = macd
            df['macd_signal'] = signal
        
        elif feature == 'macd_signal':
            if 'macd' not in df.columns:
                macd, signal = self._calculate_macd(df['close_15m'])
                df['macd'] = macd
                df['macd_signal'] = signal
        
        return df
    
    def _calculate_rsi(self, series: pd.Series, period: int = 14) -> pd.Series:
        """Calcule le RSI"""
        delta = series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def _calculate_atr(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calcule l'ATR (Average True Range)"""
        high = df['high_15m']
        low = df['low_15m']
        close = df['close_15m']
        
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean()
        
        return atr
    
    def _calculate_adx(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calcule l'ADX (Average Directional Index)"""
        high = df['high_15m']
        low = df['low_15m']
        close = df['close_15m']
        
        # +DM et -DM
        plus_dm = high.diff()
        minus_dm = -low.diff()
        
        plus_dm[plus_dm < 0] = 0
        minus_dm[minus_dm < 0] = 0
        
        # ATR
        atr = self._calculate_atr(df, period)
        
        # +DI et -DI
        plus_di = 100 * (plus_dm.rolling(window=period).mean() / atr)
        minus_di = 100 * (minus_dm.rolling(window=period).mean() / atr)
        
        # DX
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
        
        # ADX
        adx = dx.rolling(window=period).mean()
        
        return adx
    
    def _calculate_macd(
        self, 
        series: pd.Series, 
        fast: int = 12, 
        slow: int = 26, 
        signal: int = 9
    ) -> tuple:
        """Calcule le MACD et son signal"""
        ema_fast = series.ewm(span=fast, adjust=False).mean()
        ema_slow = series.ewm(span=slow, adjust=False).mean()
        
        macd = ema_fast - ema_slow
        macd_signal = macd.ewm(span=signal, adjust=False).mean()
        
        return macd, macd_signal