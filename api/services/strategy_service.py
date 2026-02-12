"""Service pour les stratégies de trading"""

import pandas as pd
import numpy as np
from typing import List, Dict
from api.services.feature_service import FeatureService
from api.schemas.strategies import Trade, BacktestResult


class StrategyService:
    """Service de backtesting des stratégies"""
    
    def __init__(self):
        self.feature_service = FeatureService()
        
        self.strategies = {
            'random': {
                'name': 'Random',
                'description': 'Achète/vend aléatoirement'
            },
            'buy_hold': {
                'name': 'Buy & Hold',
                'description': 'Achète au début, vend à la fin'
            },
            'sma_crossover': {
                'name': 'SMA Crossover',
                'description': 'EMA20 croise EMA50'
            }
        }
    
    def get_available_strategies(self) -> List[Dict]:
        """Liste des stratégies disponibles"""
        return [
            {'id': key, 'name': val['name'], 'description': val['description']}
            for key, val in self.strategies.items()
        ]
    
    def backtest(
        self, 
        strategy: str, 
        year: int, 
        initial_capital: float = 10000.0
    ) -> BacktestResult:
        """
        Backteste une stratégie
        
        Args:
            strategy: Nom de la stratégie
            year: Année
            initial_capital: Capital initial
        
        Returns:
            BacktestResult
        """
        
        # Charger les données avec features
        df = self.feature_service.compute_features(year)
        df = df.dropna()  # Supprimer les NaN
        
        # Appeler la stratégie appropriée
        if strategy == 'random':
            trades = self._random_strategy(df, initial_capital)
        elif strategy == 'buy_hold':
            trades = self._buy_hold_strategy(df, initial_capital)
        elif strategy == 'sma_crossover':
            trades = self._sma_crossover_strategy(df, initial_capital)
        else:
            raise ValueError(f"Stratégie inconnue: {strategy}")
        
        # Calculer les métriques
        result = self._calculate_metrics(
            strategy=strategy,
            year=year,
            initial_capital=initial_capital,
            trades=trades
        )
        
        return result
    
    def _random_strategy(self, df: pd.DataFrame, initial_capital: float) -> List[Trade]:
        """Stratégie aléatoire"""
        
        trades = []
        capital = initial_capital
        position = 0  # Quantité détenue
        
        np.random.seed(42)  # Reproductibilité
        
        for idx, row in df.iterrows():
            # Décision aléatoire
            action = np.random.choice(['buy', 'sell', 'hold'], p=[0.1, 0.1, 0.8])
            
            if action == 'buy' and capital > 0:
                # Acheter avec 10% du capital
                amount = capital * 0.1
                quantity = amount / row['close_15m']
                position += quantity
                capital -= amount
                
                trades.append(Trade(
                    timestamp=row['timestamp_15m'],
                    action='buy',
                    price=row['close_15m'],
                    quantity=quantity,
                    capital_after=capital + (position * row['close_15m'])
                ))
            
            elif action == 'sell' and position > 0:
                # Vendre 10% de la position
                quantity = position * 0.1
                amount = quantity * row['close_15m']
                position -= quantity
                capital += amount
                
                trades.append(Trade(
                    timestamp=row['timestamp_15m'],
                    action='sell',
                    price=row['close_15m'],
                    quantity=quantity,
                    capital_after=capital + (position * row['close_15m'])
                ))
        
        # Vendre tout à la fin
        if position > 0:
            last_row = df.iloc[-1]
            amount = position * last_row['close_15m']
            capital += amount
            
            trades.append(Trade(
                timestamp=last_row['timestamp_15m'],
                action='sell',
                price=last_row['close_15m'],
                quantity=position,
                capital_after=capital
            ))
        
        return trades
    
    def _buy_hold_strategy(self, df: pd.DataFrame, initial_capital: float) -> List[Trade]:
        """Stratégie Buy & Hold"""
        
        trades = []
        
        # Acheter au début
        first_row = df.iloc[0]
        quantity = initial_capital / first_row['close_15m']
        
        trades.append(Trade(
            timestamp=first_row['timestamp_15m'],
            action='buy',
            price=first_row['close_15m'],
            quantity=quantity,
            capital_after=initial_capital
        ))
        
        # Vendre à la fin
        last_row = df.iloc[-1]
        final_capital = quantity * last_row['close_15m']
        
        trades.append(Trade(
            timestamp=last_row['timestamp_15m'],
            action='sell',
            price=last_row['close_15m'],
            quantity=quantity,
            capital_after=final_capital
        ))
        
        return trades
    
    def _sma_crossover_strategy(self, df: pd.DataFrame, initial_capital: float) -> List[Trade]:
        """Stratégie SMA Crossover (EMA20 x EMA50)"""
        
        trades = []
        capital = initial_capital
        position = 0
        
        for i in range(1, len(df)):
            prev_row = df.iloc[i-1]
            curr_row = df.iloc[i]
            
            # Signal d'achat : EMA20 croise EMA50 vers le haut
            if (prev_row['ema_20'] <= prev_row['ema_50'] and 
                curr_row['ema_20'] > curr_row['ema_50'] and 
                capital > 0):
                
                # Acheter avec 50% du capital
                amount = capital * 0.5
                quantity = amount / curr_row['close_15m']
                position += quantity
                capital -= amount
                
                trades.append(Trade(
                    timestamp=curr_row['timestamp_15m'],
                    action='buy',
                    price=curr_row['close_15m'],
                    quantity=quantity,
                    capital_after=capital + (position * curr_row['close_15m'])
                ))
            
            # Signal de vente : EMA20 croise EMA50 vers le bas
            elif (prev_row['ema_20'] >= prev_row['ema_50'] and 
                  curr_row['ema_20'] < curr_row['ema_50'] and 
                  position > 0):
                
                # Vendre 50% de la position
                quantity = position * 0.5
                amount = quantity * curr_row['close_15m']
                position -= quantity
                capital += amount
                
                trades.append(Trade(
                    timestamp=curr_row['timestamp_15m'],
                    action='sell',
                    price=curr_row['close_15m'],
                    quantity=quantity,
                    capital_after=capital + (position * curr_row['close_15m'])
                ))
        
        # Vendre tout à la fin
        if position > 0:
            last_row = df.iloc[-1]
            amount = position * last_row['close_15m']
            capital += amount
            
            trades.append(Trade(
                timestamp=last_row['timestamp_15m'],
                action='sell',
                price=last_row['close_15m'],
                quantity=position,
                capital_after=capital
            ))
        
        return trades
    
    def _calculate_metrics(
        self, 
        strategy: str, 
        year: int, 
        initial_capital: float, 
        trades: List[Trade]
    ) -> BacktestResult:
        """Calcule les métriques de performance"""
        
        if not trades:
            return BacktestResult(
                strategy=strategy,
                year=year,
                initial_capital=initial_capital,
                final_capital=initial_capital,
                total_return=0.0,
                total_trades=0,
                winning_trades=0,
                losing_trades=0,
                win_rate=0.0,
                trades=[]
            )
        
        final_capital = trades[-1].capital_after
        total_return = ((final_capital - initial_capital) / initial_capital) * 100
        
        # Compter les trades gagnants/perdants
        winning_trades = 0
        losing_trades = 0
        
        for i in range(1, len(trades)):
            if trades[i].action == 'sell':
                # Trouver le buy correspondant
                for j in range(i-1, -1, -1):
                    if trades[j].action == 'buy':
                        if trades[i].price > trades[j].price:
                            winning_trades += 1
                        else:
                            losing_trades += 1
                        break
        
        total_trades = winning_trades + losing_trades
        win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0.0
        
        return BacktestResult(
            strategy=strategy,
            year=year,
            initial_capital=initial_capital,
            final_capital=final_capital,
            total_return=total_return,
            total_trades=len(trades),
            winning_trades=winning_trades,
            losing_trades=losing_trades,
            win_rate=win_rate,
            trades=trades
        )