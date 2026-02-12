"""
ML Backtester - Backtest ML model predictions
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List
import pickle


class MLBacktester:
    """Backtest ML trading strategies"""
    
    def __init__(self, initial_capital: float = 10000.0):
        """
        Args:
            initial_capital: Starting capital in EUR
        """
        self.initial_capital = initial_capital
        self.results = {}
    
    def backtest_ml_strategy(
        self,
        df: pd.DataFrame,
        model: Any,
        scaler: Any,
        feature_names: List[str],
        position_size: float = 0.5,
        model_name: str = "ml_model"
    ) -> Dict:
        """
        Backtest ML model predictions
        
        Args:
            df: DataFrame with features and actual prices
            model: Trained ML model
            scaler: Fitted scaler
            feature_names: List of feature column names
            position_size: Fraction of capital to use per trade
            model_name: Name for the model (for results)
            
        Returns:
            Dictionary with backtest results
        """
        print(f"\n=== Backtesting {model_name} ===")
        
        # Prepare features
        X = df[feature_names].values
        X_scaled = scaler.transform(X)
        
        # Get predictions
        predictions = model.predict(X_scaled)
        
        # Adjust XGBoost predictions (0,1,2 -> -1,0,1)
        if model_name == 'xgboost' or 'xgb' in str(type(model)).lower():
            predictions = predictions - 1
        
        # Déterminer le nom de la colonne close
        close_col = 'close_15m' if 'close_15m' in df.columns else 'close'
        
        # Initialize
        capital = self.initial_capital
        position = 0  # Current position in GBP
        trades = []
        
        # Backtest loop
        for i in range(len(df)):
            timestamp = df.index[i]
            price = df.iloc[i][close_col]
            signal = predictions[i]
            
            # Execute trades based on signal
            if signal == 1 and position <= 0:  # BUY signal
                # Close short if any
                if position < 0:
                    capital += abs(position) * price
                    trades.append({
                        'timestamp': timestamp,
                        'action': 'sell',
                        'price': price,
                        'quantity': abs(position),
                        'capital_after': capital
                    })
                    position = 0
                
                # Open long
                quantity = (capital * position_size) / price
                position = quantity
                capital -= quantity * price
                
                trades.append({
                    'timestamp': timestamp,
                    'action': 'buy',
                    'price': price,
                    'quantity': quantity,
                    'capital_after': capital
                })
            
            elif signal == -1 and position >= 0:  # SELL signal
                # Close long if any
                if position > 0:
                    capital += position * price
                    trades.append({
                        'timestamp': timestamp,
                        'action': 'sell',
                        'price': price,
                        'quantity': position,
                        'capital_after': capital
                    })
                    position = 0
                
                # Open short
                quantity = (capital * position_size) / price
                position = -quantity
                capital += quantity * price
                
                trades.append({
                    'timestamp': timestamp,
                    'action': 'buy',
                    'price': price,
                    'quantity': quantity,
                    'capital_after': capital
                })
        
        # Close any remaining position
        if position != 0:
            final_price = df.iloc[-1][close_col]
            if position > 0:
                capital += position * final_price
                trades.append({
                    'timestamp': df.index[-1],
                    'action': 'sell',
                    'price': final_price,
                    'quantity': position,
                    'capital_after': capital
                })
            else:
                capital -= abs(position) * final_price
                trades.append({
                    'timestamp': df.index[-1],
                    'action': 'buy',
                    'price': final_price,
                    'quantity': abs(position),
                    'capital_after': capital
                })
            position = 0
        
        # Calculate metrics
        final_capital = capital
        total_return = ((final_capital - self.initial_capital) / self.initial_capital) * 100
        
        # Analyze trades
        winning_trades = 0
        losing_trades = 0
        
        for i in range(0, len(trades) - 1, 2):
            if i + 1 < len(trades):
                entry = trades[i]
                exit_trade = trades[i + 1]
                
                if entry['action'] == 'buy':
                    pnl = (exit_trade['price'] - entry['price']) * entry['quantity']
                else:
                    pnl = (entry['price'] - exit_trade['price']) * entry['quantity']
                
                if pnl > 0:
                    winning_trades += 1
                else:
                    losing_trades += 1
        
        total_trades = winning_trades + losing_trades
        win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
        
        # Signal distribution
        signal_counts = pd.Series(predictions).value_counts()
        
        results = {
            'model': model_name,
            'initial_capital': self.initial_capital,
            'final_capital': final_capital,
            'total_return': total_return,
            'total_trades': len(trades),
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'win_rate': win_rate,
            'trades': trades,
            'signal_distribution': {
                'DOWN': int(signal_counts.get(-1, 0)),
                'HOLD': int(signal_counts.get(0, 0)),
                'UP': int(signal_counts.get(1, 0))
            }
        }
        
        # Print summary
        print(f"Initial Capital: {self.initial_capital:.2f} €")
        print(f"Final Capital: {final_capital:.2f} €")
        print(f"Total Return: {total_return:.2f}%")
        print(f"Total Trades: {len(trades)}")
        print(f"Winning Trades: {winning_trades}")
        print(f"Losing Trades: {losing_trades}")
        print(f"Win Rate: {win_rate:.2f}%")
        print(f"\nSignal Distribution:")
        print(f"  DOWN: {results['signal_distribution']['DOWN']}")
        print(f"  HOLD: {results['signal_distribution']['HOLD']}")
        print(f"  UP: {results['signal_distribution']['UP']}")
        
        self.results[model_name] = results
        return results
    
    def compare_models(self):
        """Compare all backtested models"""
        if not self.results:
            print("No models backtested yet!")
            return
        
        print("\n" + "=" * 80)
        print("MODEL BACKTEST COMPARISON")
        print("=" * 80)
        
        print(f"\n{'Model':<20} {'Return %':<12} {'Trades':<10} {'Win Rate %':<12} {'Final Capital':<15}")
        print("-" * 80)
        
        for model_name, results in self.results.items():
            print(f"{model_name:<20} "
                  f"{results['total_return']:<12.2f} "
                  f"{results['total_trades']:<10} "
                  f"{results['win_rate']:<12.2f} "
                  f"{results['final_capital']:<15.2f}")
        
        # Best model
        best_model = max(self.results.items(), key=lambda x: x[1]['total_return'])
        print(f"\n🏆 Best model (Return): {best_model[0]} ({best_model[1]['total_return']:.2f}%)")


def load_model_artifacts(model_path: str, scaler_path: str, features_path: str):
    """
    Load model, scaler, and feature names
    
    Args:
        model_path: Path to saved model
        scaler_path: Path to saved scaler
        features_path: Path to saved feature names
        
    Returns:
        model, scaler, feature_names
    """
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    
    with open(features_path, 'rb') as f:
        feature_names = pickle.load(f)
    
    return model, scaler, feature_names


if __name__ == "__main__":
    # Example usage
    print("ML Backtester - Example Usage\n")
    
    # Load test data (2023 for out-of-sample testing)
    # This assumes you have features for 2023
    # df_test = pd.read_parquet('data/processed/ml_dataset_2023.parquet')
    
    # Load model artifacts (adjust paths to your actual saved models)
    # model, scaler, feature_names = load_model_artifacts(
    #     'models/saved/2022_random_forest_TIMESTAMP.pkl',
    #     'models/saved/2022_scaler_main_TIMESTAMP.pkl',
    #     'models/saved/2022_feature_names_TIMESTAMP.pkl'
    # )
    
    # Initialize backtester
    # backtester = MLBacktester(initial_capital=10000.0)
    
    # Backtest
    # results = backtester.backtest_ml_strategy(
    #     df_test, model, scaler, feature_names,
    #     position_size=0.5,
    #     model_name='random_forest'
    # )
    
    print("Example code - uncomment and adjust paths to run!")