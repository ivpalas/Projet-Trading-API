"""
ML Trainer - Train and evaluate machine learning models
Supports: Logistic Regression, Random Forest, XGBoost
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any
import pickle
from pathlib import Path
from datetime import datetime

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("Warning: XGBoost not available. Install with: pip install xgboost")


class MLTrainer:
    """Train and evaluate ML models for trading predictions"""
    
    def __init__(self, model_dir: str = "models/saved"):
        """
        Args:
            model_dir: Directory to save trained models
        """
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        self.models = {}
        self.scalers = {}
        self.feature_names = []
        self.metrics = {}
    
    def prepare_data(
        self,
        df: pd.DataFrame,
        target_col: str = 'target',
        test_size: float = 0.2,
        exclude_cols: List[str] = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Prepare data for training (split and scale)
        
        Args:
            df: Input dataframe with features and target
            target_col: Name of target column
            test_size: Proportion of data for testing
            exclude_cols: Columns to exclude from features
            
        Returns:
            X_train, X_test, y_train, y_test
        """
        # Separate features and target
        if exclude_cols is None:
            exclude_cols = ['target']
        else:
            exclude_cols = list(exclude_cols) + ['target']
        
        # Remove non-numeric columns and target
        feature_cols = [col for col in df.columns 
                       if col not in exclude_cols 
                       and df[col].dtype in ['int64', 'float64']]
        
        X = df[feature_cols].values
        y = df[target_col].values
        
        self.feature_names = feature_cols
        
        print(f"Features: {len(feature_cols)}")
        print(f"Samples: {len(X)}")
        print(f"Target distribution: {np.bincount(y + 1)}")  # +1 because target is -1, 0, 1
        
        # Split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        # Scale features
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        
        self.scalers['main'] = scaler
        
        print(f"\nTrain set: {len(X_train)} samples")
        print(f"Test set: {len(X_test)} samples")
        
        return X_train, X_test, y_train, y_test
    
    def train_logistic_regression(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        **kwargs
    ) -> LogisticRegression:
        """
        Train Logistic Regression model
        
        Args:
            X_train: Training features
            y_train: Training target
            **kwargs: Additional arguments for LogisticRegression
            
        Returns:
            Trained model
        """
        print("\n=== Training Logistic Regression ===")
        
        default_params = {
            'max_iter': 1000,
            'random_state': 42,
            'multi_class': 'multinomial',
            'solver': 'lbfgs'
        }
        params = {**default_params, **kwargs}
        
        model = LogisticRegression(**params)
        model.fit(X_train, y_train)
        
        self.models['logistic_regression'] = model
        
        print("✓ Logistic Regression trained")
        return model
    
    def train_random_forest(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        **kwargs
    ) -> RandomForestClassifier:
        """
        Train Random Forest model
        
        Args:
            X_train: Training features
            y_train: Training target
            **kwargs: Additional arguments for RandomForestClassifier
            
        Returns:
            Trained model
        """
        print("\n=== Training Random Forest ===")
        
        default_params = {
            'n_estimators': 100,
            'max_depth': 10,
            'min_samples_split': 10,
            'min_samples_leaf': 5,
            'random_state': 42,
            'n_jobs': -1
        }
        params = {**default_params, **kwargs}
        
        model = RandomForestClassifier(**params)
        model.fit(X_train, y_train)
        
        self.models['random_forest'] = model
        
        print("✓ Random Forest trained")
        return model
    
    def train_xgboost(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        **kwargs
    ):
        """
        Train XGBoost model
        
        Args:
            X_train: Training features
            y_train: Training target (must be 0, 1, 2 for XGBoost)
            **kwargs: Additional arguments for XGBClassifier
            
        Returns:
            Trained model
        """
        if not XGBOOST_AVAILABLE:
            print("XGBoost not available!")
            return None
        
        print("\n=== Training XGBoost ===")
        
        # XGBoost requires labels starting from 0
        # Our labels are -1, 0, 1, so we add 1 to get 0, 1, 2
        y_train_xgb = y_train + 1
        
        default_params = {
            'n_estimators': 100,
            'max_depth': 6,
            'learning_rate': 0.1,
            'objective': 'multi:softmax',
            'num_class': 3,
            'random_state': 42,
            'n_jobs': -1
        }
        params = {**default_params, **kwargs}
        
        model = xgb.XGBClassifier(**params)
        model.fit(X_train, y_train_xgb)
        
        self.models['xgboost'] = model
        
        print("✓ XGBoost trained")
        return model
    
    def evaluate_model(
        self,
        model: Any,
        X_test: np.ndarray,
        y_test: np.ndarray,
        model_name: str
    ) -> Dict:
        """
        Evaluate model performance
        
        Args:
            model: Trained model
            X_test: Test features
            y_test: Test target
            model_name: Name of the model
            
        Returns:
            Dictionary with metrics
        """
        print(f"\n=== Evaluating {model_name} ===")
        
        # Predictions
        if model_name == 'xgboost' and XGBOOST_AVAILABLE:
            # XGBoost returns 0, 1, 2, we need to convert back to -1, 0, 1
            y_pred = model.predict(X_test) - 1
        else:
            y_pred = model.predict(X_test)
        
        # Metrics
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision_macro': precision_score(y_test, y_pred, average='macro', zero_division=0),
            'recall_macro': recall_score(y_test, y_pred, average='macro', zero_division=0),
            'f1_macro': f1_score(y_test, y_pred, average='macro', zero_division=0),
        }
        
        # Per-class metrics
        for label, name in [(-1, 'DOWN'), (0, 'HOLD'), (1, 'UP')]:
            if label in y_test:
                metrics[f'precision_{name}'] = precision_score(
                    y_test == label, y_pred == label, zero_division=0
                )
                metrics[f'recall_{name}'] = recall_score(
                    y_test == label, y_pred == label, zero_division=0
                )
        
        # Print metrics
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print(f"Precision (macro): {metrics['precision_macro']:.4f}")
        print(f"Recall (macro): {metrics['recall_macro']:.4f}")
        print(f"F1 Score (macro): {metrics['f1_macro']:.4f}")
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        print("\nConfusion Matrix:")
        print(cm)
        
        # Classification report
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=['DOWN', 'HOLD', 'UP'], zero_division=0))
        
        self.metrics[model_name] = metrics
        
        return metrics
    
    def train_all_models(
        self,
        X_train: np.ndarray,
        X_test: np.ndarray,
        y_train: np.ndarray,
        y_test: np.ndarray
    ) -> Dict[str, Any]:
        """
        Train all available models and evaluate them
        
        Args:
            X_train, X_test: Training and test features
            y_train, y_test: Training and test targets
            
        Returns:
            Dictionary with all trained models
        """
        print("=" * 60)
        print("TRAINING ALL MODELS")
        print("=" * 60)
        
        # 1. Logistic Regression
        lr_model = self.train_logistic_regression(X_train, y_train)
        self.evaluate_model(lr_model, X_test, y_test, 'logistic_regression')
        
        # 2. Random Forest
        rf_model = self.train_random_forest(X_train, y_train)
        self.evaluate_model(rf_model, X_test, y_test, 'random_forest')
        
        # 3. XGBoost (if available)
        if XGBOOST_AVAILABLE:
            xgb_model = self.train_xgboost(X_train, y_train)
            # For evaluation, we need to adjust y_test for XGBoost
            self.evaluate_model(xgb_model, X_test, y_test, 'xgboost')
        
        print("\n" + "=" * 60)
        print("TRAINING COMPLETE")
        print("=" * 60)
        
        # Print comparison
        self.print_comparison()
        
        return self.models
    
    def print_comparison(self):
        """Print comparison of all models"""
        if not self.metrics:
            print("No models evaluated yet!")
            return
        
        print("\n=== MODEL COMPARISON ===\n")
        
        # Create comparison table
        print(f"{'Model':<20} {'Accuracy':<12} {'F1 (macro)':<12} {'Precision':<12} {'Recall':<12}")
        print("-" * 68)
        
        for model_name, metrics in self.metrics.items():
            print(f"{model_name:<20} "
                  f"{metrics['accuracy']:<12.4f} "
                  f"{metrics['f1_macro']:<12.4f} "
                  f"{metrics['precision_macro']:<12.4f} "
                  f"{metrics['recall_macro']:<12.4f}")
        
        # Best model
        best_model = max(self.metrics.items(), key=lambda x: x[1]['f1_macro'])
        print(f"\n🏆 Best model (F1): {best_model[0]} ({best_model[1]['f1_macro']:.4f})")
    
    def save_models(self, prefix: str = ""):
        """
        Save all trained models and scalers
        
        Args:
            prefix: Prefix for saved files (e.g., "2022_")
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        for model_name, model in self.models.items():
            filename = f"{prefix}{model_name}_{timestamp}.pkl"
            filepath = self.model_dir / filename
            
            with open(filepath, 'wb') as f:
                pickle.dump(model, f)
            
            print(f"✓ Saved: {filepath}")
        
        # Save scalers
        for scaler_name, scaler in self.scalers.items():
            filename = f"{prefix}scaler_{scaler_name}_{timestamp}.pkl"
            filepath = self.model_dir / filename
            
            with open(filepath, 'wb') as f:
                pickle.dump(scaler, f)
            
            print(f"✓ Saved: {filepath}")
        
        # Save feature names
        filename = f"{prefix}feature_names_{timestamp}.pkl"
        filepath = self.model_dir / filename
        with open(filepath, 'wb') as f:
            pickle.dump(self.feature_names, f)
        print(f"✓ Saved: {filepath}")
        
        # Save metrics
        filename = f"{prefix}metrics_{timestamp}.pkl"
        filepath = self.model_dir / filename
        with open(filepath, 'wb') as f:
            pickle.dump(self.metrics, f)
        print(f"✓ Saved: {filepath}")
    
    def load_model(self, filepath: str) -> Any:
        """
        Load a saved model
        
        Args:
            filepath: Path to the saved model
            
        Returns:
            Loaded model
        """
        with open(filepath, 'rb') as f:
            model = pickle.load(f)
        
        print(f"✓ Loaded model from: {filepath}")
        return model


if __name__ == "__main__":
    # Example usage
    print("ML Trainer - Example Usage\n")
    
    # Load ML dataset (created by feature_engineering.py)
    df = pd.read_parquet('data/processed/ml_dataset_2022.parquet')
    
    # Initialize trainer
    trainer = MLTrainer(model_dir='models/saved')
    
    # Prepare data
    X_train, X_test, y_train, y_test = trainer.prepare_data(df, test_size=0.2)
    
    # Train all models
    models = trainer.train_all_models(X_train, X_test, y_train, y_test)
    
    # Save models
    trainer.save_models(prefix="2022_")
    
    print("\n✓ Training pipeline complete!")