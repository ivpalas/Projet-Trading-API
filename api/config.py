"""Configuration de l'API"""

from pathlib import Path

# Chemins
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_RAW = BASE_DIR / "data" / "raw"
DATA_PROCESSED = BASE_DIR / "data" / "processed"
DATA_FEATURES = BASE_DIR / "data" / "features"
MODELS_DIR = BASE_DIR / "models"

# API
API_TITLE = "GBP/USD Trading System API"
API_DESCRIPTION = """
API pour le système de trading algorithmique GBP/USD.

## Fonctionnalités

- 📊 Accès aux données M1 et M15
- 🔧 Calcul de features (indicateurs techniques)
- 📈 Stratégies baseline
- 🤖 Machine Learning (LogReg, RF, XGBoost)
- 🎮 Reinforcement Learning (DQN, PPO)
- 📉 Backtesting et évaluation

## Split temporel

- **2022** : Entraînement
- **2023** : Validation
- **2024** : Test final
"""
API_VERSION = "1.0.0"

# Trading
YEARS = [2022, 2023, 2024]
TRAIN_YEAR = 2022
VALID_YEAR = 2023
TEST_YEAR = 2024