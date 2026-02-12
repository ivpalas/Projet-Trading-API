# T07 - Machine Learning Models

## 📋 Overview

This module implements **Machine Learning models** for GBP/USD trading predictions:
- **Logistic Regression** (baseline ML)
- **Random Forest** (ensemble method)
- **XGBoost** (gradient boosting)

## 🎯 Target Variable

Classification task with 3 classes:
- **UP (1)**: Price increases > 0.1% in next period
- **HOLD (0)**: Price change within ±0.1%
- **DOWN (-1)**: Price decreases > 0.1%

## 🔧 Features

### Technical Indicators (from T05)
- RSI, MACD, Bollinger Bands, ATR
- EMA20, EMA50, SMA20, SMA50
- Stochastic, Williams %R
- ADX, OBV, CMF, Momentum
- CCI, ROC, TRIX, VWAP

### Engineered Features
1. **Lag Features**: Price/volume at t-1, t-2, t-3, t-5, t-10, t-20
2. **Rolling Statistics**: Mean, std, min, max over 5/10/20/50 periods
3. **Price Changes**: Returns, log returns over 1/5/10/20 periods
4. **Volatility**: Rolling std of returns
5. **Time Features**: Hour, day of week, trading sessions
6. **Price Levels**: Distance from high/low, candle patterns

## 📂 File Structure

```
src/models/
├── feature_engineering.py   # Feature creation
├── ml_trainer.py             # Model training
├── ml_backtester.py          # Backtesting
└── run_ml_pipeline.py        # Complete pipeline

models/saved/                 # Trained models (.pkl)
data/processed/               # ML datasets
```

## 🚀 Usage

### 1. Install Dependencies

```bash
pip install -r requirements_ml.txt
```

### 2. Run Complete Pipeline

```bash
python src/models/run_ml_pipeline.py
```

This will:
1. Create ML datasets (2020-2023)
2. Train models on 2020-2022
3. Backtest on 2023

### 3. Individual Steps

#### Create Features
```python
from feature_engineering import FeatureEngineer, add_target_variable

# Load data
df = pd.read_parquet('data/processed/m15_features_2022.parquet')

# Create features
engineer = FeatureEngineer()
df_features = engineer.create_all_features(df)

# Add target
df_ml = add_target_variable(df_features, threshold=0.001)

# Save
df_ml.to_parquet('data/processed/ml_dataset_2022.parquet')
```

#### Train Models
```python
from ml_trainer import MLTrainer

# Load ML dataset
df = pd.read_parquet('data/processed/ml_dataset_2022.parquet')

# Initialize trainer
trainer = MLTrainer(model_dir='models/saved')

# Prepare data
X_train, X_test, y_train, y_test = trainer.prepare_data(df)

# Train all models
models = trainer.train_all_models(X_train, X_test, y_train, y_test)

# Save
trainer.save_models(prefix="2022_")
```

#### Backtest
```python
from ml_backtester import MLBacktester, load_model_artifacts

# Load model
model, scaler, features = load_model_artifacts(
    'models/saved/random_forest.pkl',
    'models/saved/scaler_main.pkl',
    'models/saved/feature_names.pkl'
)

# Load test data
df_test = pd.read_parquet('data/processed/ml_dataset_2023.parquet')

# Backtest
backtester = MLBacktester(initial_capital=10000)
results = backtester.backtest_ml_strategy(
    df_test, model, scaler, features,
    position_size=0.5,
    model_name='random_forest'
)
```

## 📊 Expected Results

### Training (2020-2022)
- Logistic Regression: ~40-45% accuracy
- Random Forest: ~45-50% accuracy
- XGBoost: ~50-55% accuracy

### Backtesting (2023)
Models should outperform baseline strategies (Buy&Hold, Random) from T06.

## 🔍 Feature Importance

After training, you can analyze feature importance:

```python
# Random Forest
importances = trainer.models['random_forest'].feature_importances_
feature_imp = pd.DataFrame({
    'feature': trainer.feature_names,
    'importance': importances
}).sort_values('importance', ascending=False)

print(feature_imp.head(20))
```

## ⚙️ Hyperparameter Tuning

To improve models, you can tune hyperparameters:

```python
# Random Forest with custom params
rf_model = trainer.train_random_forest(
    X_train, y_train,
    n_estimators=200,
    max_depth=15,
    min_samples_split=20
)
```

## 📈 Next Steps (T08)

- Implement Reinforcement Learning agents (DQN, PPO)
- Add ensemble methods (voting, stacking)
- Implement walk-forward analysis
- Add transaction costs simulation

## 🐛 Troubleshooting

### XGBoost Not Available
```bash
pip install xgboost
```

### Memory Issues
Reduce data size or use smaller models:
```python
trainer.train_random_forest(
    X_train, y_train,
    n_estimators=50,  # Reduce from 100
    max_depth=5        # Reduce from 10
)
```

### Imbalanced Classes
The HOLD class might be overrepresented. Consider:
- Adjusting threshold (increase from 0.1%)
- Using class weights
- SMOTE for oversampling

## 📝 Notes

- Models are saved with timestamps for versioning
- Feature names are saved for consistency
- Scalers must be saved and loaded with models
- XGBoost uses 0-indexed labels internally (0,1,2)
