# 📊 Projet de Trading GBP/USD avec ML et RL

Système de trading algorithmique complet pour la paire GBP/USD utilisant Machine Learning et Reinforcement Learning.

---

## 🎯 Objectif

Développer un système de trading automatisé capable de :
- Analyser les données de marché GBP/USD (timeframe 15 minutes)
- Générer des signaux de trading avec ML et RL
- Backtester les stratégies
- Optimiser les performances

---

## 🏆 Résultats

### Meilleure Performance : **+297% de return** (Logistic Regression - T07)

| Modèle | Return | Trades | Win Rate | Année Test |
|--------|--------|--------|----------|------------|
| **Logistic Regression** | **+297.54%** | 10 | 40% | 2024 |
| Random Forest | 0% | 0 | - | 2024 |
| XGBoost | -0.14% | 2 | 0% | 2024 |
| DQN Agent (RL) | Variable | Variable | ~50% | 2024 |

---

## 📂 Structure du Projet

```
Projet/
├── api/                      # API FastAPI
│   ├── routers/
│   │   ├── data.py          # T01-T04 : Données
│   │   ├── features.py      # T05 : Features techniques
│   │   └── strategies.py    # T06 : Stratégies baseline
│   ├── services/
│   └── main.py
├── src/
│   ├── models/              # T07 : Machine Learning
│   │   ├── feature_engineering.py
│   │   ├── ml_trainer.py
│   │   ├── ml_backtester.py
│   │   └── run_ml_pipeline.py
│   └── rl/                  # T08 : Reinforcement Learning
│       ├── trading_env.py
│       ├── dqn_agent.py
│       ├── train_rl.py
│       └── rl_backtester.py
├── data/
│   ├── raw/                 # Données brutes (.csv)
│   └── processed/           # Données traitées (.parquet)
├── models/saved/            # Modèles entraînés
│   ├── ml/                  # Modèles ML
│   └── rl/                  # Agents RL
├── notebooks/               # Jupyter notebooks
│   ├── 01_data_pipeline.ipynb
│   ├── 02_ml_models.ipynb
│   └── 03_reinforcement_learning.ipynb
├── requirements.txt         # Dépendances Python
└── README.md               # Ce fichier
```

---

## 🚀 Installation

### Prérequis

- Python 3.10+
- pip
- Git

### Installation rapide

```bash
# Cloner le repository
git clone <votre-repo>
cd Projet

# Installer les dépendances
pip install -r requirements.txt

# Vérifier l'installation
python -c "import pandas, numpy, sklearn, torch; print('✓ Installation réussie')"
```

---

## 📖 Utilisation

### 1️⃣ API FastAPI (T05-T06)

```bash
# Démarrer l'API
uvicorn api.main:app --reload

# Accéder à la documentation
# http://localhost:8000/docs
```

**Endpoints disponibles** :
- `POST /features/compute/{year}` - Calculer les features techniques
- `POST /strategies/backtest/{strategy}` - Backtester une stratégie

### 2️⃣ Machine Learning (T07)

```bash
# Pipeline complet (création datasets + entraînement + backtesting)
python src/models/run_ml_pipeline.py

# Fichiers générés :
# - data/processed/ml_dataset_*.parquet
# - models/saved/*.pkl
```

**Résultats attendus** :
- Logistic Regression : +297% return sur 2024
- Random Forest : 0% (trop conservateur)
- XGBoost : -0.14% (peu actif)

### 3️⃣ Reinforcement Learning (T08)

```bash
# Entraîner l'agent DQN (20 épisodes pour test rapide)
python src/rl/train_rl.py

# Backtester l'agent entraîné
python src/rl/rl_backtester.py
```

**Configuration** :
- Environnement : Gym custom
- Agent : DQN (Deep Q-Network)
- Actions : HOLD, BUY, SELL
- Training : 2022, Validation : 2023, Test : 2024

### 4️⃣ Notebooks Jupyter

```bash
# Lancer Jupyter
jupyter notebook

# Ouvrir les notebooks dans notebooks/
```

**Notebooks disponibles** :
- `01_data_pipeline.ipynb` - Exploration des données (T01-T04)
- `02_ml_models.ipynb` - Machine Learning (T07)
- `03_reinforcement_learning.ipynb` - RL Analysis (T08)

---

## 📊 Pipeline de Données

### T01-T04 : Data Pipeline

1. **T01** : Chargement données brutes (CSV)
2. **T02** : Agrégation en bougies M15
3. **T03** : Nettoyage et validation
4. **T04** : Contrôle qualité

**Données disponibles** : 2022, 2023, 2024 (format M15 - 15 minutes)

### T05 : Feature Engineering

**20+ indicateurs techniques** calculés via l'API :
- Prix : returns, volatility, body, wicks
- Trend : EMA (20, 50, 200), slope
- Momentum : RSI, MACD, ADX
- Volatility : ATR, rolling std

### T06 : Baseline Strategies

**3 stratégies de référence** :
- Buy & Hold : -10.21% (2022)
- Random Trading : -7.99% (2022)
- SMA Crossover : -5.67% (2022)

---

## 🤖 Machine Learning (T07)

### Approche

- **Target** : Classification 3 classes (UP, HOLD, DOWN)
- **Threshold** : 0.1% (10 pips)
- **Features** : 100+ features (lag, rolling, technical indicators)
- **Train** : 2022-2023
- **Test** : 2024

### Modèles

1. **Logistic Regression** ⭐
   - Return : +297.54%
   - Trades : 10
   - Win Rate : 40%
   - **Meilleur modèle !**

2. **Random Forest**
   - Return : 0%
   - Trades : 0
   - Trop conservateur

3. **XGBoost**
   - Return : -0.14%
   - Trades : 2
   - Peu actif

### Fichiers générés

```
models/saved/
├── 2022_2023_logistic_regression_*.pkl
├── 2022_2023_random_forest_*.pkl
├── 2022_2023_xgboost_*.pkl
├── 2022_2023_scaler_main_*.pkl
├── 2022_2023_feature_names_*.pkl
└── 2022_2023_metrics_*.pkl
```

---

## 🎮 Reinforcement Learning (T08)

### Agent DQN

**Architecture** :
```
State (features + position) 
  ↓
Dense(128) + ReLU + Dropout(0.2)
  ↓
Dense(64) + ReLU + Dropout(0.2)
  ↓
Output(3) → Q-values [HOLD, BUY, SELL]
```

**Hyperparamètres** :
- Learning rate : 0.001
- Gamma : 0.99
- Epsilon decay : 0.995
- Buffer size : 10,000
- Batch size : 64

### Entraînement

- **Train** : 2022 (24,814 périodes)
- **Validation** : 2023 (21,450 périodes)
- **Test** : 2024 (24,831 périodes)
- **Épisodes** : 20-200 (configurable)

### Résultats

Performance variable selon hyperparamètres et reward function.
L'agent DQN nécessite optimisation pour battre les modèles ML.

---

## 📈 Comparaison des Approches

| Critère | ML (T07) | RL (T08) |
|---------|----------|----------|
| **Meilleur Return** | +297% (LogReg) | Variable |
| **Trades** | 10 (sélectif) | Variable |
| **Temps training** | ~5 min | ~30 min |
| **Complexité** | Moyenne | Élevée |
| **Stabilité** | ✅ Stable | ⚠️ Variable |
| **Adaptabilité** | ❌ Statique | ✅ Apprend en continu |

**Recommandation actuelle** : Logistic Regression (T07) pour production

---

## 🔧 Configuration

### Variables d'environnement (optionnel)

Créer un fichier `.env` :

```env
# API
API_HOST=0.0.0.0
API_PORT=8000

# Données
DATA_PATH=data/processed

# Modèles
MODELS_PATH=models/saved
```

### Fichiers de données requis

```
data/processed/
├── m15_2022.csv
├── m15_2023.csv
└── m15_2024.csv
```

---

## 🐛 Dépannage

### Problèmes courants

**1. Import Error : "No module named 'gymnasium'"**
```bash
pip install gymnasium torch
```

**2. PyArrow Error (lecture parquet)**
```bash
pip install pyarrow
```

**3. API ne démarre pas**
```bash
# Vérifier que vous êtes à la racine du projet
cd Projet
uvicorn api.main:app --reload
```

**4. CUDA Out of Memory (RL)**
```bash
# Utiliser CPU au lieu de GPU
# Dans train_rl.py, ligne 60 :
device='cpu'
```

---

## 📚 Documentation Détaillée

Chaque tâche (T01-T08) possède sa propre documentation :

- **T01-T06** : Voir documentation API (`/docs`)
- **T07** : Voir `README_T07.md` (à créer si besoin)
- **T08** : Voir `README_T08.md`

---

## 🎓 Ressources

### Articles de référence

- **DQN** : [Mnih et al., 2015](https://www.nature.com/articles/nature14236)
- **Feature Engineering** : Indicateurs techniques standards

### Technologies utilisées

- **Backend** : FastAPI, Pandas, NumPy
- **ML** : scikit-learn, XGBoost
- **RL** : PyTorch, Gymnasium
- **Visualisation** : Matplotlib, Seaborn
- **Data** : Parquet, CSV

---

## 🤝 Contribution

Ce projet est un projet éducatif de trading algorithmique.

### Améliorations possibles

- [ ] Agents RL avancés (PPO, A2C)
- [ ] Optimisation hyperparamètres (Grid Search)
- [ ] Feature selection automatique
- [ ] Dashboard Streamlit
- [ ] Déploiement Docker
- [ ] Trading multi-actifs
- [ ] Gestion de portefeuille

---

## ⚠️ Disclaimer

**Ce projet est à but éducatif uniquement.**

- ❌ Ne constitue PAS un conseil financier
- ❌ Trading réel à vos risques et périls
- ❌ Performances passées ne garantissent pas les futures
- ✅ Utilisez un compte démo pour tester

---

## 📝 Licence

Projet éducatif - Tous droits réservés

---

## 📧 Contact

Pour toute question sur le projet, veuillez créer une issue sur GitHub.

---

## ✅ Statut du Projet

- [x] T01 - Chargement données
- [x] T02 - Agrégation M15
- [x] T03 - Nettoyage
- [x] T04 - Qualité
- [x] T05 - Features API
- [x] T06 - Baseline strategies
- [x] T07 - Machine Learning (+297% !)
- [x] T08 - Reinforcement Learning
- [x] T09 - Production & Documentation

**Projet complété !** 🎉
