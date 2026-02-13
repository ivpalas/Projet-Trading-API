# T08 - Agent de Trading par Reinforcement Learning

##  Vue d'ensemble

Implémentation d'un agent **Deep Q-Network (DQN)** pour le trading GBP/USD utilisant le Reinforcement Learning.

### Composants :

1. **`trading_env.py`** - Environnement Gymnasium personnalisé
2. **`dqn_agent.py`** - Agent DQN avec experience replay
3. **`train_rl.py`** - Script d'entraînement
4. **`rl_backtester.py`** - Script de backtesting

---

##  Environnement de Trading

### Espace d'états
- **Features de prix** : OHLC, returns, volatilité
- **Indicateurs techniques** : RSI, MACD, EMAs, ADX, ATR, etc.
- **Info position** : Position actuelle, P&L non réalisé, balance

### Espace d'actions
- **0** : HOLD (ne rien faire)
- **1** : BUY (position longue)
- **2** : SELL (position courte / fermer position longue)

### Fonction de récompense
```
reward = profit/perte des trades
       - coûts de transaction
       + 0.1 × P&L non réalisé (encourage à garder positions profitables)
```

---

##  Agent DQN

### Architecture
```
Input (état) 
  ↓
Dense(128) + ReLU + Dropout(0.2)
  ↓
Dense(64) + ReLU + Dropout(0.2)
  ↓
Output(3) → Q-values pour [HOLD, BUY, SELL]
```

### Fonctionnalités clés
- ✅ **Experience Replay** : Buffer de 10,000 expériences
- ✅ **Target Network** : Stabilise l'entraînement (mis à jour tous les 10 épisodes)
- ✅ **Epsilon-Greedy** : Exploration vs exploitation
- ✅ **Gradient Clipping** : Prévient l'explosion des gradients
- ✅ **Support GPU** : CPU ou CUDA (GPU)

---

##  Utilisation

### 1. Installer les dépendances

```bash
pip install -r requirements_rl.txt
```

### 2. Entraîner l'agent

```bash
python src/rl/train_rl.py
```

**Paramètres d'entraînement** :
- Épisodes : 200
- Validation : Tous les 10 épisodes
- Sauvegarde : Tous les 50 épisodes
- Données : Entraînement sur 2022, validation sur 2023

**Sorties** :
- `models/saved/rl/best_agent_epXX.pth` - Meilleur modèle
- `models/saved/rl/final_agent.pth` - Modèle final
- `models/saved/rl/training_history.json` - Métriques d'entraînement
- `models/saved/rl/training_progress.png` - Graphiques

### 3. Backtester l'agent

```bash
python src/rl/rl_backtester.py
```

**Test** : Performance sur données 2024

**Sorties** :
- Métriques de performance (return, trades, win rate)
- Distribution des actions
- Visualisations
- `backtest_results.csv`

---

##  Résultats attendus

### Entraînement (2022)
- Return moyen : ~20-50%
- Win Rate : ~50-60%
- Epsilon decay : 1.0 → 0.01

### Validation (2023)
- Return moyen : ~10-30%
- Win Rate : ~45-55%

### Test (2024)
- Objectif : Battre T07 baseline (+297%)
- Attendu : +50-150% (dépend de l'entraînement)

---

##  Hyperparamètres

### Agent
```python
learning_rate = 0.001
gamma = 0.99  # Facteur de discount
epsilon_start = 1.0
epsilon_end = 0.01
epsilon_decay = 0.995
buffer_size = 10000
batch_size = 64
```

### Environnement
```python
initial_balance = 10000
transaction_cost = 0.0001  # 1 pip de spread
max_position = 1.0  # 100% du capital
```

---

##  Suivi de l'entraînement

Le script d'entraînement fournit :
- **Barre de progression** (tqdm)
- **Stats par épisode** : Return, balance, trades, win rate
- **Validation** : Tous les 10 épisodes
- **Graphiques** : Returns, balance, epsilon, win rate

---

##  Utilisation avancée

### Charger et continuer l'entraînement

```python
from dqn_agent import DQNAgent

# Charger l'agent
agent = DQNAgent(state_size=29, action_size=3)
agent.load("models/saved/rl/best_agent_ep100.pth")

# Continuer l'entraînement
# ... (utiliser avec RLTrainer)
```

### Fonction de récompense personnalisée

Modifier `trading_env.py`, méthode `_execute_action()` :

```python
# Exemple : Ajouter pénalité pour trading excessif
if action != 0:  # Pas HOLD
    reward -= 0.01  # Petite pénalité pour chaque trade
```

### Optimisation hyperparamètres

Modifier `train_rl.py` :

```python
agent = DQNAgent(
    state_size=state_size,
    action_size=action_size,
    learning_rate=0.0005,  # ← Changer ici
    gamma=0.95,            # ← Et ici
    # ...
)
```

---

##  Dépannage

### CUDA Out of Memory
```python
# Utiliser CPU
agent = DQNAgent(..., device='cpu')
```

### L'agent n'apprend pas
- Vérifier fonction de récompense (récompenses positives ?)
- Augmenter nombre d'épisodes
- Ajuster learning rate
- Vérifier epsilon decay (trop rapide ?)

### Mauvaise performance en validation
- Overfitting → Augmenter dropout, réduire taille modèle
- Underfitting → Augmenter taille modèle, plus d'épisodes
- Ajouter régularisation

---