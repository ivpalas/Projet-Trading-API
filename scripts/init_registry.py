"""
Script d'initialisation du registry (T11)
Enregistre tous les modèles existants dans le registry
"""

from pathlib import Path
import joblib
import sys

# Ajouter le répertoire racine au path
sys.path.append('.')

from api.services.model_registry import ModelRegistry


def init_registry():
    """Initialiser le registry avec les modèles existants"""
    
    print("\n" + "="*80)
    print("INITIALISATION DU MODEL REGISTRY")
    print("="*80 + "\n")
    
    registry = ModelRegistry()
    
    # Chemin des modèles
    models_dir = Path("models/saved")
    
    # ML Models
    ml_models = {
        'logistic_regression': 'Logistic Regression',
        'random_forest': 'Random Forest',
        'xgboost': 'XGBoost'
    }
    
    for model_type, model_name in ml_models.items():
        print(f"📦 Enregistrement de {model_name}...")
        
        # Trouver les fichiers
        pattern = f"2022_2023_{model_type}_*.pkl"
        model_files = list(models_dir.glob(pattern))
        
        if not model_files:
            print(f"   ⚠️ Aucun fichier trouvé pour {pattern}")
            continue
        
        # Prendre le plus récent
        model_file = sorted(model_files)[-1]
        
        # Charger les métriques si disponibles
        metrics_pattern = f"2022_2023_metrics_*.pkl"
        metrics_files = list(models_dir.glob(metrics_pattern))
        
        metrics = {}
        if metrics_files:
            metrics_file = sorted(metrics_files)[-1]
            try:
                all_metrics = joblib.load(metrics_file)
                if model_type in all_metrics:
                    metrics = all_metrics[model_type]
            except Exception as e:
                print(f"   ⚠️ Erreur chargement métriques: {e}")
        
        # Enregistrer v1.0
        success = registry.register_model(
            model_type=model_type,
            version="v1.0",
            file_path=str(model_file),
            metrics=metrics,
            description=f"{model_name} trained on 2022-2023 data",
            author="Ivin Palas"
        )
        
        if success:
            print(f"   ✓ {model_name} v1.0 enregistré")
            
            # Mettre en production
            registry.set_production(model_type, "v1.0")
    
    # RL Agent
    print(f"\n📦 Enregistrement de DQN Agent...")
    
    rl_dir = models_dir / "rl"
    agent_files = list(rl_dir.glob("best_agent_ep*.pth"))
    
    if agent_files:
        agent_file = sorted(agent_files)[0]  # Premier agent (ep5)
        
        success = registry.register_model(
            model_type="dqn_agent",
            version="v1.0",
            file_path=str(agent_file),
            metrics={'episode': 5, 'state_size': 24, 'action_size': 3},
            description="DQN Agent trained on 2022-2023 (20 episodes)",
            author="Ivin Palas"
        )
        
        if success:
            print(f"   ✓ DQN Agent v1.0 enregistré")
            registry.set_production("dqn_agent", "v1.0")
    else:
        print(f"   ⚠️ Aucun agent RL trouvé")
    
    # Résumé
    print("\n" + "="*80)
    print("RÉSUMÉ DU REGISTRY")
    print("="*80 + "\n")
    
    all_models = registry.list_all_models()
    
    for model in all_models:
        print(f"✓ {model['model_type']}")
        print(f"  Versions: {model['total_versions']}")
        print(f"  Latest: {model['latest_version']}")
        print(f"  Production: {model['production_version']}\n")
    
    print(f"Total: {len(all_models)} modèles enregistrés")
    print("="*80 + "\n")
    
    # Exporter un backup
    backup_file = registry.export_registry()
    print(f"✓ Backup créé: {backup_file}\n")


if __name__ == "__main__":
    init_registry()