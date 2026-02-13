"""
Model Loader - Gestionnaire de modèles pour l'API
Charge et gère les modèles ML et RL en mémoire
"""

import joblib
import torch
from pathlib import Path
from typing import Dict, Optional, Any
from datetime import datetime
import json
import sys

# Ajouter les paths nécessaires
sys.path.append('src/models')
sys.path.append('src/rl')

from dqn_agent import DQNAgent


class ModelLoader:
    """
    Gestionnaire de modèles pour l'API de prédiction
    Charge et maintient les modèles en mémoire
    """
    
    def __init__(self, models_dir: str = "models/saved"):
        """
        Initialiser le loader
        
        Args:
            models_dir: Répertoire racine des modèles sauvegardés
        """
        self.models_dir = Path(models_dir)
        self.ml_dir = self.models_dir  # ← CHANGÉ: Les modèles ML sont directement dans saved/
        self.rl_dir = self.models_dir / "rl"
        
        # Stockage des modèles en mémoire
        self.models: Dict[str, Any] = {}
        self.scalers: Dict[str, Any] = {}
        self.feature_names: Dict[str, list] = {}
        self.metadata: Dict[str, Dict] = {}
        
        # Timestamp de démarrage
        self.start_time = datetime.now()
    
    def load_ml_model(
        self, 
        model_type: str,
        pattern: str = "2022_2023",
        version: str = "latest"
    ) -> bool:
        """
        Charger un modèle ML (LogReg, RF, XGBoost)
        
        Args:
            model_type: Type de modèle (logistic_regression, random_forest, xgboost)
            pattern: Pattern du fichier (ex: 2022_2023)
            version: Version du modèle (latest = le plus récent)
            
        Returns:
            True si chargé avec succès
        """
        try:
            # Trouver les fichiers
            model_pattern = f"{pattern}_{model_type}_*.pkl"
            model_files = list(self.ml_dir.glob(model_pattern))
            
            if not model_files:
                print(f"⚠️ Aucun fichier trouvé pour {model_pattern}")
                return False
            
            # Prendre le plus récent si version=latest
            if version == "latest":
                model_file = sorted(model_files)[-1]
            else:
                model_file = model_files[0]
            
            # Charger le modèle
            model = joblib.load(model_file)
            self.models[model_type] = model
            
            # Charger le scaler
            scaler_pattern = f"{pattern}_scaler_main_*.pkl"
            scaler_files = list(self.ml_dir.glob(scaler_pattern))
            if scaler_files:
                scaler_file = sorted(scaler_files)[-1]
                self.scalers[model_type] = joblib.load(scaler_file)
            
            # Charger les feature names
            features_pattern = f"{pattern}_feature_names_*.pkl"
            features_files = list(self.ml_dir.glob(features_pattern))
            if features_files:
                features_file = sorted(features_files)[-1]
                self.feature_names[model_type] = joblib.load(features_file)
            
            # Charger les métriques
            metrics_pattern = f"{pattern}_metrics_*.pkl"
            metrics_files = list(self.ml_dir.glob(metrics_pattern))
            if metrics_files:
                metrics_file = sorted(metrics_files)[-1]
                metrics = joblib.load(metrics_file)
                self.metadata[model_type] = {
                    'metrics': metrics,
                    'file': str(model_file),
                    'loaded_at': datetime.now().isoformat(),
                    'version': 'v1.0'
                }
            
            print(f"✓ Modèle {model_type} chargé depuis {model_file.name}")
            return True
            
        except Exception as e:
            print(f"✗ Erreur chargement {model_type}: {e}")
            return False
    
    def load_rl_model(
        self,
        model_name: str = "dqn_agent",
        agent_file: str = "best_agent_ep5.pth",  # ← CHANGÉ: Utiliser ep5
        state_size: int = 24,  # ← CHANGÉ: 24 au lieu de 29
        action_size: int = 3
    ) -> bool:
        """
        Charger un agent RL
        
        Args:
            model_name: Nom du modèle (dqn_agent)
            agent_file: Nom du fichier (best_agent.pth, final_agent.pth, etc.)
            state_size: Taille de l'état
            action_size: Nombre d'actions
            
        Returns:
            True si chargé avec succès
        """
        try:
            # Chercher le fichier
            agent_path = self.rl_dir / agent_file
            
            # Si pas trouvé, chercher best_agent_ep*.pth
            if not agent_path.exists():
                pattern = "best_agent_ep*.pth"
                agent_files = list(self.rl_dir.glob(pattern))
                if agent_files:
                    agent_path = sorted(agent_files)[-1]
                else:
                    print(f"⚠️ Aucun agent RL trouvé")
                    return False
            
            # Créer et charger l'agent
            agent = DQNAgent(state_size=state_size, action_size=action_size)
            agent.load(str(agent_path))
            
            self.models[model_name] = agent
            
            # Metadata
            self.metadata[model_name] = {
                'file': str(agent_path),
                'loaded_at': datetime.now().isoformat(),
                'version': 'v1.0',
                'episode_count': agent.episode_count,
                'epsilon': agent.epsilon
            }
            
            print(f"✓ Agent RL chargé depuis {agent_path.name}")
            return True
            
        except Exception as e:
            print(f"✗ Erreur chargement RL: {e}")
            return False
    
    def load_all_models(self) -> Dict[str, bool]:
        """
        Charger tous les modèles disponibles
        
        Returns:
            Dictionnaire {model_name: success}
        """
        results = {}
        
        print("\n" + "="*80)
        print("CHARGEMENT DES MODÈLES")
        print("="*80)
        
        # ML Models
        ml_models = [
            'logistic_regression',
            'random_forest',
            'xgboost'
        ]
        
        for model_type in ml_models:
            results[model_type] = self.load_ml_model(model_type)
        
        # RL Agent
        results['dqn_agent'] = self.load_rl_model()
        
        # Résumé
        loaded_count = sum(results.values())
        total_count = len(results)
        
        print("\n" + "="*80)
        print(f"RÉSUMÉ: {loaded_count}/{total_count} modèles chargés")
        print("="*80 + "\n")
        
        return results
    
    def get_model(self, model_type: str) -> Optional[Any]:
        """Récupérer un modèle chargé"""
        return self.models.get(model_type)
    
    def get_scaler(self, model_type: str) -> Optional[Any]:
        """Récupérer le scaler d'un modèle"""
        return self.scalers.get(model_type)
    
    def get_feature_names(self, model_type: str) -> Optional[list]:
        """Récupérer les noms de features d'un modèle"""
        return self.feature_names.get(model_type)
    
    def is_loaded(self, model_type: str) -> bool:
        """Vérifier si un modèle est chargé"""
        return model_type in self.models
    
    def get_loaded_models(self) -> list:
        """Liste des modèles chargés"""
        return list(self.models.keys())
    
    def get_model_info(self, model_type: str) -> Optional[Dict]:
        """Récupérer les informations d'un modèle"""
        if not self.is_loaded(model_type):
            return None
        
        info = {
            'model_type': model_type,
            'is_loaded': True,
            'version': self.metadata.get(model_type, {}).get('version', 'v1.0'),
            'loaded_at': self.metadata.get(model_type, {}).get('loaded_at'),
        }
        
        # Ajouter métriques si ML
        if model_type in self.metadata and 'metrics' in self.metadata[model_type]:
            metrics = self.metadata[model_type]['metrics']
            if model_type in metrics:
                info['metrics'] = metrics[model_type]
        
        # Ajouter features si disponibles
        if model_type in self.feature_names:
            info['features_required'] = self.feature_names[model_type]
            info['num_features'] = len(self.feature_names[model_type])
        
        return info
    
    def get_all_models_info(self) -> list:
        """Récupérer les infos de tous les modèles chargés"""
        return [
            self.get_model_info(model_type) 
            for model_type in self.get_loaded_models()
        ]
    
    def unload_model(self, model_type: str) -> bool:
        """Décharger un modèle de la mémoire"""
        if model_type in self.models:
            del self.models[model_type]
            if model_type in self.scalers:
                del self.scalers[model_type]
            if model_type in self.feature_names:
                del self.feature_names[model_type]
            if model_type in self.metadata:
                del self.metadata[model_type]
            print(f"✓ Modèle {model_type} déchargé")
            return True
        return False
    
    def get_uptime(self) -> float:
        """Temps de fonctionnement en secondes"""
        return (datetime.now() - self.start_time).total_seconds()
    
    def get_health_status(self) -> Dict:
        """Status de santé du loader"""
        loaded_models = self.get_loaded_models()
        
        status = "healthy" if loaded_models else "unhealthy"
        
        return {
            'status': status,
            'models_loaded': len(loaded_models),
            'loaded_models': loaded_models,
            'uptime_seconds': self.get_uptime(),
            'timestamp': datetime.now().isoformat()
        }


# Singleton global
_model_loader: Optional[ModelLoader] = None


def get_model_loader() -> ModelLoader:
    """
    Récupérer le loader global (singleton)
    Charge les modèles au premier appel
    """
    global _model_loader
    
    if _model_loader is None:
        _model_loader = ModelLoader()
        _model_loader.load_all_models()
    
    return _model_loader


if __name__ == "__main__":
    # Test du loader
    loader = ModelLoader()
    results = loader.load_all_models()
    
    print("\nModèles chargés:")
    for model_type in loader.get_loaded_models():
        info = loader.get_model_info(model_type)
        print(f"  - {model_type}: {info}")
    
    print("\nHealth status:")
    print(loader.get_health_status())