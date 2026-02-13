"""
Model Registry - Gestion des versions de modèles (T11)
Permet de gérer plusieurs versions de modèles ML/RL
"""

import json
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime
import shutil


class ModelRegistry:
    """
    Registry pour gérer les versions de modèles
    Stocke les métadonnées et permet le rollback
    """
    
    def __init__(self, registry_dir: str = "models/registry"):
        """
        Initialiser le registry
        
        Args:
            registry_dir: Répertoire du registry
        """
        self.registry_dir = Path(registry_dir)
        self.registry_dir.mkdir(parents=True, exist_ok=True)
        
        self.registry_file = self.registry_dir / "registry.json"
        self.registry = self._load_registry()
    
    def _load_registry(self) -> Dict:
        """Charger le fichier registry"""
        if self.registry_file.exists():
            with open(self.registry_file, 'r') as f:
                return json.load(f)
        return {}
    
    def _save_registry(self):
        """Sauvegarder le registry"""
        with open(self.registry_file, 'w') as f:
            json.dump(self.registry, f, indent=2)
    
    def register_model(
        self,
        model_type: str,
        version: str,
        file_path: str,
        metrics: Optional[Dict] = None,
        description: str = "",
        author: str = "system"
    ) -> bool:
        """
        Enregistrer un modèle dans le registry
        
        Args:
            model_type: Type de modèle (logistic_regression, xgboost, etc.)
            version: Version (v1.0, v2.0, etc.)
            file_path: Chemin vers le fichier modèle
            metrics: Métriques du modèle
            description: Description du modèle
            author: Auteur du modèle
            
        Returns:
            True si enregistré avec succès
        """
        # Créer l'entrée si elle n'existe pas
        if model_type not in self.registry:
            self.registry[model_type] = {
                'versions': {},
                'latest': None,
                'production': None
            }
        
        # Vérifier que la version n'existe pas déjà
        if version in self.registry[model_type]['versions']:
            print(f"⚠️ Version {version} existe déjà pour {model_type}")
            return False
        
        # Enregistrer la version
        self.registry[model_type]['versions'][version] = {
            'file_path': str(file_path),
            'metrics': metrics or {},
            'description': description,
            'author': author,
            'registered_at': datetime.now().isoformat(),
            'status': 'registered'  # registered, active, deprecated
        }
        
        # Mettre à jour 'latest'
        self.registry[model_type]['latest'] = version
        
        # Si c'est le premier, le mettre en production
        if self.registry[model_type]['production'] is None:
            self.registry[model_type]['production'] = version
        
        self._save_registry()
        
        print(f"✓ Modèle {model_type} v{version} enregistré")
        return True
    
    def get_model_version(
        self,
        model_type: str,
        version: Optional[str] = None
    ) -> Optional[Dict]:
        """
        Récupérer les infos d'une version de modèle
        
        Args:
            model_type: Type de modèle
            version: Version (None = latest)
            
        Returns:
            Dict avec les infos ou None
        """
        if model_type not in self.registry:
            return None
        
        # Par défaut, prendre latest
        if version is None:
            version = self.registry[model_type]['latest']
        
        if version not in self.registry[model_type]['versions']:
            return None
        
        return {
            'model_type': model_type,
            'version': version,
            **self.registry[model_type]['versions'][version]
        }
    
    def list_versions(self, model_type: str) -> List[Dict]:
        """
        Lister toutes les versions d'un modèle
        
        Args:
            model_type: Type de modèle
            
        Returns:
            Liste des versions
        """
        if model_type not in self.registry:
            return []
        
        versions = []
        for version, info in self.registry[model_type]['versions'].items():
            versions.append({
                'version': version,
                'is_latest': version == self.registry[model_type]['latest'],
                'is_production': version == self.registry[model_type]['production'],
                **info
            })
        
        # Trier par date d'enregistrement (plus récent en premier)
        versions.sort(key=lambda x: x['registered_at'], reverse=True)
        
        return versions
    
    def set_production(self, model_type: str, version: str) -> bool:
        """
        Définir une version comme production
        
        Args:
            model_type: Type de modèle
            version: Version à mettre en production
            
        Returns:
            True si succès
        """
        if model_type not in self.registry:
            print(f"⚠️ Modèle {model_type} non trouvé")
            return False
        
        if version not in self.registry[model_type]['versions']:
            print(f"⚠️ Version {version} non trouvée")
            return False
        
        # Ancienne version production
        old_prod = self.registry[model_type]['production']
        
        # Nouvelle version production
        self.registry[model_type]['production'] = version
        
        # Mettre à jour les statuts
        if old_prod:
            self.registry[model_type]['versions'][old_prod]['status'] = 'registered'
        self.registry[model_type]['versions'][version]['status'] = 'active'
        
        self._save_registry()
        
        print(f"✓ {model_type} v{version} est maintenant en production")
        if old_prod:
            print(f"  (ancienne version: v{old_prod})")
        
        return True
    
    def rollback(self, model_type: str) -> Optional[str]:
        """
        Rollback vers la version production précédente
        
        Args:
            model_type: Type de modèle
            
        Returns:
            Version après rollback ou None
        """
        if model_type not in self.registry:
            return None
        
        current_prod = self.registry[model_type]['production']
        
        # Trouver les versions triées par date
        versions = sorted(
            self.registry[model_type]['versions'].items(),
            key=lambda x: x[1]['registered_at'],
            reverse=True
        )
        
        # Trouver la version précédente
        found_current = False
        for version, info in versions:
            if version == current_prod:
                found_current = True
                continue
            
            if found_current:
                # Rollback vers cette version
                self.set_production(model_type, version)
                return version
        
        print(f"⚠️ Pas de version antérieure disponible pour rollback")
        return None
    
    def deprecate_version(self, model_type: str, version: str) -> bool:
        """
        Marquer une version comme deprecated
        
        Args:
            model_type: Type de modèle
            version: Version à déprécier
            
        Returns:
            True si succès
        """
        if model_type not in self.registry:
            return False
        
        if version not in self.registry[model_type]['versions']:
            return False
        
        # Ne pas déprécier la version en production
        if version == self.registry[model_type]['production']:
            print(f"⚠️ Impossible de déprécier la version en production")
            return False
        
        self.registry[model_type]['versions'][version]['status'] = 'deprecated'
        self._save_registry()
        
        print(f"✓ Version {version} de {model_type} marquée comme deprecated")
        return True
    
    def get_production_version(self, model_type: str) -> Optional[str]:
        """Récupérer la version en production"""
        if model_type not in self.registry:
            return None
        return self.registry[model_type]['production']
    
    def get_latest_version(self, model_type: str) -> Optional[str]:
        """Récupérer la dernière version"""
        if model_type not in self.registry:
            return None
        return self.registry[model_type]['latest']
    
    def list_all_models(self) -> List[Dict]:
        """Lister tous les modèles du registry"""
        models = []
        
        for model_type, info in self.registry.items():
            models.append({
                'model_type': model_type,
                'total_versions': len(info['versions']),
                'latest_version': info['latest'],
                'production_version': info['production'],
                'versions': list(info['versions'].keys())
            })
        
        return models
    
    def export_registry(self, output_file: str = None) -> str:
        """
        Exporter le registry en JSON
        
        Args:
            output_file: Fichier de sortie (None = registry.json)
            
        Returns:
            Chemin du fichier exporté
        """
        if output_file is None:
            output_file = self.registry_dir / f"registry_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        output_path = Path(output_file)
        
        with open(output_path, 'w') as f:
            json.dump(self.registry, f, indent=2)
        
        print(f"✓ Registry exporté vers {output_path}")
        return str(output_path)
    
    def import_registry(self, input_file: str) -> bool:
        """
        Importer un registry depuis un fichier JSON
        
        Args:
            input_file: Fichier à importer
            
        Returns:
            True si succès
        """
        input_path = Path(input_file)
        
        if not input_path.exists():
            print(f"⚠️ Fichier {input_file} non trouvé")
            return False
        
        with open(input_path, 'r') as f:
            self.registry = json.load(f)
        
        self._save_registry()
        
        print(f"✓ Registry importé depuis {input_file}")
        return True


# Singleton global
_registry: Optional[ModelRegistry] = None


def get_registry() -> ModelRegistry:
    """Récupérer le registry global (singleton)"""
    global _registry
    
    if _registry is None:
        _registry = ModelRegistry()
    
    return _registry


if __name__ == "__main__":
    # Test du registry
    registry = ModelRegistry()
    
    # Enregistrer un modèle
    registry.register_model(
        model_type="logistic_regression",
        version="v1.0",
        file_path="models/saved/logistic_regression_v1.pkl",
        metrics={'accuracy': 0.89, 'f1': 0.34},
        description="Premier modèle LogReg",
        author="Ivin"
    )
    
    # Enregistrer v2
    registry.register_model(
        model_type="logistic_regression",
        version="v2.0",
        file_path="models/saved/logistic_regression_v2.pkl",
        metrics={'accuracy': 0.91, 'f1': 0.38},
        description="Modèle optimisé",
        author="Ivin"
    )
    
    # Lister les versions
    print("\nVersions de logistic_regression:")
    for v in registry.list_versions("logistic_regression"):
        print(f"  - {v['version']}: {v['description']}")
        if v['is_production']:
            print("    (PRODUCTION)")
    
    # Mettre v2 en production
    registry.set_production("logistic_regression", "v2.0")
    
    # Lister tous les modèles
    print("\nTous les modèles:")
    for m in registry.list_all_models():
        print(f"  - {m['model_type']}: {m['total_versions']} versions")