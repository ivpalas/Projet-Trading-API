# 🐳 Guide de Déploiement - Projet Trading GBP/USD

Ce guide explique comment déployer l'application de trading avec Docker.

---

## 📋 Prérequis

- **Docker** : Version 20.10+
- **Docker Compose** : Version 2.0+
- **Git** : Pour cloner le repository

### Installation Docker

**Windows/Mac** :
- Télécharger [Docker Desktop](https://www.docker.com/products/docker-desktop)

**Linux** :
```bash
curl -fsSL https://get.docker.com -o get-docker.sh
sh get-docker.sh
```

---

## 🚀 Déploiement Rapide

### 1. Cloner le repository

```bash
git clone <votre-repo>
cd Projet
```

### 2. Configuration (optionnel)

Créer un fichier `.env` pour les variables d'environnement :

```bash
# .env
ENVIRONMENT=production
LOG_LEVEL=info
API_HOST=0.0.0.0
API_PORT=8000
```

### 3. Build et lancement

```bash
# Build l'image Docker
docker-compose build

# Lancer les services
docker-compose up -d

# Vérifier les logs
docker-compose logs -f api
```

### 4. Vérification

L'API est accessible sur : **http://localhost:8000**

- Documentation : http://localhost:8000/docs
- Health check : http://localhost:8000/health

---

## 📦 Commandes Docker

### Gestion des conteneurs

```bash
# Démarrer
docker-compose up -d

# Arrêter
docker-compose down

# Redémarrer
docker-compose restart

# Voir les logs
docker-compose logs -f api

# Voir le statut
docker-compose ps
```

### Build et mise à jour

```bash
# Rebuild après modification du code
docker-compose build --no-cache

# Rebuild et redémarrer
docker-compose up -d --build

# Nettoyer les images inutilisées
docker system prune -a
```

### Accéder au conteneur

```bash
# Shell interactif
docker-compose exec api bash

# Exécuter une commande
docker-compose exec api python scripts/init_registry.py
```

---

## 🔧 Configuration Avancée

### Variables d'environnement

Modifier `docker-compose.yml` :

```yaml
environment:
  - ENVIRONMENT=production
  - LOG_LEVEL=debug  # debug, info, warning, error
  - API_HOST=0.0.0.0
  - API_PORT=8000
```

### Volumes

Les volumes permettent de persister les données :

```yaml
volumes:
  # Données (lecture seule)
  - ./data:/app/data:ro
  # Modèles (lecture/écriture)
  - ./models:/app/models
  # Logs
  - ./logs:/app/logs
```

### Ports

Changer le port exposé :

```yaml
ports:
  - "8080:8000"  # Accès via localhost:8080
```

---

## 🌐 Déploiement Production

### Option 1 : VPS (Serveur dédié)

**Prérequis** :
- Serveur Linux (Ubuntu 20.04+)
- Accès SSH
- Nom de domaine (optionnel)

**Étapes** :

```bash
# 1. Se connecter au serveur
ssh user@votre-serveur.com

# 2. Installer Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sh get-docker.sh

# 3. Cloner le projet
git clone <votre-repo>
cd Projet

# 4. Lancer avec Docker Compose
docker-compose up -d

# 5. Configurer firewall (si nécessaire)
sudo ufw allow 8000/tcp
```

### Option 2 : Cloud (AWS, GCP, Azure)

**AWS (EC2)** :
1. Créer une instance EC2 (Ubuntu)
2. Installer Docker
3. Cloner et lancer avec docker-compose

**Google Cloud Run** :
```bash
# Build et push l'image
docker build -t gcr.io/PROJECT-ID/trading-api .
docker push gcr.io/PROJECT-ID/trading-api

# Déployer
gcloud run deploy trading-api \
  --image gcr.io/PROJECT-ID/trading-api \
  --platform managed \
  --region europe-west1 \
  --allow-unauthenticated
```

**Azure Container Instances** :
```bash
# Login
az login

# Créer container
az container create \
  --resource-group myResourceGroup \
  --name trading-api \
  --image trading-api:latest \
  --dns-name-label trading-api \
  --ports 8000
```

### Option 3 : Kubernetes (production avancée)

Créer `deployment.yaml` :

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: trading-api
spec:
  replicas: 3
  selector:
    matchLabels:
      app: trading-api
  template:
    metadata:
      labels:
        app: trading-api
    spec:
      containers:
      - name: api
        image: trading-api:latest
        ports:
        - containerPort: 8000
```

---

## 🔐 Sécurité

### HTTPS avec Let's Encrypt

**Ajouter Nginx + Certbot** :

```yaml
# docker-compose.yml
services:
  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
      - ./certbot/conf:/etc/letsencrypt
      - ./certbot/www:/var/www/certbot
  
  certbot:
    image: certbot/certbot
    volumes:
      - ./certbot/conf:/etc/letsencrypt
      - ./certbot/www:/var/www/certbot
```

### Variables sensibles

**Ne JAMAIS** commiter :
- Clés API
- Mots de passe
- Tokens

Utiliser des **secrets** ou **variables d'environnement**.

---

## 📊 Monitoring

### Logs

```bash
# Logs en temps réel
docker-compose logs -f api

# Logs avec timestamp
docker-compose logs -t api

# Dernières 100 lignes
docker-compose logs --tail=100 api
```

### Health Check

```bash
# Vérifier manuellement
curl http://localhost:8000/health

# Health check automatique (déjà configuré)
docker-compose ps  # Montre le statut "healthy"
```

### Métriques (optionnel)

Ajouter **Prometheus + Grafana** :

```yaml
services:
  prometheus:
    image: prom/prometheus
    ports:
      - "9090:9090"
  
  grafana:
    image: grafana/grafana
    ports:
      - "3000:3000"
```

---

## 🐛 Dépannage

### Problème : Conteneur ne démarre pas

```bash
# Voir les logs d'erreur
docker-compose logs api

# Vérifier les ports
sudo netstat -tulpn | grep 8000

# Rebuild complet
docker-compose down
docker-compose build --no-cache
docker-compose up -d
```

### Problème : Permission denied

```bash
# Donner les droits
chmod -R 755 models/ data/ logs/

# Ou exécuter avec sudo
sudo docker-compose up -d
```

### Problème : Out of memory

```bash
# Limiter la RAM dans docker-compose.yml
services:
  api:
    deploy:
      resources:
        limits:
          memory: 2G
```

### Problème : Modèles non chargés

```bash
# Initialiser le registry
docker-compose exec api python scripts/init_registry.py

# Vérifier les volumes
docker-compose exec api ls -la models/saved/
```

---

## 🔄 Mise à jour

### Mettre à jour le code

```bash
# 1. Pull les dernières modifications
git pull origin main

# 2. Rebuild et redémarrer
docker-compose up -d --build

# 3. Vérifier
docker-compose ps
docker-compose logs -f api
```

### Rollback

```bash
# 1. Revenir à une version précédente
git checkout <commit-hash>

# 2. Rebuild
docker-compose up -d --build
```

---

## 📚 Ressources

- **Docker Documentation** : https://docs.docker.com/
- **Docker Compose** : https://docs.docker.com/compose/
- **FastAPI Deployment** : https://fastapi.tiangolo.com/deployment/

---

## ✅ Checklist Déploiement

- [ ] Docker et Docker Compose installés
- [ ] Repository cloné
- [ ] Variables d'environnement configurées
- [ ] `docker-compose build` réussi
- [ ] `docker-compose up -d` lancé
- [ ] API accessible sur http://localhost:8000
- [ ] Health check OK (http://localhost:8000/health)
- [ ] Registry initialisé (`init_registry.py`)
- [ ] Tests API dans `/docs`
- [ ] Logs vérifiés
- [ ] Firewall configuré (si production)
- [ ] HTTPS configuré (si production)
- [ ] Monitoring en place (optionnel)

---

## 🆘 Support

En cas de problème, vérifier :
1. Les logs : `docker-compose logs -f api`
2. Le statut : `docker-compose ps`
3. La documentation API : http://localhost:8000/docs

---

**Projet déployé avec succès !** 🎉
