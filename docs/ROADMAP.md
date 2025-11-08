# 🗺️ Video Summarizer - Roadmap

## Vision
Transformer Video Summarizer en une plateforme scalable et cloud-native avec support de modèles LLM locaux (Ollama) et déploiement sur Google Cloud Platform.

---

## 📋 Vue d'ensemble

### Objectifs Principaux
1. ✅ Intégrer Ollama comme alternative performante à LED
2. 🚀 Déployer l'API sur Google Cloud Platform
3. 📊 Améliorer le monitoring et l'observabilité
4. 🔒 Sécuriser et optimiser l'infrastructure

---

## 🎯 Phase 1 : Intégration Ollama (Semaines 1-2)

### Sprint 1 : Développement Module Ollama (3-5 jours)

#### ✅ Tâches
- [ ] **Créer `src/models/ollama_model.py`**
  - Classe `OllamaSummarizer`
  - Connexion API Ollama (`http://localhost:11434`)
  - Gestion des modèles disponibles
  - Méthode `summarize()` avec prompts optimisés
  - Gestion d'erreurs et timeouts
  
- [ ] **Mettre à jour `src/models/model_manager.py`**
  - Ajouter `ModelType.OLLAMA` à l'enum
  - Intégrer Ollama dans le gestionnaire
  - Système de fallback Ollama → OpenAI → LED
  - Recommandation automatique de modèle
  
- [ ] **Configuration**
  - Ajouter section `ollama` dans `config/model_config.yaml`
  - Paramètres : base_url, model_name, temperature, max_tokens
  - Support multi-modèles (Mistral, Llama 3.1, Gemma, Qwen)

- [ ] **Tests**
  - Tests unitaires pour `OllamaSummarizer`
  - Tests d'intégration avec `ModelManager`
  - Test de disponibilité et fallback
  - Benchmarks de performance

- [ ] **Comparaison des modèles**
  - Tester Mistral 7B
  - Tester Llama 3.1 8B
  - Tester Gemma 2 9B
  - Tester Qwen 2.5 7B
  - Comparer qualité, vitesse, consommation mémoire

- [ ] **Mise à jour UI Streamlit**
  - Ajouter Ollama dans les options de modèle
  - Afficher modèles Ollama disponibles
  - Indicateur de disponibilité en temps réel

#### 📊 Modèles Recommandés

| Modèle | Taille | Vitesse | Qualité | RAM | Use Case |
|--------|--------|---------|---------|-----|----------|
| **Mistral 7B** | 4.1GB | ⚡⚡⚡ | ★★★★☆ | 8GB | Général, équilibré |
| **Llama 3.1 8B** | 4.7GB | ⚡⚡⚡ | ★★★★★ | 8GB | Multilingue, précis |
| **Gemma 2 9B** | 5.4GB | ⚡⚡☆ | ★★★★★ | 12GB | Compréhension avancée |
| **Qwen 2.5 7B** | 4.4GB | ⚡⚡⚡ | ★★★★☆ | 8GB | Résumés longs |

#### 🎯 Critères de Succès
- Ollama intégré et fonctionnel
- Temps de génération < 30s pour texte moyen (500 mots)
- Score d'évaluation ≥ 0.75
- Fallback automatique opérationnel
- Documentation complète

---

## 🚀 Phase 2 : Amélioration API (Semaines 2-3)

### Sprint 2 : API Enhancement (2-3 jours)

#### ✅ Tâches
- [ ] **Nouveaux Endpoints**
  - `GET /api/v1/models/list` - Liste des modèles disponibles
  - `GET /api/v1/models/{model_id}/info` - Infos sur un modèle
  - `POST /api/v1/summarize/stream` - Streaming de résumé
  - `GET /api/v1/health/detailed` - Health check détaillé
  
- [ ] **Sécurité**
  - Authentification JWT
  - API Keys avec rate limiting
  - CORS configuré correctement
  - Validation des inputs avec Pydantic
  - Protection CSRF

- [ ] **Performance**
  - Caching avec Redis (optionnel)
  - Compression des réponses (gzip)
  - Pagination pour endpoints de listing
  - Batch processing optimisé

- [ ] **Documentation**
  - OpenAPI/Swagger complet
  - Exemples pour chaque endpoint
  - Guide d'authentification
  - Rate limits documentés

#### 🎯 Critères de Succès
- Documentation OpenAPI complète
- Rate limiting fonctionnel (100 req/min)
- Temps de réponse API < 5s (hors génération)
- Tests de charge réussis (50 req/s)

---

## 🐳 Phase 3 : Containerisation (Semaines 3-4)

### Sprint 3 : Docker & Compose (2 jours)

#### ✅ Tâches
- [ ] **Dockerfile optimisé**
  - Multi-stage build
  - Image Python 3.11-slim
  - Installation Ollama
  - Pré-chargement modèles (optionnel)
  - Taille cible < 3GB

- [ ] **Docker Compose**
  - Service API (FastAPI)
  - Service Ollama
  - PostgreSQL pour métriques
  - Redis pour cache (optionnel)
  - Nginx comme reverse proxy

- [ ] **Configuration**
  - Variables d'environnement (.env)
  - Secrets management
  - Volumes pour persistance
  - Networks isolés

- [ ] **Tests**
  - Build automatisé
  - Tests d'intégration dans containers
  - Health checks
  - Performance tests

#### 📁 Structure Docker
```
docker/
├── Dockerfile.api
├── Dockerfile.ollama
├── docker-compose.yml
├── docker-compose.prod.yml
├── nginx.conf
└── .dockerignore
```

#### 🎯 Critères de Succès
- Build < 5 minutes
- Image finale < 3GB
- Démarrage < 2 minutes
- Tous les services communicants
- Tests d'intégration passent

---

## ☁️ Phase 4 : Google Cloud Platform Setup (Semaines 4-5)

### Sprint 4 : Infrastructure GCP (3-4 jours)

#### ✅ Tâches
- [ ] **Projet GCP**
  - Créer projet `video-summarizer-prod`
  - Configurer billing
  - Activer APIs nécessaires
  - Setup IAM et permissions

- [ ] **Cloud Storage**
  - Bucket pour modèles
  - Bucket pour assets statiques
  - Bucket pour backups
  - Lifecycle policies

- [ ] **Cloud SQL**
  - Instance PostgreSQL 15
  - Configuration HA (optionnel)
  - Backups automatiques
  - Connexion sécurisée

- [ ] **Secret Manager**
  - API keys
  - Database credentials
  - JWT secrets
  - Configurations sensibles

- [ ] **Container Registry**
  - Setup Artifact Registry
  - Policies de retention
  - Scanning de vulnérabilités

#### 🏗️ Architecture GCP

```
┌─────────────────────────────────────────────┐
│         Cloud Load Balancing                │
│         (HTTPS, SSL Certificate)            │
└──────────────────┬──────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────┐
│           Cloud Run Service                 │
│    (Auto-scaling, Serverless)               │
│                                              │
│  ┌────────────────────────────────────┐    │
│  │   FastAPI + Ollama Container       │    │
│  │   - CPU: 2 vCPU                    │    │
│  │   - RAM: 4-8 GB                    │    │
│  │   - Concurrency: 10                │    │
│  └────────────────────────────────────┘    │
└────┬──────────────────────┬─────────────────┘
     │                      │
     ▼                      ▼
┌─────────────┐      ┌──────────────┐
│  Cloud SQL  │      │Cloud Storage │
│ (PostgreSQL)│      │  (Models &   │
│  (Metrics)  │      │   Assets)    │
└─────────────┘      └──────────────┘
```

#### 🎯 Critères de Succès
- Infrastructure provisionnée avec Terraform
- Cloud SQL opérationnel
- Storage buckets configurés
- Secrets sécurisés
- IAM correctement configuré

---

## 🚀 Phase 5 : Déploiement Production (Semaines 5-6)

### Sprint 5 : Cloud Run Deployment (2-3 jours)

#### ✅ Tâches
- [ ] **CI/CD avec Cloud Build**
  - `cloudbuild.yaml`
  - Build automatique sur push
  - Tests avant déploiement
  - Déploiement automatique

- [ ] **Déploiement Cloud Run**
  - Service configuration
  - Scaling rules
  - Health checks
  - Traffic splitting (blue/green)

- [ ] **Load Balancer**
  - HTTPS avec SSL
  - Backend services
  - Health checks
  - CDN activation (optionnel)

- [ ] **Monitoring**
  - Cloud Monitoring dashboards
  - Alertes (CPU, RAM, erreurs)
  - Logs centralisés
  - Traces distribuées

- [ ] **Tests de Charge**
  - Tests avec Locust/K6
  - 100 req/s pendant 5 min
  - Latency < 2s (P95)
  - Error rate < 1%

#### 📊 Métriques Cibles

| Métrique | Cible | Critique |
|----------|-------|----------|
| Uptime | 99.5% | 99% |
| Latency P95 | < 2s | < 5s |
| Error Rate | < 1% | < 5% |
| Requests/sec | 50+ | 20+ |
| CPU Usage | < 70% | < 90% |
| Memory Usage | < 80% | < 95% |

#### 🎯 Critères de Succès
- Déploiement automatisé fonctionnel
- Load balancer configuré
- HTTPS avec certificat valide
- Monitoring opérationnel
- Tests de charge réussis

---

## ⚡ Phase 6 : Optimisations (Semaines 6-7)

### Sprint 6 : Performance & Scalability (2-3 jours)

#### ✅ Tâches
- [ ] **Caching**
  - Memorystore (Redis) pour résumés
  - TTL configuré (1h - 24h)
  - Cache invalidation
  - Hit rate > 40%

- [ ] **CDN**
  - Cloud CDN pour assets
  - Cache-Control headers
  - Compression activée
  - Edge locations

- [ ] **Database Optimization**
  - Index optimisés
  - Connection pooling
  - Query optimization
  - Partitionnement (si nécessaire)

- [ ] **Auto-scaling**
  - Min instances: 1
  - Max instances: 10
  - Scale up: CPU > 70%
  - Scale down: CPU < 30%

- [ ] **Security Hardening**
  - WAF (Cloud Armor)
  - DDoS protection
  - Rate limiting par IP
  - Input sanitization

#### 🎯 Critères de Succès
- Cache hit rate > 40%
- Temps de réponse réduit de 30%
- Auto-scaling testé et validé
- Sécurité renforcée
- Coûts optimisés

---

## 📊 Phases Futures (Mois 2-3)

### Phase 7 : Features Avancées
- [ ] Intégration Vertex AI (modèles Google)
- [ ] Support multi-tenancy
- [ ] Analytics avancés
- [ ] A/B testing des modèles
- [ ] WebSocket pour streaming temps réel
- [ ] Support fichiers audio (Whisper)
- [ ] Interface mobile (React Native)

### Phase 8 : Scaling Enterprise
- [ ] Migration GKE avec GPU
- [ ] Multi-région (HA globale)
- [ ] Disaster recovery
- [ ] Compliance (GDPR, SOC2)
- [ ] SLA 99.9%
- [ ] Support client

---

## 💰 Estimation Budgétaire

### Cloud Run (Recommandé pour démarrage)
| Service | Coût/mois | Notes |
|---------|-----------|-------|
| Cloud Run | $15-25 | 1000 req/jour |
| Cloud SQL (f1-micro) | $10 | PostgreSQL |
| Cloud Storage | $2-5 | Models + assets |
| Load Balancer | $5 | HTTPS |
| Monitoring | Gratuit | Quota de base |
| **Total** | **$32-45** | **Budget de démarrage** |

### Cloud Run + Redis (Performance)
| Service | Coût/mois | Notes |
|---------|-----------|-------|
| Cloud Run | $20-35 | 5000 req/jour |
| Cloud SQL (db-g1-small) | $25 | PostgreSQL |
| Memorystore Redis | $25 | 1GB |
| Cloud Storage | $5 | Models + assets |
| Load Balancer | $5 | HTTPS |
| Monitoring & Logging | $10 | Logs étendus |
| **Total** | **$90-105** | **Budget production** |

### GKE avec GPU (Future)
| Service | Coût/mois | Notes |
|---------|-----------|-------|
| GKE Cluster | $75 | Management fee |
| Nodes (2x n1-standard-4) | $150 | 4 vCPU, 15GB RAM |
| GPU (1x NVIDIA T4) | $150 | Pour LED/inference |
| Cloud SQL | $50 | HA PostgreSQL |
| Storage & Network | $30 | Disques + egress |
| Monitoring | $20 | Advanced |
| **Total** | **$475** | **Budget scaling** |

---

## 📈 KPIs et Métriques de Succès

### Techniques
- ✅ Uptime ≥ 99.5%
- ✅ Latency P95 < 2s
- ✅ Error rate < 1%
- ✅ Throughput ≥ 50 req/s
- ✅ Cost per request < $0.01

### Business
- ✅ 1000+ utilisateurs/mois (Mois 3)
- ✅ 10,000+ résumés générés (Mois 3)
- ✅ Score satisfaction ≥ 4.5/5
- ✅ Taux de rétention ≥ 60%

### Qualité
- ✅ Score d'évaluation moyen ≥ 0.75
- ✅ Temps génération < 30s (90% des cas)
- ✅ Support multilingue (FR, EN, ES, DE)
- ✅ Taux de succès ≥ 95%

---

## 🔄 Sprints Détaillés

### Semaine 1-2 : Ollama Integration
- **Jours 1-2** : Développement `ollama_model.py`
- **Jours 3-4** : Intégration `model_manager.py`
- **Jour 5** : Tests et benchmarks
- **Jours 6-7** : Tests modèles, comparaisons, UI

### Semaine 2-3 : API Enhancement
- **Jours 1-2** : Nouveaux endpoints + sécurité
- **Jour 3** : Documentation OpenAPI
- **Jours 4-5** : Tests et optimisations

### Semaine 3-4 : Docker & Compose
- **Jours 1-2** : Dockerfiles optimisés
- **Jour 3** : Docker Compose complet
- **Jours 4-5** : Tests d'intégration

### Semaine 4-5 : GCP Setup
- **Jours 1-2** : Infrastructure Terraform
- **Jour 3** : Cloud SQL + Storage
- **Jours 4-5** : CI/CD Pipeline

### Semaine 5-6 : Deployment
- **Jours 1-2** : Cloud Run deployment
- **Jour 3** : Load Balancer + SSL
- **Jours 4-5** : Monitoring + Tests

### Semaine 6-7 : Optimization
- **Jours 1-2** : Caching + CDN
- **Jour 3** : Auto-scaling
- **Jours 4-5** : Security + Performance

---

## 🎯 Prochaines Actions Immédiates

### À faire maintenant
1. ✅ Créer ce fichier ROADMAP.md
2. ⏳ Installer Ollama localement
3. ⏳ Créer `src/models/ollama_model.py`
4. ⏳ Mettre à jour `config/model_config.yaml`
5. ⏳ Tester avec Mistral 7B

### Commandes utiles
```bash
# Installer Ollama (macOS)
curl -fsSL https://ollama.com/install.sh | sh

# Télécharger modèles
ollama pull mistral:7b
ollama pull llama3.1:8b
ollama pull gemma2:9b

# Lancer Ollama
ollama serve

# Tester localement
curl http://localhost:11434/api/generate -d '{
  "model": "mistral:7b",
  "prompt": "Résume ce texte..."
}'
```

---

## 📚 Ressources et Documentation

### Ollama
- [Documentation officielle](https://github.com/ollama/ollama)
- [API Reference](https://github.com/ollama/ollama/blob/main/docs/api.md)
- [Modèles disponibles](https://ollama.com/library)

### Google Cloud Platform
- [Cloud Run Documentation](https://cloud.google.com/run/docs)
- [Cloud SQL Guide](https://cloud.google.com/sql/docs)
- [Terraform GCP Provider](https://registry.terraform.io/providers/hashicorp/google/latest/docs)

### Best Practices
- [12-Factor Apps](https://12factor.net/)
- [API Design Guidelines](https://cloud.google.com/apis/design)
- [Docker Best Practices](https://docs.docker.com/develop/dev-best-practices/)

---

## 🤝 Contribution

Pour contribuer à cette roadmap :
1. Créer une issue avec label `roadmap`
2. Proposer des améliorations
3. Soumettre des Pull Requests
4. Participer aux discussions

---

**Dernière mise à jour** : 8 novembre 2025  
**Version** : 1.0  
**Statut** : 🚀 En cours - Phase 1
