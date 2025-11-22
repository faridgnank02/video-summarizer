# 🚀 Guide de Déploiement AWS - Video Summarizer

Guide simplifié pour déployer Video Summarizer sur AWS ECS Fargate avec horaires 10h-18h GMT.

---

## ✅ Prérequis

### Installations
- ✅ Docker installé et lancé
- ✅ AWS CLI configuré (`aws configure`)
- ✅ Compte Docker Hub (gratuit)
- ✅ Compte AWS avec credentials

### Fichiers
- ✅ Clé OpenAI dans `.env`
- ✅ Tous les scripts dans `scripts/`

---

## 🏗️ Étape 1 : Build et Test Local (15 min)

### 1.1 Builder l'image
```bash
./scripts/build_and_test.sh
```

**Temps estimé :** 10-15 minutes  
**Taille finale :** ~3-4 GB

### 1.2 Tester localement
Une fois le build terminé, le script :
- ✅ Démarre automatiquement le container
- ✅ Attend que les services soient prêts (2-3 min)
- ✅ Télécharge gemma3:1b (~1.1GB)
- ✅ Affiche les URLs d'accès

**Testez :**
- Interface : http://localhost:8501
- API : http://localhost:8000/docs

### 1.3 Vérifications
- [ ] Streamlit s'affiche correctement
- [ ] API répond au `/health`
- [ ] Ollama disponible (modèle gemma3:1b)
- [ ] Test résumé YouTube fonctionne
- [ ] Test résumé texte direct fonctionne

### 1.4 Arrêter le test local
```bash
docker-compose down
```

---

## 📦 Étape 2 : Push vers Docker Hub (5 min)

### 2.1 Login Docker Hub
```bash
docker login
```

Entrez vos credentials Docker Hub.

### 2.2 Tag et Push
```bash
# Remplacer VOTRE_USERNAME par votre username Docker Hub
docker tag video-summarizer:latest VOTRE_USERNAME/video-summarizer:latest
docker push VOTRE_USERNAME/video-summarizer:latest
```

**Temps de push :** 5-10 minutes (selon connexion)

### 2.3 Vérifier
Allez sur https://hub.docker.com/r/VOTRE_USERNAME/video-summarizer pour confirmer.

---

## ⚙️ Étape 3 : Configuration AWS (5 min)

### 3.1 Modifier le script de déploiement
Éditez `scripts/deploy_aws.sh` ligne 17 :

```bash
# Avant
DOCKER_IMAGE="${DOCKER_IMAGE:-video-summarizer:latest}"

# Après
DOCKER_IMAGE="${DOCKER_IMAGE:-VOTRE_USERNAME/video-summarizer:latest}"
```

### 3.2 Vérifier les credentials AWS
```bash
aws sts get-caller-identity
```

Doit afficher votre Account ID et User ARN.

### 3.3 Vérifier la clé OpenAI
```bash
cat .env | grep OPENAI_API_KEY
```

Doit afficher : `OPENAI_API_KEY=sk-proj-...`

---

## 🚀 Étape 4 : Déploiement AWS ECS (15 min)

### 4.1 Lancer le déploiement
```bash
# Charger les variables d'environnement
export $(grep -v '^#' .env | xargs)

# Déployer
./scripts/deploy_aws.sh
```

**Le script va automatiquement :**
1. ✅ Créer le cluster ECS `video-summarizer-cluster`
2. ✅ Créer le secret OpenAI dans AWS Secrets Manager
3. ✅ Créer les rôles IAM (ecsTaskExecutionRole, ecsTaskRole)
4. ✅ Créer le Security Group (ports 8501 + 8000)
5. ✅ Enregistrer la Task Definition (2 vCPU / 4GB)
6. ✅ Déployer le service Fargate Spot
7. ✅ Attendre que la tâche soit RUNNING
8. ✅ Récupérer l'IP publique

**Temps estimé :** 10-15 minutes

### 4.2 Récupérer l'URL
À la fin, vous verrez :
```
╔════════════════════════════════════════════════╗
║           DEPLOYMENT SUCCESSFUL! 🎉            ║
╚════════════════════════════════════════════════╝

📱 Streamlit UI: http://XX.XXX.XXX.XX:8501
🔧 API:          http://XX.XXX.XXX.XX:8000
📚 API Docs:     http://XX.XXX.XXX.XX:8000/docs

⏰ Note: First startup takes 2-3 minutes
```

**Notez l'IP publique !**

### 4.3 Premier accès
⚠️ **Attendez 2-3 minutes** après le déploiement (téléchargement du modèle Ollama).

Puis accédez à : `http://IP_PUBLIQUE:8501`

---

## 🧪 Étape 5 : Tests de Validation (10 min)

### 5.1 Tests de santé
```bash
# Remplacer IP par votre IP publique
IP=XX.XXX.XXX.XX

# Test Streamlit
curl http://$IP:8501/_stcore/health

# Test API
curl http://$IP:8000/health | python3 -m json.tool
```

### 5.2 Tests fonctionnels
Dans l'interface Streamlit :

1. **Test YouTube**
   - URL courte avec sous-titres FR/EN
   - Modèle Ollama + Long
   - Vérifier résumé généré

2. **Test Texte Direct**
   - Texte 500 mots
   - Modèle Auto + Short
   - Vérifier export TXT/JSON

3. **Test Switch Modèle**
   - Même texte avec Ollama puis OpenAI
   - Comparer qualité et temps

### 5.3 Vérifier les logs
```bash
# Voir les logs en temps réel
aws logs tail /ecs/video-summarizer-task --region us-east-1 --follow

# Logs des 5 dernières minutes
aws logs tail /ecs/video-summarizer-task --region us-east-1 --since 5m
```

---

## ⏰ Étape 6 : Configuration Horaire 10h-18h GMT (5 min)

### Option 1 : Manuel (Simple)

**Arrêter le service (après 18h GMT) :**
```bash
./scripts/schedule_service.sh stop
```

**Démarrer le service (à 10h GMT) :**
```bash
./scripts/schedule_service.sh start
```

**Vérifier l'état :**
```bash
./scripts/schedule_service.sh status
```

### Option 2 : EventBridge (Automatique)

```bash
./scripts/setup_eventbridge_schedule.sh
```

Puis configurer manuellement les targets dans AWS Console :
1. https://console.aws.amazon.com/events/
2. Règles : `video-summarizer-start-10h` et `video-summarizer-stop-18h`
3. Pour chaque règle : Edit > Targets > ECS task
4. Configurer Cluster, Task Definition, Subnets, Security Groups

**Horaires :**
- ⏰ **Démarrage** : 10:00 GMT (cron: `0 10 * * ? *`)
- ⏰ **Arrêt** : 18:00 GMT (cron: `0 18 * * ? *`)

---

## 📊 Étape 7 : Monitoring et Logs

### Voir les métriques ECS
```bash
# Status du service
aws ecs describe-services \
  --cluster video-summarizer-cluster \
  --services video-summarizer-app \
  --region us-east-1 \
  --query 'services[0].{Status:status,Running:runningCount,Desired:desiredCount}'

# Détails de la tâche
aws ecs describe-tasks \
  --cluster video-summarizer-cluster \
  --tasks $(aws ecs list-tasks --cluster video-summarizer-cluster --service-name video-summarizer-app --region us-east-1 --query 'taskArns[0]' --output text) \
  --region us-east-1
```

### Console AWS
- **ECS** : https://console.aws.amazon.com/ecs/v2/clusters/video-summarizer-cluster
- **CloudWatch Logs** : https://console.aws.amazon.com/cloudwatch/home?region=us-east-1#logsV2:log-groups/log-group/$252Fecs$252Fvideo-summarizer-task
- **Secrets Manager** : https://console.aws.amazon.com/secretsmanager/

---

## 📝 Étape 8 : Documentation pour Testeurs

### 8.1 Préparer le guide
Le fichier `docs/GUIDE_TESTEURS.md` est prêt !

### 8.2 Créer un document partagé
Créez un Google Doc avec :

```markdown
# Video Summarizer - Accès Testeurs

## 🌐 URL d'Accès
http://[VOTRE_IP_PUBLIQUE]:8501

## ⏰ Horaires
10h00 - 18h00 GMT (tous les jours)

## 📖 Guide Complet
[Lien vers GUIDE_TESTEURS.md]

## 📝 Formulaire Feedback
[Lien Google Forms]

## 📧 Contact
[Votre email]
```

### 8.3 Partager aux testeurs
- Email avec le lien
- Instructions de base
- Lien vers le guide complet
- Formulaire de feedback

---

## 🔧 Gestion et Maintenance

### Mettre à jour l'application
```bash
# 1. Rebuild avec changements
./scripts/build_and_test.sh

# 2. Push nouvelle version
docker tag video-summarizer:latest VOTRE_USERNAME/video-summarizer:v2
docker push VOTRE_USERNAME/video-summarizer:v2

# 3. Update task definition dans deploy_aws.sh si besoin

# 4. Force new deployment
aws ecs update-service \
  --cluster video-summarizer-cluster \
  --service video-summarizer-app \
  --region us-east-1 \
  --force-new-deployment
```

### Redémarrer le service
```bash
# Si le service crash ou freeze
aws ecs update-service \
  --cluster video-summarizer-cluster \
  --service video-summarizer-app \
  --region us-east-1 \
  --force-new-deployment
```

### Changer la capacité
```bash
# Scale up (2 instances)
aws ecs update-service \
  --cluster video-summarizer-cluster \
  --service video-summarizer-app \
  --desired-count 2 \
  --region us-east-1

# Scale down (0 instances = stop)
aws ecs update-service \
  --cluster video-summarizer-cluster \
  --service video-summarizer-app \
  --desired-count 0 \
  --region us-east-1
```

---

## 💰 Estimation des Coûts

### Configuration Actuelle
- **CPU** : 2 vCPU
- **RAM** : 4 GB
- **Type** : Fargate Spot (70% moins cher)
- **Horaires** : 8h/jour

### Coûts Mensuels
| Service | Coût |
|---------|------|
| Fargate Spot (2 vCPU + 4GB) | ~$14.40/mois |
| OpenAI API | ~$2-5/mois |
| Secrets Manager | ~$0.40/mois |
| CloudWatch Logs | ~$0.50/mois |
| **TOTAL** | **~$17-20/mois** |

### Réduire les Coûts
- ✅ **Arrêter quand non utilisé** : `--desired-count 0`
- ✅ **Limiter OpenAI** : Utiliser plus Ollama
- ✅ **Réduire logs** : Ajuster retention CloudWatch
- ✅ **Moins de vCPU** : 1 vCPU / 2GB si pas assez de charge

---

## 🗑️ Nettoyage Complet

### Supprimer toutes les ressources AWS
```bash
# 1. Supprimer le service
aws ecs delete-service \
  --cluster video-summarizer-cluster \
  --service video-summarizer-app \
  --region us-east-1 \
  --force

# 2. Attendre 30 secondes
sleep 30

# 3. Supprimer le cluster
aws ecs delete-cluster \
  --cluster video-summarizer-cluster \
  --region us-east-1

# 4. Supprimer le security group
SG_ID=$(aws ec2 describe-security-groups \
  --filters "Name=group-name,Values=video-summarizer-sg" \
  --region us-east-1 \
  --query 'SecurityGroups[0].GroupId' \
  --output text)
aws ec2 delete-security-group --group-id $SG_ID --region us-east-1

# 5. Supprimer le secret
aws secretsmanager delete-secret \
  --secret-id video-summarizer/openai-key \
  --region us-east-1 \
  --force-delete-without-recovery

# 6. Supprimer les logs
aws logs delete-log-group \
  --log-group-name /ecs/video-summarizer-task \
  --region us-east-1

# 7. Supprimer les règles EventBridge (si configurées)
aws events remove-targets --rule video-summarizer-start-10h --ids 1 --region us-east-1
aws events remove-targets --rule video-summarizer-stop-18h --ids 1 --region us-east-1
aws events delete-rule --name video-summarizer-start-10h --region us-east-1
aws events delete-rule --name video-summarizer-stop-18h --region us-east-1
```

---

## ⚠️ Troubleshooting

### Container crash (exit code 255)
```bash
# Vérifier les logs
aws logs tail /ecs/video-summarizer-task --region us-east-1 --since 10m

# Si "exec format error" : rebuilder pour AMD64
docker buildx build --platform linux/amd64 -t video-summarizer:latest . --load
```

### Out of memory
```bash
# Augmenter la mémoire dans deploy_aws.sh
# Ligne MEMORY="4096" -> MEMORY="8192"
# Puis redéployer
```

### Ollama model download timeout
```bash
# Première fois peut être long (2-3 min)
# Vérifier les logs CloudWatch pour progression
aws logs tail /ecs/video-summarizer-task --region us-east-1 --follow
```

### IP change après restart
```bash
# L'IP change à chaque redémarrage du service
# Récupérer la nouvelle IP :
TASK_ARN=$(aws ecs list-tasks --cluster video-summarizer-cluster --service-name video-summarizer-app --region us-east-1 --query 'taskArns[0]' --output text)
ENI_ID=$(aws ecs describe-tasks --cluster video-summarizer-cluster --tasks $TASK_ARN --region us-east-1 --query 'tasks[0].attachments[0].details[?name==`networkInterfaceId`].value' --output text)
aws ec2 describe-network-interfaces --network-interface-ids $ENI_ID --region us-east-1 --query 'NetworkInterfaces[0].Association.PublicIp' --output text
```

---

## 📚 Ressources

- **Documentation AWS ECS** : https://docs.aws.amazon.com/ecs/
- **Fargate Pricing** : https://aws.amazon.com/fargate/pricing/
- **Docker Hub** : https://hub.docker.com/
- **Guide Testeurs** : `docs/GUIDE_TESTEURS.md`
- **Documentation Technique** : `docs/TECHNICAL_DOCUMENTATION.md`

---

## ✅ Checklist de Déploiement

- [ ] Build local réussi
- [ ] Test local fonctionnel
- [ ] Image pushée sur Docker Hub
- [ ] Script deploy_aws.sh modifié avec bonne image
- [ ] AWS CLI configuré
- [ ] Clé OpenAI dans .env
- [ ] Déploiement AWS réussi
- [ ] IP publique récupérée
- [ ] Tests de validation OK
- [ ] Horaires configurés (manuel ou EventBridge)
- [ ] Guide testeur préparé
- [ ] Testeurs invités avec lien

---

**Version** : 1.0.0  
**Date** : 21 novembre 2025  
**Configuration** : 2 vCPU / 4GB / Fargate Spot / gemma3:1b
