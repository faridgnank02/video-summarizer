# 🚀 Guide de Gestion AWS - Video Summarizer

## 📱 Obtenir l'URL Actuelle de l'Application

L'IP publique peut changer lors des redémarrages (Fargate Spot). Utilisez ce script pour récupérer l'URL actuelle :

```bash
./scripts/get_app_url.sh
```

**Sortie attendue :**
```
✅ Application is running!

📱 Streamlit UI:  http://18.212.89.148:8501
🔧 API:           http://18.212.89.148:8000
📚 API Docs:      http://18.212.89.148:8000/docs
```

---

## ⏰ Gestion du Service (Démarrage/Arrêt)

### Démarrer l'Application

```bash
./scripts/schedule_service.sh start
```

**Résultat :** Service ECS passe à `desired-count=1`, Fargate lance le conteneur (~2-3 min)

### Arrêter l'Application (Économiser $$$)

```bash
./scripts/schedule_service.sh stop
```

**Résultat :** Service ECS passe à `desired-count=0`, conteneur s'arrête immédiatement

### Vérifier l'État du Service

```bash
aws ecs describe-services \
  --cluster video-summarizer-cluster \
  --services video-summarizer-app \
  --region us-east-1 \
  --query 'services[0].{Status:status,Desired:desiredCount,Running:runningCount}'
```

**Sortie :**
```json
{
    "Status": "ACTIVE",
    "Desired": 1,
    "Running": 1
}
```

---

## 📊 Consultation des Logs

### Logs en Temps Réel

```bash
aws logs tail /ecs/video-summarizer-task --region us-east-1 --follow
```

**Utilisation :**
- Voir les démarrages Ollama, FastAPI, Streamlit
- Débugger les erreurs
- Monitorer les requêtes utilisateurs
- Ctrl+C pour quitter

### Logs des 30 Dernières Minutes

```bash
aws logs tail /ecs/video-summarizer-task --region us-east-1 --since 30m
```

### Recherche dans les Logs

```bash
aws logs tail /ecs/video-summarizer-task --region us-east-1 --filter-pattern "ERROR"
```

---

## 🔄 Mise à Jour de l'Application

### 1. Modifier le Code Localement

Éditez vos fichiers Python, testez localement avec Docker :

```bash
docker-compose up
```

### 2. Reconstruire l'Image Docker

```bash
docker buildx build --platform linux/amd64 -t frtrenton002/video-summarizer:latest . --load
```

### 3. Pousser vers Docker Hub

```bash
docker push frtrenton002/video-summarizer:latest
```

### 4. Redéployer sur AWS

```bash
aws ecs update-service \
  --cluster video-summarizer-cluster \
  --service video-summarizer-app \
  --force-new-deployment \
  --region us-east-1
```

**Résultat :** ECS télécharge la nouvelle image, arrête l'ancien conteneur, démarre le nouveau (~3-5 min)

### 5. Vérifier le Redéploiement

```bash
aws ecs describe-services \
  --cluster video-summarizer-cluster \
  --services video-summarizer-app \
  --region us-east-1 \
  --query 'services[0].deployments'
```

Attendez que `rolloutState` = `COMPLETED`

---

## 💰 Monitoring des Coûts

### Voir les Coûts AWS Actuels

1. Ouvrir : https://console.aws.amazon.com/billing/home#/bills
2. Filtrer par service : `ECS`, `Secrets Manager`, `CloudWatch`

### Estimation Mensuelle Actuelle

**Configuration :** 2 vCPU / 4GB RAM / Fargate Spot

| Scénario | Heures/Mois | Coût Fargate | Coût Secrets | Coût Logs | **Total** |
|----------|-------------|--------------|--------------|-----------|-----------|
| 24/7 | 720h | $21.31 | $0.40 | $0.50 | **~$22/mois** |
| 8h/jour | 240h | $7.10 | $0.40 | $0.50 | **~$8/mois** |

### Réduire les Coûts

1. **Arrêter quand inutile** : `./scripts/schedule_service.sh stop`
2. **Utiliser le scheduling** : 8h/jour = -66% de coût
3. **Fargate Spot** : Déjà activé (-70% vs Fargate standard)
4. **Réduire RAM** : Si 4GB trop, tester 2GB (éditer `deploy_aws.sh` ligne 22)

---

## 🔐 Mise à Jour de la Clé OpenAI

### Via AWS Secrets Manager

```bash
aws secretsmanager update-secret \
  --secret-id video-summarizer/openai-key \
  --secret-string "sk-proj-NOUVELLE_CLE_ICI" \
  --region us-east-1
```

### Redémarrer pour Appliquer

```bash
./scripts/schedule_service.sh stop
sleep 10
./scripts/schedule_service.sh start
```

---

## 🛑 Suppression Complète de l'Infrastructure

⚠️ **ATTENTION** : Ceci supprime TOUT et arrête la facturation.

```bash
# 1. Supprimer le service ECS
aws ecs delete-service \
  --cluster video-summarizer-cluster \
  --service video-summarizer-app \
  --force \
  --region us-east-1

# 2. Supprimer le cluster
aws ecs delete-cluster \
  --cluster video-summarizer-cluster \
  --region us-east-1

# 3. Supprimer le secret
aws secretsmanager delete-secret \
  --secret-id video-summarizer/openai-key \
  --force-delete-without-recovery \
  --region us-east-1

# 4. Supprimer le security group
SG_ID=$(aws ec2 describe-security-groups \
  --filters "Name=group-name,Values=video-summarizer-sg" \
  --query 'SecurityGroups[0].GroupId' \
  --output text \
  --region us-east-1)

aws ec2 delete-security-group \
  --group-id $SG_ID \
  --region us-east-1

# 5. Supprimer les règles EventBridge
aws events remove-targets \
  --rule video-summarizer-start-10h \
  --ids "1" \
  --region us-east-1

aws events delete-rule \
  --name video-summarizer-start-10h \
  --region us-east-1

aws events remove-targets \
  --rule video-summarizer-stop-18h \
  --ids "1" \
  --region us-east-1

aws events delete-rule \
  --name video-summarizer-stop-18h \
  --region us-east-1

# 6. Supprimer les logs CloudWatch
aws logs delete-log-group \
  --log-group-name /ecs/video-summarizer-task \
  --region us-east-1
```

---

## 📞 Dépannage Rapide

### L'IP ne répond plus

**Cause :** IP changée (Fargate Spot redémarrage)  
**Solution :** `./scripts/get_app_url.sh`

### Service bloqué en PENDING

**Cause :** Image trop volumineuse ou subnet sans route Internet  
**Solution :**
```bash
aws ecs describe-tasks \
  --cluster video-summarizer-cluster \
  --tasks $(aws ecs list-tasks --cluster video-summarizer-cluster --region us-east-1 --query 'taskArns[0]' --output text) \
  --region us-east-1 \
  --query 'tasks[0].stopCode'
```

### Health Check échoue

**Cause :** Ollama met >2 min à télécharger gemma3:1b  
**Solution :** Attendre 3-5 min après premier démarrage

### Coûts inattendus

**Cause :** Service laissé actif 24/7  
**Solution :** `./scripts/schedule_service.sh stop` immédiatement

---

## 🔗 Liens Utiles AWS

- **Console ECS** : https://console.aws.amazon.com/ecs/home?region=us-east-1
- **Logs CloudWatch** : https://console.aws.amazon.com/cloudwatch/home?region=us-east-1#logsV2:log-groups/log-group/$252Fecs$252Fvideo-summarizer-task
- **Secrets Manager** : https://console.aws.amazon.com/secretsmanager/home?region=us-east-1#!/secret?name=video-summarizer/openai-key
- **Billing Dashboard** : https://console.aws.amazon.com/billing/home#/bills
- **Cost Explorer** : https://console.aws.amazon.com/cost-management/home#/cost-explorer

---

## 📝 Commandes Rapides (Cheat Sheet)

```bash
# Obtenir l'URL actuelle
./scripts/get_app_url.sh

# Démarrer
./scripts/schedule_service.sh start

# Arrêter (économiser $$$)
./scripts/schedule_service.sh stop

# Logs temps réel
aws logs tail /ecs/video-summarizer-task --region us-east-1 --follow

# Redéployer nouvelle version
docker push frtrenton002/video-summarizer:latest
aws ecs update-service --cluster video-summarizer-cluster --service video-summarizer-app --force-new-deployment --region us-east-1

# Vérifier coûts
aws ce get-cost-and-usage --time-period Start=$(date -u -d '1 month ago' +%Y-%m-%d),End=$(date -u +%Y-%m-%d) --granularity MONTHLY --metrics BlendedCost --region us-east-1
```

---

**🎉 Votre application est maintenant déployée sur AWS !**

URL actuelle : http://18.212.89.148:8501  
_(Utilisez `./scripts/get_app_url.sh` si l'IP change)_
