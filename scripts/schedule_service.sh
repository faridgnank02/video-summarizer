#!/bin/bash
# Script de gestion horaire pour Video Summarizer
# Démarrage automatique : 10h GMT
# Arrêt automatique : 18h GMT
# Usage: Ajouter dans crontab ou AWS EventBridge

set -e

CLUSTER_NAME="video-summarizer-cluster"
SERVICE_NAME="video-summarizer-app"
AWS_REGION="${AWS_REGION:-us-east-1}"

# Fonction pour démarrer le service
start_service() {
    echo "⏰ [$(date)] Starting Video Summarizer service..."
    
    aws ecs update-service \
        --cluster $CLUSTER_NAME \
        --service $SERVICE_NAME \
        --desired-count 1 \
        --region $AWS_REGION
    
    echo "✅ Service started - will be ready in 2-3 minutes"
    
    # Attendre que la tâche soit RUNNING et récupérer l'IP
    sleep 90
    
    TASK_ARN=$(aws ecs list-tasks \
        --cluster $CLUSTER_NAME \
        --service-name $SERVICE_NAME \
        --region $AWS_REGION \
        --desired-status RUNNING \
        --query 'taskArns[0]' \
        --output text)
    
    if [ "$TASK_ARN" != "None" ] && [ ! -z "$TASK_ARN" ]; then
        ENI_ID=$(aws ecs describe-tasks \
            --cluster $CLUSTER_NAME \
            --tasks $TASK_ARN \
            --region $AWS_REGION \
            --query 'tasks[0].attachments[0].details[?name==`networkInterfaceId`].value' \
            --output text)
        
        PUBLIC_IP=$(aws ec2 describe-network-interfaces \
            --network-interface-ids $ENI_ID \
            --region $AWS_REGION \
            --query 'NetworkInterfaces[0].Association.PublicIp' \
            --output text)
        
        echo "🌐 Application available at: http://${PUBLIC_IP}:8501"
    fi
}

# Fonction pour arrêter le service
stop_service() {
    echo "⏰ [$(date)] Stopping Video Summarizer service..."
    
    aws ecs update-service \
        --cluster $CLUSTER_NAME \
        --service $SERVICE_NAME \
        --desired-count 0 \
        --region $AWS_REGION
    
    echo "✅ Service stopped - no charges until next start"
}

# Fonction pour vérifier l'état
check_status() {
    STATUS=$(aws ecs describe-services \
        --cluster $CLUSTER_NAME \
        --services $SERVICE_NAME \
        --region $AWS_REGION \
        --query 'services[0].{Running:runningCount,Desired:desiredCount}' \
        --output text)
    
    echo "📊 Current status: $STATUS"
}

# Menu principal
case "${1}" in
    start)
        start_service
        ;;
    stop)
        stop_service
        ;;
    status)
        check_status
        ;;
    *)
        echo "Usage: $0 {start|stop|status}"
        echo ""
        echo "For automatic scheduling (10h-18h GMT):"
        echo ""
        echo "Option 1: Local crontab (if your machine is always on)"
        echo "  # Add to crontab: crontab -e"
        echo "  0 10 * * * $PWD/$0 start  # Start at 10:00 GMT"
        echo "  0 18 * * * $PWD/$0 stop   # Stop at 18:00 GMT"
        echo ""
        echo "Option 2: AWS EventBridge (recommended)"
        echo "  See scripts/setup_eventbridge_schedule.sh"
        echo ""
        exit 1
        ;;
esac
