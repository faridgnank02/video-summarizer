#!/bin/bash
# Script pour obtenir l'URL publique de l'application AWS

set -e

GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m'

CLUSTER_NAME="video-summarizer-cluster"
SERVICE_NAME="video-summarizer-app"
AWS_REGION="${AWS_REGION:-us-east-1}"

echo -e "${BLUE}🔍 Fetching application URL...${NC}\n"

# Vérifier si le service tourne
RUNNING_COUNT=$(aws ecs describe-services \
    --cluster $CLUSTER_NAME \
    --services $SERVICE_NAME \
    --region $AWS_REGION \
    --query 'services[0].runningCount' \
    --output text)

if [ "$RUNNING_COUNT" = "0" ]; then
    echo -e "❌ Service is not running (desired count = 0)"
    echo -e "\n💡 Start the service with:"
    echo -e "   ./scripts/schedule_service.sh start"
    exit 1
fi

# Récupérer l'ARN de la tâche
TASK_ARN=$(aws ecs list-tasks \
    --cluster $CLUSTER_NAME \
    --service-name $SERVICE_NAME \
    --region $AWS_REGION \
    --query 'taskArns[0]' \
    --output text)

if [ -z "$TASK_ARN" ] || [ "$TASK_ARN" = "None" ]; then
    echo -e "❌ No running task found"
    exit 1
fi

# Récupérer l'interface réseau
ENI_ID=$(aws ecs describe-tasks \
    --cluster $CLUSTER_NAME \
    --tasks $TASK_ARN \
    --region $AWS_REGION \
    --query 'tasks[0].attachments[0].details[?name==`networkInterfaceId`].value' \
    --output text)

# Récupérer l'IP publique
PUBLIC_IP=$(aws ec2 describe-network-interfaces \
    --network-interface-ids $ENI_ID \
    --region $AWS_REGION \
    --query 'NetworkInterfaces[0].Association.PublicIp' \
    --output text)

if [ -z "$PUBLIC_IP" ] || [ "$PUBLIC_IP" = "None" ]; then
    echo -e "❌ No public IP assigned"
    exit 1
fi

# Afficher les URLs
echo -e "${GREEN}✅ Application is running!${NC}\n"
echo -e "📱 ${BLUE}Streamlit UI:${NC}  http://$PUBLIC_IP:8501"
echo -e "🔧 ${BLUE}API:${NC}           http://$PUBLIC_IP:8000"
echo -e "📚 ${BLUE}API Docs:${NC}      http://$PUBLIC_IP:8000/docs"
echo -e "\n💡 ${BLUE}Health Check:${NC}  curl http://$PUBLIC_IP:8000/health"
echo -e "\n📊 ${BLUE}View Logs:${NC}"
echo -e "   aws logs tail /ecs/video-summarizer-task --region $AWS_REGION --follow"
