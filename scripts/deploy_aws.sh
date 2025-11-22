#!/bin/bash
# Script de déploiement AWS ECS Fargate pour Video Summarizer
# Configuration: 2 vCPU / 4GB RAM / Fargate Spot
# Horaires: 10h-18h GMT (8h/jour)

set -e

# Couleurs
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m'

# Configuration
CLUSTER_NAME="video-summarizer-cluster"
SERVICE_NAME="video-summarizer-app"
TASK_FAMILY="video-summarizer-task"
AWS_REGION="${AWS_REGION:-us-east-1}"
DOCKER_IMAGE="${DOCKER_IMAGE:-frtrenton002/video-summarizer:latest}"
CPU="2048"  # 2 vCPU
MEMORY="4096"  # 4 GB RAM (suffisant pour image ~2-3GB + gemma3:1b ~1.1GB)
DESIRED_COUNT="1"

# Note: Image optimisée ~2-3GB (sans PyTorch/transformers)
# Mémoire task: 4GB permet de charger image + modèle Ollama confortablement

echo -e "${BLUE}╔════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║   AWS ECS Deployment - Video Summarizer       ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════════╝${NC}"
echo ""

# Vérifier les prérequis
echo -e "${GREEN}[1/10]${NC} Checking prerequisites..."

if ! command -v aws &> /dev/null; then
    echo -e "${RED}❌ AWS CLI not found. Install it first.${NC}"
    exit 1
fi

if [ -z "$OPENAI_API_KEY" ]; then
    echo -e "${YELLOW}⚠️  OPENAI_API_KEY not set. Loading from .env file...${NC}"
    if [ -f .env ]; then
        export $(grep -v '^#' .env | xargs)
    else
        echo -e "${RED}❌ .env file not found and OPENAI_API_KEY not set${NC}"
        exit 1
    fi
fi

# Vérifier la connexion AWS
if ! aws sts get-caller-identity &> /dev/null; then
    echo -e "${RED}❌ AWS credentials not configured. Run 'aws configure' first.${NC}"
    exit 1
fi

echo -e "${GREEN}✅ Prerequisites OK${NC}"

# 1. Créer ou vérifier le cluster ECS
echo -e "${GREEN}[2/10]${NC} Creating/checking ECS cluster..."
if aws ecs describe-clusters --clusters $CLUSTER_NAME --region $AWS_REGION | grep -q "ACTIVE"; then
    echo -e "${YELLOW}✓ Cluster already exists${NC}"
else
    aws ecs create-cluster \
        --cluster-name $CLUSTER_NAME \
        --region $AWS_REGION
    echo -e "${GREEN}✅ Cluster created${NC}"
fi

# 2. Créer le secret OpenAI dans Secrets Manager
echo -e "${GREEN}[3/10]${NC} Creating/updating OpenAI API key secret..."
SECRET_NAME="video-summarizer/openai-key"

# Vérifier si le secret existe
if aws secretsmanager describe-secret --secret-id $SECRET_NAME --region $AWS_REGION &> /dev/null; then
    # Mettre à jour
    aws secretsmanager update-secret \
        --secret-id $SECRET_NAME \
        --secret-string "$OPENAI_API_KEY" \
        --region $AWS_REGION
    echo -e "${YELLOW}✓ Secret updated${NC}"
else
    # Créer
    aws secretsmanager create-secret \
        --name $SECRET_NAME \
        --description "OpenAI API key for Video Summarizer" \
        --secret-string "$OPENAI_API_KEY" \
        --region $AWS_REGION
    echo -e "${GREEN}✅ Secret created${NC}"
fi

# Récupérer l'ARN du secret
SECRET_ARN=$(aws secretsmanager describe-secret \
    --secret-id $SECRET_NAME \
    --region $AWS_REGION \
    --query 'ARN' \
    --output text)

# 3. Créer les rôles IAM si nécessaire
echo -e "${GREEN}[4/10]${NC} Checking IAM roles..."

# ecsTaskExecutionRole
if ! aws iam get-role --role-name ecsTaskExecutionRole &> /dev/null; then
    echo -e "${YELLOW}Creating ecsTaskExecutionRole...${NC}"
    aws iam create-role \
        --role-name ecsTaskExecutionRole \
        --assume-role-policy-document '{
            "Version": "2012-10-17",
            "Statement": [{
                "Effect": "Allow",
                "Principal": {"Service": "ecs-tasks.amazonaws.com"},
                "Action": "sts:AssumeRole"
            }]
        }'
    
    aws iam attach-role-policy \
        --role-name ecsTaskExecutionRole \
        --policy-arn arn:aws:iam::aws:policy/service-role/AmazonECSTaskExecutionRolePolicy
    
    aws iam attach-role-policy \
        --role-name ecsTaskExecutionRole \
        --policy-arn arn:aws:iam::aws:policy/SecretsManagerReadWrite
fi

# ecsTaskRole
if ! aws iam get-role --role-name ecsTaskRole &> /dev/null; then
    echo -e "${YELLOW}Creating ecsTaskRole...${NC}"
    aws iam create-role \
        --role-name ecsTaskRole \
        --assume-role-policy-document '{
            "Version": "2012-10-17",
            "Statement": [{
                "Effect": "Allow",
                "Principal": {"Service": "ecs-tasks.amazonaws.com"},
                "Action": "sts:AssumeRole"
            }]
        }'
    
    aws iam attach-role-policy \
        --role-name ecsTaskRole \
        --policy-arn arn:aws:iam::aws:policy/SecretsManagerReadWrite
fi

echo -e "${GREEN}✅ IAM roles ready${NC}"

# Récupérer les ARNs des rôles
ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
EXECUTION_ROLE_ARN="arn:aws:iam::${ACCOUNT_ID}:role/ecsTaskExecutionRole"
TASK_ROLE_ARN="arn:aws:iam::${ACCOUNT_ID}:role/ecsTaskRole"

# 4. Enregistrer la task definition
echo -e "${GREEN}[5/10]${NC} Registering task definition..."

cat > /tmp/task-definition.json <<EOF
{
    "family": "${TASK_FAMILY}",
    "networkMode": "awsvpc",
    "requiresCompatibilities": ["FARGATE"],
    "cpu": "${CPU}",
    "memory": "${MEMORY}",
    "executionRoleArn": "${EXECUTION_ROLE_ARN}",
    "taskRoleArn": "${TASK_ROLE_ARN}",
    "containerDefinitions": [
        {
            "name": "video-summarizer",
            "image": "${DOCKER_IMAGE}",
            "essential": true,
            "portMappings": [
                {
                    "containerPort": 8501,
                    "protocol": "tcp"
                },
                {
                    "containerPort": 8000,
                    "protocol": "tcp"
                }
            ],
            "environment": [
                {
                    "name": "OLLAMA_MODEL",
                    "value": "gemma3:1b"
                },
                {
                    "name": "STREAMLIT_SERVER_PORT",
                    "value": "8501"
                },
                {
                    "name": "API_PORT",
                    "value": "8000"
                },
                {
                    "name": "AWS_REGION",
                    "value": "${AWS_REGION}"
                }
            ],
            "secrets": [
                {
                    "name": "OPENAI_API_KEY",
                    "valueFrom": "${SECRET_ARN}"
                }
            ],
            "logConfiguration": {
                "logDriver": "awslogs",
                "options": {
                    "awslogs-group": "/ecs/${TASK_FAMILY}",
                    "awslogs-region": "${AWS_REGION}",
                    "awslogs-stream-prefix": "ecs",
                    "awslogs-create-group": "true"
                }
            },
            "healthCheck": {
                "command": ["CMD-SHELL", "curl -f http://localhost:8501/_stcore/health || exit 1"],
                "interval": 30,
                "timeout": 10,
                "retries": 3,
                "startPeriod": 120
            }
        }
    ]
}
EOF

aws ecs register-task-definition \
    --cli-input-json file:///tmp/task-definition.json \
    --region $AWS_REGION

echo -e "${GREEN}✅ Task definition registered${NC}"

# 5. Créer le security group
echo -e "${GREEN}[6/10]${NC} Creating security group..."

# Récupérer le VPC par défaut
VPC_ID=$(aws ec2 describe-vpcs \
    --filters "Name=isDefault,Values=true" \
    --query 'Vpcs[0].VpcId' \
    --output text \
    --region $AWS_REGION)

# Créer ou récupérer le security group
SG_NAME="video-summarizer-sg"
if aws ec2 describe-security-groups \
    --filters "Name=group-name,Values=$SG_NAME" \
    --region $AWS_REGION | grep -q "GroupId"; then
    
    SG_ID=$(aws ec2 describe-security-groups \
        --filters "Name=group-name,Values=$SG_NAME" \
        --query 'SecurityGroups[0].GroupId' \
        --output text \
        --region $AWS_REGION)
    echo -e "${YELLOW}✓ Security group already exists: $SG_ID${NC}"
else
    SG_ID=$(aws ec2 create-security-group \
        --group-name $SG_NAME \
        --description "Security group for Video Summarizer" \
        --vpc-id $VPC_ID \
        --region $AWS_REGION \
        --query 'GroupId' \
        --output text)
    
    # Autoriser port 8501 (Streamlit)
    aws ec2 authorize-security-group-ingress \
        --group-id $SG_ID \
        --protocol tcp \
        --port 8501 \
        --cidr 0.0.0.0/0 \
        --region $AWS_REGION
    
    # Autoriser port 8000 (API)
    aws ec2 authorize-security-group-ingress \
        --group-id $SG_ID \
        --protocol tcp \
        --port 8000 \
        --cidr 0.0.0.0/0 \
        --region $AWS_REGION
    
    echo -e "${GREEN}✅ Security group created: $SG_ID${NC}"
fi

# 6. Récupérer les subnets
echo -e "${GREEN}[7/10]${NC} Getting subnets..."
SUBNETS=$(aws ec2 describe-subnets \
    --filters "Name=vpc-id,Values=$VPC_ID" \
    --query 'Subnets[*].SubnetId' \
    --output text \
    --region $AWS_REGION | tr '\t' ',')

echo -e "${YELLOW}✓ Subnets: $SUBNETS${NC}"

# 7. Créer ou mettre à jour le service
echo -e "${GREEN}[8/10]${NC} Creating/updating ECS service..."

# Vérifier si le service existe
if aws ecs describe-services \
    --cluster $CLUSTER_NAME \
    --services $SERVICE_NAME \
    --region $AWS_REGION | grep -q "ACTIVE"; then
    
    echo -e "${YELLOW}Updating existing service...${NC}"
    aws ecs update-service \
        --cluster $CLUSTER_NAME \
        --service $SERVICE_NAME \
        --task-definition $TASK_FAMILY \
        --desired-count $DESIRED_COUNT \
        --region $AWS_REGION \
        --force-new-deployment
else
    echo -e "${YELLOW}Creating new service...${NC}"
    aws ecs create-service \
        --cluster $CLUSTER_NAME \
        --service-name $SERVICE_NAME \
        --task-definition $TASK_FAMILY \
        --desired-count $DESIRED_COUNT \
        --capacity-provider-strategy capacityProvider=FARGATE_SPOT,weight=1 \
        --network-configuration "awsvpcConfiguration={
            subnets=[$SUBNETS],
            securityGroups=[$SG_ID],
            assignPublicIp=ENABLED
        }" \
        --region $AWS_REGION
fi

echo -e "${GREEN}✅ Service deployed${NC}"

# 8. Attendre que la tâche soit RUNNING
echo -e "${GREEN}[9/10]${NC} Waiting for task to be RUNNING (this may take 2-3 minutes)..."
sleep 60

for i in {1..20}; do
    TASK_STATUS=$(aws ecs describe-services \
        --cluster $CLUSTER_NAME \
        --services $SERVICE_NAME \
        --region $AWS_REGION \
        --query 'services[0].{Running:runningCount,Desired:desiredCount}' \
        --output text)
    
    if echo "$TASK_STATUS" | grep -q "1.*1"; then
        echo -e "${GREEN}✅ Task is RUNNING!${NC}"
        break
    fi
    
    if [ $i -eq 20 ]; then
        echo -e "${YELLOW}⚠️  Task not running yet. Check CloudWatch logs for details.${NC}"
        break
    fi
    
    echo -e "${YELLOW}⏳ Waiting... ($i/20)${NC}"
    sleep 15
done

# 9. Récupérer l'IP publique
echo -e "${GREEN}[10/10]${NC} Getting public IP..."

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
    
    echo ""
    echo -e "${BLUE}╔════════════════════════════════════════════════╗${NC}"
    echo -e "${BLUE}║           DEPLOYMENT SUCCESSFUL! 🎉            ║${NC}"
    echo -e "${BLUE}╚════════════════════════════════════════════════╝${NC}"
    echo ""
    echo -e "${GREEN}📱 Streamlit UI:${NC} ${YELLOW}http://${PUBLIC_IP}:8501${NC}"
    echo -e "${GREEN}🔧 API:${NC}          ${YELLOW}http://${PUBLIC_IP}:8000${NC}"
    echo -e "${GREEN}📚 API Docs:${NC}     ${YELLOW}http://${PUBLIC_IP}:8000/docs${NC}"
    echo ""
    echo -e "${YELLOW}⏰ Note: First startup takes 2-3 minutes (Ollama model download)${NC}"
    echo ""
else
    echo -e "${YELLOW}⚠️  Task ARN not found yet. The service is deploying.${NC}"
    echo -e "${YELLOW}Run this command in a few minutes to get the IP:${NC}"
    echo ""
    echo "TASK_ARN=\$(aws ecs list-tasks --cluster $CLUSTER_NAME --service-name $SERVICE_NAME --region $AWS_REGION --query 'taskArns[0]' --output text)"
    echo "ENI_ID=\$(aws ecs describe-tasks --cluster $CLUSTER_NAME --tasks \$TASK_ARN --region $AWS_REGION --query 'tasks[0].attachments[0].details[?name==\`networkInterfaceId\`].value' --output text)"
    echo "aws ec2 describe-network-interfaces --network-interface-ids \$ENI_ID --region $AWS_REGION --query 'NetworkInterfaces[0].Association.PublicIp' --output text"
fi

echo ""
echo -e "${GREEN}📊 View logs:${NC}"
echo -e "   aws logs tail /ecs/${TASK_FAMILY} --region $AWS_REGION --follow"
echo ""
echo -e "${GREEN}🛑 Stop service (save \$\$\$):${NC}"
echo -e "   aws ecs update-service --cluster $CLUSTER_NAME --service $SERVICE_NAME --desired-count 0 --region $AWS_REGION"
echo ""
echo -e "${GREEN}🔄 Restart service:${NC}"
echo -e "   aws ecs update-service --cluster $CLUSTER_NAME --service $SERVICE_NAME --desired-count 1 --region $AWS_REGION"
echo ""
