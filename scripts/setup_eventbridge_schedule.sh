#!/bin/bash
# Configuration AWS EventBridge pour démarrage/arrêt automatique
# Horaires: 10h-18h GMT (8h/jour)

set -e

CLUSTER_NAME="video-summarizer-cluster"
SERVICE_NAME="video-summarizer-app"
AWS_REGION="${AWS_REGION:-us-east-1}"
ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)

echo "🔧 Setting up EventBridge scheduled tasks..."

# 1. Créer un rôle IAM pour EventBridge
echo "[1/5] Creating IAM role for EventBridge..."

ROLE_NAME="VideoSummarizerSchedulerRole"

if ! aws iam get-role --role-name $ROLE_NAME &> /dev/null; then
    aws iam create-role \
        --role-name $ROLE_NAME \
        --assume-role-policy-document '{
            "Version": "2012-10-17",
            "Statement": [{
                "Effect": "Allow",
                "Principal": {"Service": "events.amazonaws.com"},
                "Action": "sts:AssumeRole"
            }]
        }'
    
    # Créer une policy inline
    aws iam put-role-policy \
        --role-name $ROLE_NAME \
        --policy-name ECSTaskSchedulingPolicy \
        --policy-document '{
            "Version": "2012-10-17",
            "Statement": [{
                "Effect": "Allow",
                "Action": [
                    "ecs:UpdateService",
                    "ecs:DescribeServices"
                ],
                "Resource": "*"
            }]
        }'
    
    echo "✅ Role created"
else
    echo "✓ Role already exists"
fi

ROLE_ARN="arn:aws:iam::${ACCOUNT_ID}:role/${ROLE_NAME}"

# 2. Créer une règle pour démarrer à 10h GMT
echo "[2/5] Creating start rule (10:00 GMT)..."

aws events put-rule \
    --name video-summarizer-start-10h \
    --description "Start Video Summarizer at 10:00 GMT" \
    --schedule-expression "cron(0 10 * * ? *)" \
    --state ENABLED \
    --region $AWS_REGION

# 3. Créer une règle pour arrêter à 18h GMT
echo "[3/5] Creating stop rule (18:00 GMT)..."

aws events put-rule \
    --name video-summarizer-stop-18h \
    --description "Stop Video Summarizer at 18:00 GMT" \
    --schedule-expression "cron(0 18 * * ? *)" \
    --state ENABLED \
    --region $AWS_REGION

# 4. Créer les targets pour démarrer
echo "[4/5] Configuring start target..."

cat > /tmp/start-target.json <<EOF
{
    "Cluster": "${CLUSTER_NAME}",
    "Service": "${SERVICE_NAME}",
    "DesiredCount": 1
}
EOF

aws events put-targets \
    --rule video-summarizer-start-10h \
    --targets "Id=1,Arn=arn:aws:ecs:${AWS_REGION}:${ACCOUNT_ID}:cluster/${CLUSTER_NAME},RoleArn=${ROLE_ARN},EcsParameters={TaskDefinitionArn=arn:aws:ecs:${AWS_REGION}:${ACCOUNT_ID}:task-definition/video-summarizer-task,LaunchType=FARGATE,NetworkConfiguration={awsvpcConfiguration={Subnets=[subnet-xxx],SecurityGroups=[sg-xxx],AssignPublicIp=ENABLED}}}" \
    --region $AWS_REGION 2>/dev/null || echo "⚠️  Manual target configuration required"

# 5. Créer les targets pour arrêter
echo "[5/5] Configuring stop target..."

cat > /tmp/stop-target.json <<EOF
{
    "Cluster": "${CLUSTER_NAME}",
    "Service": "${SERVICE_NAME}",
    "DesiredCount": 0
}
EOF

aws events put-targets \
    --rule video-summarizer-stop-18h \
    --targets "Id=1,Arn=arn:aws:ecs:${AWS_REGION}:${ACCOUNT_ID}:cluster/${CLUSTER_NAME},RoleArn=${ROLE_ARN},EcsParameters={TaskDefinitionArn=arn:aws:ecs:${AWS_REGION}:${ACCOUNT_ID}:task-definition/video-summarizer-task,LaunchType=FARGATE,NetworkConfiguration={awsvpcConfiguration={Subnets=[subnet-xxx],SecurityGroups=[sg-xxx],AssignPublicIp=ENABLED}}}" \
    --region $AWS_REGION 2>/dev/null || echo "⚠️  Manual target configuration required"

echo ""
echo "✅ EventBridge rules created!"
echo ""
echo "📋 Summary:"
echo "  - Start rule: video-summarizer-start-10h (10:00 GMT daily)"
echo "  - Stop rule:  video-summarizer-stop-18h (18:00 GMT daily)"
echo ""
echo "⚠️  IMPORTANT: You need to manually configure the targets in AWS Console:"
echo ""
echo "1. Go to: https://console.aws.amazon.com/events/home?region=${AWS_REGION}#/rules"
echo "2. Click on 'video-summarizer-start-10h'"
echo "3. Click 'Targets' > 'Edit'"
echo "4. Select target type: 'ECS task'"
echo "5. Configure:"
echo "   - Cluster: ${CLUSTER_NAME}"
echo "   - Task definition: video-summarizer-task (latest)"
echo "   - Launch type: FARGATE"
echo "   - Platform version: LATEST"
echo "   - Subnets: Select your VPC subnets"
echo "   - Security groups: Select video-summarizer-sg"
echo "   - Auto-assign public IP: ENABLED"
echo "6. Repeat for 'video-summarizer-stop-18h'"
echo ""
echo "💡 Alternative: Use simple script scheduling"
echo "   ./scripts/schedule_service.sh start   # Manual start"
echo "   ./scripts/schedule_service.sh stop    # Manual stop"
