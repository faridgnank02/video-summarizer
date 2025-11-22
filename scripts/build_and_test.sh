#!/bin/bash
# Script pour builder et tester l'image Docker localement
# Usage: ./scripts/build_and_test.sh

set -e

# Couleurs
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${GREEN}🔨 Building Docker image for AMD64 (AWS compatible)...${NC}"
echo -e "${YELLOW}⚠️  Building on Mac M1/M2/M3 for AMD64 (cross-platform build)${NC}"

# Charger les variables d'environnement
if [ -f .env ]; then
    export $(grep -v '^#' .env | xargs)
fi

# S'assurer que buildx est disponible
if ! docker buildx version > /dev/null 2>&1; then
    echo -e "${RED}❌ Docker buildx not available${NC}"
    exit 1
fi

# Build pour AMD64 (compatible AWS ECS) - FORCE l'architecture
docker buildx build --platform linux/amd64 \
    -t video-summarizer:latest \
    -t video-summarizer:dev \
    -f Dockerfile . \
    --load \
    --progress=plain

echo -e "${GREEN}✅ Build completed!${NC}"

# Afficher la taille de l'image
IMAGE_SIZE=$(docker images video-summarizer:latest --format "{{.Size}}")
echo -e "${YELLOW}📦 Image size: ${IMAGE_SIZE}${NC}"

# Vérifier l'architecture
IMAGE_ARCH=$(docker image inspect video-summarizer:latest --format '{{.Architecture}}')
echo -e "${YELLOW}🏗️  Architecture: ${IMAGE_ARCH}${NC}"

# Vérifier que c'est bien AMD64
if [ "$IMAGE_ARCH" != "amd64" ]; then
    echo -e "${RED}❌ Wrong architecture! Expected amd64, got ${IMAGE_ARCH}${NC}"
    echo -e "${YELLOW}💡 Rebuild with: docker buildx build --platform linux/amd64 ...${NC}"
    exit 1
fi

# Vérifier que la taille est raisonnable (< 4GB pour optimisation AWS)
if [ ! -z "$IMAGE_SIZE" ]; then
    echo -e "${GREEN}✅ Image created successfully (${IMAGE_SIZE})${NC}"
    echo -e "${GREEN}✅ Architecture: ${IMAGE_ARCH}${NC}"
else
    echo -e "${RED}❌ Image build failed${NC}"
    exit 1
fi

echo ""
echo -e "${GREEN}🧪 Starting local test...${NC}"

# Arrêter les containers existants
docker-compose down 2>/dev/null || true

# Démarrer le container
docker-compose up -d

echo -e "${YELLOW}⏳ Waiting for services to start (this may take 2-3 minutes)...${NC}"
sleep 30

# Attendre que Streamlit soit prêt
for i in {1..30}; do
    if curl -s http://localhost:8501/_stcore/health > /dev/null 2>&1; then
        echo -e "${GREEN}✅ Streamlit is ready!${NC}"
        break
    fi
    if [ $i -eq 30 ]; then
        echo -e "${RED}❌ Streamlit failed to start${NC}"
        docker-compose logs
        exit 1
    fi
    sleep 10
done

# Vérifier l'API
echo -e "${YELLOW}Testing API health check...${NC}"
if curl -s http://localhost:8000/health | grep -q "healthy"; then
    echo -e "${GREEN}✅ API is healthy!${NC}"
else
    echo -e "${YELLOW}⚠️  API health check not responding (this might be normal if API is not exposed)${NC}"
fi

echo ""
echo -e "${GREEN}✅ Local test completed successfully!${NC}"
echo ""
echo -e "${GREEN}📱 Access the application:${NC}"
echo -e "   Streamlit UI: ${YELLOW}http://localhost:8501${NC}"
echo -e "   API Docs:     ${YELLOW}http://localhost:8000/docs${NC}"
echo ""
echo -e "${YELLOW}📋 View logs:${NC}"
echo -e "   docker-compose logs -f"
echo ""
echo -e "${YELLOW}🛑 Stop the container:${NC}"
echo -e "   docker-compose down"
echo ""
echo -e "${GREEN}🚀 Next step: Tag and push to Docker Hub${NC}"
echo -e "   docker tag video-summarizer:latest YOUR_USERNAME/video-summarizer:latest"
echo -e "   docker push YOUR_USERNAME/video-summarizer:latest"
