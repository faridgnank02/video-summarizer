#!/bin/bash
# Script de démarrage pour Video Summarizer
# Gère Ollama + Streamlit + FastAPI dans un seul conteneur

set -e

echo "🚀 Starting Video Summarizer..."

# Couleurs pour les logs
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# 1. Démarrer Ollama en arrière-plan
echo -e "${GREEN}[1/5]${NC} Starting Ollama server..."
ollama serve > /app/logs/ollama.log 2>&1 &
OLLAMA_PID=$!

# Attendre que Ollama soit prêt
echo -e "${YELLOW}⏳${NC} Waiting for Ollama to be ready..."
for i in {1..30}; do
    if curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
        echo -e "${GREEN}✅${NC} Ollama server is ready!"
        break
    fi
    if [ $i -eq 30 ]; then
        echo -e "${RED}❌${NC} Ollama failed to start"
        exit 1
    fi
    sleep 2
done

# 2. Télécharger le modèle Ollama si nécessaire
echo -e "${GREEN}[2/5]${NC} Checking Ollama model: ${OLLAMA_MODEL}..."
if ! ollama list | grep -q "${OLLAMA_MODEL}"; then
    echo -e "${YELLOW}⬇️${NC} Downloading model ${OLLAMA_MODEL}... (this may take a few minutes)"
    ollama pull ${OLLAMA_MODEL}
    echo -e "${GREEN}✅${NC} Model ${OLLAMA_MODEL} downloaded successfully!"
else
    echo -e "${GREEN}✅${NC} Model ${OLLAMA_MODEL} already available"
fi

# 3. Vérifier les variables d'environnement
echo -e "${GREEN}[3/5]${NC} Checking configuration..."
if [ -z "$OPENAI_API_KEY" ]; then
    echo -e "${YELLOW}⚠️${NC} OPENAI_API_KEY not set - OpenAI features will be disabled"
else
    echo -e "${GREEN}✅${NC} OpenAI API key configured"
fi

# 4. Démarrer FastAPI en arrière-plan
echo -e "${GREEN}[4/5]${NC} Starting FastAPI server..."
cd /app
python -m uvicorn src.api.main:app \
    --host ${API_HOST:-0.0.0.0} \
    --port ${API_PORT:-8000} \
    --log-level info \
    > /app/logs/api.log 2>&1 &
API_PID=$!

# Attendre que l'API soit prête
echo -e "${YELLOW}⏳${NC} Waiting for API to be ready..."
for i in {1..20}; do
    if curl -s http://localhost:8000/health > /dev/null 2>&1; then
        echo -e "${GREEN}✅${NC} API server is ready!"
        break
    fi
    sleep 1
done

# 5. Démarrer Streamlit (processus principal)
echo -e "${GREEN}[5/5]${NC} Starting Streamlit interface..."
echo -e "${GREEN}🎉${NC} Video Summarizer is starting up!"
echo -e "${GREEN}📱${NC} Streamlit UI will be available on port ${STREAMLIT_SERVER_PORT:-8501}"
echo -e "${GREEN}🔧${NC} API available on port ${API_PORT:-8000}"
echo ""

# Streamlit en premier plan (pour que le conteneur reste actif)
streamlit run /app/src/ui/streamlit_app.py \
    --server.port=${STREAMLIT_SERVER_PORT:-8501} \
    --server.address=${STREAMLIT_SERVER_ADDRESS:-0.0.0.0} \
    --server.headless=true \
    --server.runOnSave=false \
    --browser.gatherUsageStats=false \
    --logger.level=info

# Si Streamlit s'arrête, arrêter les autres processus
echo -e "${YELLOW}⚠️${NC} Streamlit stopped, shutting down..."
kill $OLLAMA_PID $API_PID 2>/dev/null || true
