# Multi-stage Dockerfile pour Video Summarizer
# Architecture tout-en-un : Streamlit + FastAPI + Ollama
# Optimisé pour AWS ECS Fargate (AMD64) - Taille réduite

# IMPORTANT: Forcer architecture AMD64 pour AWS
FROM --platform=linux/amd64 python:3.10-slim AS base

# Métadonnées
LABEL maintainer="faridgnank02"
LABEL description="Video Summarizer - Streamlit + FastAPI + Ollama (gemma3:1b)"
LABEL version="1.0.0"

# Variables d'environnement
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    DEBIAN_FRONTEND=noninteractive

# Installer les dépendances système
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    wget \
    ca-certificates \
    supervisor \
    nginx \
    && rm -rf /var/lib/apt/lists/*

# Installer Ollama
RUN curl -fsSL https://ollama.com/install.sh | sh

# Créer répertoire de travail
WORKDIR /app

# Copier requirements
COPY requirements.txt .

# Installer UNIQUEMENT les dépendances essentielles (sans PyTorch pour réduire la taille)
RUN pip install --no-cache-dir \
    streamlit>=1.25.0 \
    fastapi \
    uvicorn \
    requests \
    pyyaml \
    python-dotenv \
    youtube-transcript-api>=0.6.0 \
    yt-dlp>=2023.7.6 \
    openai>=0.27.0 \
    psutil>=5.9.0 \
    pydantic \
    scikit-learn \
    pandas \
    numpy \
    && rm -rf /root/.cache/pip

# NE PAS installer spaCy/transformers/torch pour économiser ~8GB
# L'évaluation qualité est désactivée dans cette version

# Copier le code de l'application
COPY src/ /app/src/
COPY config/ /app/config/
COPY scripts/ /app/scripts/

# Créer les répertoires nécessaires
RUN mkdir -p /app/data/cache \
    /app/data/assets \
    /app/logs \
    /app/temp \
    /var/log/supervisor \
    /var/log/nginx \
    /root/.ollama

# Configuration Nginx (reverse proxy interne)
COPY nginx.conf /etc/nginx/nginx.conf

# Configuration Supervisor (gestion multi-process)
COPY supervisord.conf /etc/supervisor/conf.d/supervisord.conf

# Script de démarrage
COPY scripts/startup.sh /app/startup.sh
RUN chmod +x /app/startup.sh /app/scripts/*.sh

# Exposer les ports
# 8501 : Streamlit (interface principale)
# 8000 : FastAPI (API REST)
# 11434 : Ollama (interne)
EXPOSE 8501 8000

# Variables d'environnement par défaut
ENV OLLAMA_HOST=0.0.0.0:11434 \
    OLLAMA_MODELS=/root/.ollama/models \
    OLLAMA_MODEL=gemma3:1b \
    STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    API_HOST=0.0.0.0 \
    API_PORT=8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=180s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

# Commande de démarrage
CMD ["/app/startup.sh"]

# Taille attendue: ~2-3GB (vs 11GB avant optimisation)
# - Python base: ~150MB
# - Ollama: ~600MB  
# - Modèle gemma3:1b: ~1.1GB
# - Dépendances Python: ~300MB
# - Application: ~50MB
