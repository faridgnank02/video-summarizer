"""
API FastAPI Simplifiée pour Production - Video Summarizer
Optimisée pour déploiement AWS avec OpenAI uniquement
"""

import os
import time
import logging
from typing import Optional, Dict, Any
from datetime import datetime

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn

# Configuration logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import modèles
try:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    
    from models.openai_model import OpenAISummarizer
    from data.preprocessing import TextPreprocessor
except ImportError as e:
    logger.error(f"Erreur import: {e}")
    raise

# Configuration API
app = FastAPI(
    title="Video Summarizer API",
    description="API de résumé automatique avec OpenAI GPT",
    version="3.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Variables globales
summarizer: Optional[OpenAISummarizer] = None
preprocessor: Optional[TextPreprocessor] = None
start_time = time.time()

# Modèles Pydantic

class SummarizeRequest(BaseModel):
    """Requête de résumé"""
    text: str = Field(..., description="Texte à résumer", min_length=50, max_length=50000)
    length: str = Field("short", description="Longueur du résumé: 'short' ou 'long'")
    language: Optional[str] = Field(None, description="Langue: 'fr' ou 'en' (auto-détecté si non spécifié)")

class SummarizeResponse(BaseModel):
    """Réponse de résumé"""
    success: bool
    summary: str
    original_length: int
    summary_length: int
    compression_ratio: float
    model_used: str
    processing_time: float
    timestamp: str

class HealthResponse(BaseModel):
    """Réponse health check"""
    status: str
    uptime: float
    openai_configured: bool
    version: str
    timestamp: str

class ErrorResponse(BaseModel):
    """Réponse d'erreur"""
    success: bool = False
    error: str
    detail: Optional[str] = None
    timestamp: str

# Events

@app.on_event("startup")
async def startup_event():
    """Initialisation au démarrage"""
    global summarizer, preprocessor
    
    logger.info("🚀 Démarrage API Video Summarizer")
    
    try:
        # Vérifier clé OpenAI
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            logger.warning("⚠️  OPENAI_API_KEY non définie - API limitée")
        else:
            logger.info("✅ OpenAI API key configurée")
        
        # Initialiser composants
        summarizer = OpenAISummarizer(
            model_name=os.getenv('OPENAI_MODEL', 'gpt-4o-mini')
        )
        preprocessor = TextPreprocessor()
        
        logger.info("✅ API initialisée avec succès")
        
    except Exception as e:
        logger.error(f"❌ Erreur initialisation: {e}")
        raise

@app.on_event("shutdown")
async def shutdown_event():
    """Nettoyage à l'arrêt"""
    logger.info("🛑 Arrêt de l'API")

# Endpoints

@app.get("/")
async def root():
    """Point d'entrée de l'API"""
    return {
        "service": "Video Summarizer API",
        "version": "3.0.0",
        "status": "running",
        "endpoints": {
            "health": "/health",
            "docs": "/docs",
            "summarize": "/api/v1/summarize"
        }
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check pour AWS ECS"""
    uptime = time.time() - start_time
    
    return HealthResponse(
        status="healthy",
        uptime=uptime,
        openai_configured=bool(os.getenv('OPENAI_API_KEY')),
        version="3.0.0",
        timestamp=datetime.now().isoformat()
    )

@app.post("/api/v1/summarize", response_model=SummarizeResponse)
async def summarize(request: SummarizeRequest):
    """
    Génère un résumé du texte fourni
    
    - **text**: Texte à résumer (50-50000 caractères)
    - **length**: 'short' (100-200 mots) ou 'long' (300-500 mots)
    - **language**: 'fr' ou 'en' (auto-détecté si non spécifié)
    """
    start = time.time()
    
    try:
        if not summarizer:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Service de résumé non disponible"
            )
        
        # Vérifier clé OpenAI
        if not os.getenv('OPENAI_API_KEY'):
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="OpenAI API key non configurée"
            )
        
        # Prétraitement
        processed = preprocessor.preprocess(request.text)
        
        # Génération résumé
        summary = summarizer.summarize(
            text=processed.text,
            summary_type=request.length,
            language=request.language
        )
        
        processing_time = time.time() - start
        
        logger.info(
            f"Résumé généré: {len(request.text)} -> {len(summary)} chars "
            f"en {processing_time:.2f}s"
        )
        
        return SummarizeResponse(
            success=True,
            summary=summary,
            original_length=len(request.text),
            summary_length=len(summary),
            compression_ratio=len(summary) / len(request.text),
            model_used=summarizer.model_name,
            processing_time=processing_time,
            timestamp=datetime.now().isoformat()
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Erreur résumé: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Erreur lors de la génération du résumé: {str(e)}"
        )

@app.get("/api/v1/stats")
async def get_stats():
    """Statistiques d'utilisation"""
    if not summarizer:
        return {"error": "Summarizer non initialisé"}
    
    return {
        "uptime": time.time() - start_time,
        "usage_stats": summarizer.get_usage_stats(),
        "model_info": summarizer.get_model_info()
    }

# Point d'entrée pour exécution directe
if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=port,
        log_level="info"
    )
