"""
API REST complète pour Video Summarizer
Endpoints pour résumé, évaluation, monitoring et administration
"""

import os
import time
import json
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime

from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field
import uvicorn

# Imports locaux
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.model_manager import ModelManager, SummaryRequest, SummaryResponse
from data.ingestion import DataIngestion
from data.preprocessing import TextPreprocessor
from evaluation.evaluator import SummaryEvaluator, EvaluationReport
from monitoring.metrics import MetricsCollector, AlertManager, setup_default_alerts

logger = logging.getLogger(__name__)

# Configuration de l'API
app = FastAPI(
    title="Video Summarizer API",
    description="API professionnelle pour le résumé automatique de vidéos et textes",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS pour les applications web
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Sécurité basique
security = HTTPBearer(auto_error=False)

# Instances globales
model_manager = None
ingestion = None
preprocessor = None
evaluator = None
metrics_collector = None
alert_manager = None

# Modèles Pydantic pour l'API

class TextSummaryRequest(BaseModel):
    """Requête de résumé de texte"""
    text: str = Field(..., description="Texte à résumer", min_length=50)
    model_type: str = Field("auto", description="Type de modèle (auto, openai, led)")
    summary_length: str = Field("short", description="Longueur du résumé (short, long)")
    user_id: Optional[str] = Field(None, description="ID utilisateur pour le tracking")
    evaluate: bool = Field(False, description="Évaluer la qualité du résumé")

class YouTubeSummaryRequest(BaseModel):
    """Requête de résumé YouTube"""
    url: str = Field(..., description="URL de la vidéo YouTube")
    model_type: str = Field("auto", description="Type de modèle")
    summary_length: str = Field("short", description="Longueur du résumé")
    user_id: Optional[str] = Field(None, description="ID utilisateur")
    evaluate: bool = Field(False, description="Évaluer la qualité")

class BatchSummaryRequest(BaseModel):
    """Requête de résumé en lot"""
    items: List[Dict[str, Any]] = Field(..., description="Liste des éléments à résumer")
    model_type: str = Field("auto", description="Type de modèle")
    summary_length: str = Field("short", description="Longueur du résumé")
    user_id: Optional[str] = Field(None, description="ID utilisateur")

class EvaluationRequest(BaseModel):
    """Requête d'évaluation de résumé"""
    original_text: str = Field(..., description="Texte original")
    generated_summary: str = Field(..., description="Résumé généré")
    reference_summary: Optional[str] = Field(None, description="Résumé de référence")
    model_name: str = Field("unknown", description="Nom du modèle")

class APIResponse(BaseModel):
    """Réponse API standardisée"""
    success: bool
    data: Optional[Dict[str, Any]] = None
    message: str = ""
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())
    processing_time: Optional[float] = None

class HealthResponse(BaseModel):
    """Réponse de santé du système"""
    status: str
    uptime: float
    system_health: Dict[str, Any]
    models_status: Dict[str, bool]
    metrics: Dict[str, Any]

# Dépendances

async def get_api_key(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Vérification basique de l'API key (optionnelle)"""
    if credentials and credentials.credentials:
        # TODO: Implémenter la vérification des clés API
        return credentials.credentials
    return None

# Événements de l'application

@app.on_event("startup")
async def startup_event():
    """Initialisation au démarrage"""
    global model_manager, ingestion, preprocessor, evaluator, metrics_collector, alert_manager
    
    logger.info("Démarrage de l'API Video Summarizer")
    
    try:
        # Initialiser les composants
        model_manager = ModelManager()
        ingestion = DataIngestion()
        preprocessor = TextPreprocessor()
        evaluator = SummaryEvaluator()
        
        # Monitoring
        metrics_collector = MetricsCollector()
        metrics_collector.start_collection()
        
        alert_manager = AlertManager(metrics_collector)
        setup_default_alerts(alert_manager)
        
        logger.info("API initialisée avec succès")
        
    except Exception as e:
        logger.error(f"Erreur lors de l'initialisation: {e}")
        raise

@app.on_event("shutdown")
async def shutdown_event():
    """Nettoyage à l'arrêt"""
    logger.info("Arrêt de l'API")
    
    if metrics_collector:
        metrics_collector.stop_collection()

# Endpoints principaux

@app.get("/", response_model=APIResponse)
async def root():
    """Point d'entrée de l'API"""
    return APIResponse(
        success=True,
        message="Video Summarizer API v2.0 - Prêt",
        data={
            "version": "2.0.0",
            "docs": "/docs",
            "health": "/health",
            "endpoints": {
                "text_summary": "/api/v1/summarize/text",
                "youtube_summary": "/api/v1/summarize/youtube",
                "batch_summary": "/api/v1/summarize/batch",
                "evaluation": "/api/v1/evaluate",
                "metrics": "/api/v1/metrics"
            }
        }
    )

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Vérification de la santé du système"""
    start_time = time.time()
    
    # État des modèles
    models_status = {}
    if model_manager:
        from models.model_manager import ModelType
        led_available, _ = model_manager.is_model_available(ModelType.LED)
        openai_available, _ = model_manager.is_model_available(ModelType.OPENAI)
        models_status = {
            "led": led_available,
            "openai": openai_available
        }
    
    # Métriques système
    system_health = {}
    current_metrics = {}
    if metrics_collector:
        current_stats = metrics_collector.get_current_stats()
        system_health = current_stats.get("health", {})
        current_metrics = current_stats
    
    # Uptime (approximation)
    uptime = time.time() - start_time
    
    return HealthResponse(
        status=system_health.get("status", "unknown"),
        uptime=uptime,
        system_health=system_health,
        models_status=models_status,
        metrics=current_metrics
    )

@app.post("/api/v1/summarize/text", response_model=APIResponse)
async def summarize_text(
    request: TextSummaryRequest,
    background_tasks: BackgroundTasks,
    api_key: Optional[str] = Depends(get_api_key)
):
    """Résume un texte"""
    start_time = time.time()
    
    try:
        # Préprocessing
        processed = preprocessor.preprocess(request.text)
        
        # Résumé
        summary_request = SummaryRequest(
            text=processed.text,
            model_type=request.model_type,
            summary_length=request.summary_length
        )
        
        response = model_manager.summarize(summary_request)
        processing_time = time.time() - start_time
        
        # Évaluation optionnelle
        evaluation_report = None
        if request.evaluate:
            evaluation_report = evaluator.evaluate_summary(
                original_text=request.text,
                generated_summary=response.summary,
                model_name=response.model_used,
                processing_time=processing_time
            )
        
        # Métriques
        if metrics_collector:
            background_tasks.add_task(
                metrics_collector.record_performance,
                operation="text_summary",
                model_name=response.model_used,
                processing_time=processing_time,
                input_length=len(request.text),
                output_length=len(response.summary),
                success=True,
                user_id=request.user_id
            )
        
        # Réponse
        data = {
            "summary": response.summary,
            "model_used": response.model_used,
            "original_length": len(request.text),
            "summary_length": len(response.summary),
            "compression_ratio": len(response.summary) / len(request.text),
            "word_count": response.word_count
        }
        
        if evaluation_report:
            data["evaluation"] = {
                "overall_score": evaluation_report.metrics.overall_score,
                "semantic_similarity": evaluation_report.metrics.semantic_similarity,
                "coherence_score": evaluation_report.metrics.coherence_score,
                "recommendations": evaluation_report.recommendations
            }
        
        return APIResponse(
            success=True,
            data=data,
            message="Résumé généré avec succès",
            processing_time=processing_time
        )
        
    except Exception as e:
        processing_time = time.time() - start_time
        
        # Métriques d'erreur
        if metrics_collector:
            background_tasks.add_task(
                metrics_collector.record_performance,
                operation="text_summary",
                model_name="unknown",
                processing_time=processing_time,
                input_length=len(request.text),
                output_length=0,
                success=False,
                error_message=str(e),
                user_id=request.user_id
            )
        
        logger.error(f"Erreur résumé texte: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Erreur lors du résumé: {str(e)}"
        )

@app.post("/api/v1/summarize/youtube", response_model=APIResponse)
async def summarize_youtube(
    request: YouTubeSummaryRequest,
    background_tasks: BackgroundTasks,
    api_key: Optional[str] = Depends(get_api_key)
):
    """Résume une vidéo YouTube"""
    start_time = time.time()
    
    try:
        # Ingestion YouTube
        video_data = ingestion.process_youtube_url(request.url)
        
        # Préprocessing
        processed = preprocessor.preprocess(video_data.transcript)
        
        # Résumé
        summary_request = SummaryRequest(
            text=processed.text,
            model_type=request.model_type,
            summary_length=request.summary_length
        )
        
        response = model_manager.summarize(summary_request)
        processing_time = time.time() - start_time
        
        # Évaluation optionnelle
        evaluation_report = None
        if request.evaluate:
            evaluation_report = evaluator.evaluate_summary(
                original_text=video_data.transcript,
                generated_summary=response.summary,
                model_name=response.model_used,
                processing_time=processing_time
            )
        
        # Métriques
        if metrics_collector:
            background_tasks.add_task(
                metrics_collector.record_performance,
                operation="youtube_summary",
                model_name=response.model_used,
                processing_time=processing_time,
                input_length=len(video_data.transcript),
                output_length=len(response.summary),
                success=True,
                user_id=request.user_id
            )
        
        # Réponse
        data = {
            "summary": response.summary,
            "video_info": {
                "title": video_data.title,
                "duration": video_data.duration,
                "language": video_data.language,
                "url": request.url
            },
            "model_used": response.model_used,
            "transcript_length": len(video_data.transcript),
            "summary_length": len(response.summary),
            "compression_ratio": len(response.summary) / len(video_data.transcript)
        }
        
        if evaluation_report:
            data["evaluation"] = {
                "overall_score": evaluation_report.metrics.overall_score,
                "semantic_similarity": evaluation_report.metrics.semantic_similarity,
                "coherence_score": evaluation_report.metrics.coherence_score,
                "recommendations": evaluation_report.recommendations
            }
        
        return APIResponse(
            success=True,
            data=data,
            message="Résumé YouTube généré avec succès",
            processing_time=processing_time
        )
        
    except Exception as e:
        processing_time = time.time() - start_time
        
        # Métriques d'erreur
        if metrics_collector:
            background_tasks.add_task(
                metrics_collector.record_performance,
                operation="youtube_summary",
                model_name="unknown",
                processing_time=processing_time,
                input_length=len(request.url),
                output_length=0,
                success=False,
                error_message=str(e),
                user_id=request.user_id
            )
        
        logger.error(f"Erreur résumé YouTube: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Erreur lors du résumé YouTube: {str(e)}"
        )

@app.post("/api/v1/summarize/batch", response_model=APIResponse)
async def summarize_batch(
    request: BatchSummaryRequest,
    background_tasks: BackgroundTasks,
    api_key: Optional[str] = Depends(get_api_key)
):
    """Résume plusieurs éléments en lot"""
    start_time = time.time()
    
    try:
        results = []
        total_input_length = 0
        total_output_length = 0
        
        for i, item in enumerate(request.items):
            item_start = time.time()
            
            try:
                # Déterminer le type d'élément
                if "text" in item:
                    # Texte direct
                    text = item["text"]
                    processed = preprocessor.preprocess(text)
                    
                elif "url" in item:
                    # URL YouTube
                    video_data = ingestion.process_youtube_url(item["url"])
                    text = video_data.transcript
                    processed = preprocessor.preprocess(text)
                    
                else:
                    raise ValueError("Item doit contenir 'text' ou 'url'")
                
                # Résumé
                summary_request = SummaryRequest(
                    text=processed.text,
                    model_type=request.model_type,
                    summary_length=request.summary_length
                )
                
                response = model_manager.summarize(summary_request)
                item_time = time.time() - item_start
                
                total_input_length += len(text)
                total_output_length += len(response.summary)
                
                results.append({
                    "index": i,
                    "success": True,
                    "summary": response.summary,
                    "model_used": response.model_used,
                    "processing_time": item_time,
                    "original_length": len(text),
                    "summary_length": len(response.summary)
                })
                
            except Exception as e:
                results.append({
                    "index": i,
                    "success": False,
                    "error": str(e),
                    "processing_time": time.time() - item_start
                })
        
        total_time = time.time() - start_time
        successful_count = sum(1 for r in results if r["success"])
        
        # Métriques
        if metrics_collector:
            background_tasks.add_task(
                metrics_collector.record_performance,
                operation="batch_summary",
                model_name=request.model_type,
                processing_time=total_time,
                input_length=total_input_length,
                output_length=total_output_length,
                success=successful_count == len(request.items),
                user_id=request.user_id
            )
        
        return APIResponse(
            success=True,
            data={
                "results": results,
                "summary_stats": {
                    "total_items": len(request.items),
                    "successful": successful_count,
                    "failed": len(request.items) - successful_count,
                    "total_input_length": total_input_length,
                    "total_output_length": total_output_length,
                    "avg_compression_ratio": total_output_length / total_input_length if total_input_length > 0 else 0
                }
            },
            message=f"Traitement en lot terminé: {successful_count}/{len(request.items)} réussis",
            processing_time=total_time
        )
        
    except Exception as e:
        logger.error(f"Erreur batch: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Erreur lors du traitement en lot: {str(e)}"
        )

@app.post("/api/v1/evaluate", response_model=APIResponse)
async def evaluate_summary(
    request: EvaluationRequest,
    api_key: Optional[str] = Depends(get_api_key)
):
    """Évalue la qualité d'un résumé"""
    start_time = time.time()
    
    try:
        report = evaluator.evaluate_summary(
            original_text=request.original_text,
            generated_summary=request.generated_summary,
            reference_summary=request.reference_summary,
            model_name=request.model_name
        )
        
        processing_time = time.time() - start_time
        
        return APIResponse(
            success=True,
            data={
                "summary_id": report.summary_id,
                "overall_score": report.metrics.overall_score,
                "metrics": {
                    "rouge_scores": {
                        "rouge-1": {
                            "f": report.metrics.rouge_1_f,
                            "p": report.metrics.rouge_1_p,
                            "r": report.metrics.rouge_1_r
                        },
                        "rouge-2": {
                            "f": report.metrics.rouge_2_f,
                            "p": report.metrics.rouge_2_p,
                            "r": report.metrics.rouge_2_r
                        },
                        "rouge-l": {
                            "f": report.metrics.rouge_l_f,
                            "p": report.metrics.rouge_l_p,
                            "r": report.metrics.rouge_l_r
                        }
                    },
                    "semantic_similarity": report.metrics.semantic_similarity,
                    "coherence_score": report.metrics.coherence_score,
                    "readability_score": report.metrics.readability_score,
                    "compression_ratio": report.metrics.compression_ratio,
                    "factual_consistency": report.metrics.factual_consistency
                },
                "recommendations": report.recommendations,
                "model_used": report.metrics.model_used
            },
            message="Évaluation terminée",
            processing_time=processing_time
        )
        
    except Exception as e:
        logger.error(f"Erreur évaluation: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Erreur lors de l'évaluation: {str(e)}"
        )

@app.get("/api/v1/metrics", response_model=APIResponse)
async def get_metrics(
    hours: int = 24,
    metric_type: str = "system",
    api_key: Optional[str] = Depends(get_api_key)
):
    """Récupère les métriques du système"""
    try:
        if not metrics_collector:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Service de métriques non disponible"
            )
        
        current_stats = metrics_collector.get_current_stats()
        historical_data = metrics_collector.get_historical_data(hours, metric_type)
        
        return APIResponse(
            success=True,
            data={
                "current": current_stats,
                "historical": historical_data,
                "period_hours": hours,
                "metric_type": metric_type
            },
            message="Métriques récupérées avec succès"
        )
        
    except Exception as e:
        logger.error(f"Erreur métriques: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Erreur lors de la récupération des métriques: {str(e)}"
        )

@app.get("/api/v1/models", response_model=APIResponse)
async def get_models_info(api_key: Optional[str] = Depends(get_api_key)):
    """Informations sur les modèles disponibles"""
    try:
        if not model_manager:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Gestionnaire de modèles non disponible"
            )
        
        from models.model_manager import ModelType
        
        models_info = {}
        for model_type in ModelType:
            available, message = model_manager.is_model_available(model_type)
            models_info[model_type.value] = {
                "available": available,
                "status": message,
                "recommended_for": model_manager._get_model_recommendations(model_type)
            }
        
        return APIResponse(
            success=True,
            data={
                "models": models_info,
                "recommendation_engine": True,
                "default_model": "auto"
            },
            message="Informations sur les modèles récupérées"
        )
        
    except Exception as e:
        logger.error(f"Erreur info modèles: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Erreur lors de la récupération des informations: {str(e)}"
        )

# Point d'entrée pour le développement
if __name__ == "__main__":
    uvicorn.run(
        "summarization:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )