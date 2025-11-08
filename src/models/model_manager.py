"""
Gestionnaire de modèles - Orchestration entre LED et OpenAI
Permet de choisir entre qualité (LED) et rapidité (OpenAI)
"""

import os
import logging
import time
from typing import Dict, Any, Optional, List, Tuple
from enum import Enum
from dataclasses import dataclass
import yaml

from .openai_model import OpenAISummarizer
from .ollama_model import OllamaSummarizer, OllamaConnectionError

logger = logging.getLogger(__name__)


class ModelType(Enum):
    """Types de modèles disponibles"""
    LED = "led"
    OPENAI = "openai"
    OLLAMA = "ollama"


class SummaryLength(Enum):
    """Longueurs de résumé disponibles"""
    SHORT = "short"
    LONG = "long"


@dataclass
class SummaryRequest:
    """Requête de résumé"""
    text: str
    model_type: ModelType
    summary_length: SummaryLength
    language: Optional[str] = None
    max_length: Optional[int] = None
    min_length: Optional[int] = None


@dataclass
class SummaryResponse:
    """Réponse de résumé"""
    summary: str
    model_used: str
    processing_time: float
    word_count: int
    confidence_score: Optional[float] = None
    metadata: Optional[Dict[str, Any]] = None


class ModelManager:
    """Gestionnaire principal des modèles de résumé"""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialise le gestionnaire de modèles
        
        Args:
            config_path: Chemin vers le fichier de configuration
        """
        self.config_path = config_path
        self.config = self._load_config()
        
        # Modèles (chargés à la demande)
        # LED support is disabled in this build to reduce memory footprint
        self._led_model = None
        self._openai_model = None
        self._ollama_model = None
        
        # Statistiques globales
        self.stats = {
            'total_requests': 0,
            'led_requests': 0,
            'openai_requests': 0,
            'ollama_requests': 0,
            'total_processing_time': 0.0,
            'average_processing_time': 0.0
        }
        
        logger.info("ModelManager initialisé")
    
    def _load_config(self) -> Dict[str, Any]:
        """Charge la configuration"""
        default_config = {
            'default_model': 'ollama',
            'auto_fallback': True,
            'models': {
                'openai': {
                    'model_name': 'gpt-4o-mini',
                    'fallback_model': 'gpt-3.5-turbo'
                },
                'ollama': {
                    'model_name': 'gemma3:1b',
                    'base_url': 'http://localhost:11434'
                }
            },
            'summary_lengths': {
                'short': {'min_length': 50, 'max_length': 200},
                'long': {'min_length': 200, 'max_length': 500}
            }
        }
        
        if self.config_path and os.path.exists(self.config_path):
            try:
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    file_config = yaml.safe_load(f)
                    default_config.update(file_config)
            except Exception as e:
                logger.warning(f"Erreur lors du chargement de la config: {e}")
        
        return default_config
    
    @property
    def led_model(self) -> Optional[Any]:
        """Accès lazy au modèle LED

        NOTE: LED support is intentionally disabled in this runtime to avoid
        heavy memory/GPU usage. This property always returns None.
        """
        logger.debug("LED model support is disabled in this build; led_model returns None")
        return None
    
    @property
    def openai_model(self) -> OpenAISummarizer:
        """Accès lazy au modèle OpenAI"""
        if self._openai_model is None:
            logger.info("Initialisation du modèle OpenAI...")
            
            openai_config = self.config['models']['openai']
            self._openai_model = OpenAISummarizer(
                model_name=openai_config['model_name'],
                config_path=self.config_path
            )
            
            logger.info("Modèle OpenAI initialisé")
        
        return self._openai_model
    
    @property
    def ollama_model(self) -> Optional[OllamaSummarizer]:
        """Accès lazy au modèle Ollama"""
        if self._ollama_model is None:
            logger.info("Initialisation du modèle Ollama...")
            
            try:
                ollama_config = self.config['models'].get('ollama', {})
                self._ollama_model = OllamaSummarizer(
                    base_url=ollama_config.get('base_url', 'http://localhost:11434'),
                    model_name=ollama_config.get('model_name', 'gemma3:1b'),
                    config_path=self.config_path
                )
                logger.info("Modèle Ollama initialisé")
            except OllamaConnectionError as e:
                logger.error(f"Ollama non disponible: {e}")
                self._ollama_model = False
                return None
            except Exception as e:
                logger.error(f"Impossible de charger Ollama: {e}")
                self._ollama_model = False
                return None
        
        # Si le chargement a échoué précédemment
        if self._ollama_model is False:
            return None
            
        return self._ollama_model
    
    def is_model_available(self, model_type: ModelType) -> Tuple[bool, str]:
        """
        Vérifie si un modèle est disponible
        
        Args:
            model_type: Type de modèle à vérifier
            
        Returns:
            Tuple[bool, str]: (Disponible, Message d'erreur si applicable)
        """
        try:
            if model_type == ModelType.LED:
                # LED disabled in this distribution/version
                return False, "LED support disabled in this build"
            
            elif model_type == ModelType.OPENAI:
                # Vérifier si la clé API OpenAI est configurée
                api_key = os.getenv('OPENAI_API_KEY')
                if not api_key:
                    return False, "Clé API OpenAI manquante (OPENAI_API_KEY)"
                
                _ = self.openai_model
                return True, ""
            
            elif model_type == ModelType.OLLAMA:
                # Vérifier si Ollama est disponible
                ollama_model = self.ollama_model
                if ollama_model is None:
                    return False, "Ollama non disponible (serveur non démarré ou modèle manquant)"
                return True, ""
            
        except Exception as e:
            return False, str(e)
        
        return False, "Type de modèle inconnu"
    
    def recommend_model(self, 
                       text: str, 
                       priority: str = "balanced") -> ModelType:
        """
        Recommande un modèle basé sur le texte et les priorités
        
        Args:
            text: Texte à analyser
            priority: Priorité ('speed', 'quality', 'balanced')
            
        Returns:
            ModelType: Modèle recommandé
        """
        text_length = len(text.split())
        
        # Vérifier la disponibilité des modèles (LED intentionally disabled)
        led_available = False
        openai_available, _ = self.is_model_available(ModelType.OPENAI)
        ollama_available, _ = self.is_model_available(ModelType.OLLAMA)
        
        if priority == "speed":
            # Priorité à la vitesse : OpenAI > Ollama
            if openai_available:
                return ModelType.OPENAI
            elif ollama_available:
                return ModelType.OLLAMA
        
        elif priority == "quality":
            # Priorité à la qualité : OpenAI > Ollama
            if openai_available:
                return ModelType.OPENAI
            elif ollama_available:
                return ModelType.OLLAMA
        
        elif priority == "cost":
            # Priorité au coût (gratuit) : Ollama > OpenAI
            if ollama_available:
                return ModelType.OLLAMA
            elif openai_available:
                return ModelType.OPENAI
        
        else:  # balanced
            # Pour les textes courts, préférer OpenAI (plus rapide)
            if text_length < 500 and openai_available:
                return ModelType.OPENAI
            # Pour les textes moyens, préférer Ollama (bon compromis)
            elif text_length < 2000 and ollama_available:
                return ModelType.OLLAMA
            # Pour les textes longs, préférer Ollama (LED removed)
            elif text_length >= 2000 and ollama_available:
                return ModelType.OLLAMA
            # Fallback
            elif ollama_available:
                return ModelType.OLLAMA
            elif openai_available:
                return ModelType.OPENAI
            # no LED fallback
        
        # Par défaut, retourner le premier disponible
        if ollama_available:
            return ModelType.OLLAMA
        elif openai_available:
            return ModelType.OPENAI

        # Aucun modèle disponible
        raise RuntimeError("Aucun modèle disponible (Ollama et OpenAI indisponibles)")
    
    def summarize(self, request: SummaryRequest) -> SummaryResponse:
        """
        Génère un résumé selon la requête
        
        Args:
            request: Requête de résumé
            
        Returns:
            SummaryResponse: Réponse avec le résumé
        """
        start_time = time.time()
        
        # Vérifier la disponibilité du modèle demandé
        model_available, error_msg = self.is_model_available(request.model_type)
        
        if not model_available:
            # Toujours essayer le fallback automatique
            # Ordre de fallback : Ollama → OpenAI
            fallback_models = [ModelType.OLLAMA, ModelType.OPENAI]
            fallback_models.remove(request.model_type)  # Retirer le modèle déjà essayé
            
            for fallback_model in fallback_models:
                fallback_available, _ = self.is_model_available(fallback_model)
                if fallback_available:
                    logger.warning(f"Modèle {request.model_type.value} indisponible, "
                                 f"fallback vers {fallback_model.value}")
                    request.model_type = fallback_model
                    break
            else:
                # Aucun modèle disponible
                raise RuntimeError(f"Aucun modèle disponible. Erreur: {error_msg}")
        
        # LED support removed: ensure request.model_type is not LED
        if request.model_type == ModelType.LED:
            logger.warning("Requested LED model but LED support is disabled; switching to recommendation")
            request.model_type = self.recommend_model(request.text)
        
        # Ajuster les longueurs selon la configuration
        length_config = self.config['summary_lengths'][request.summary_length.value]
        max_length = request.max_length or length_config['max_length']
        min_length = request.min_length or length_config['min_length']
        
        # Générer le résumé
        try:
            if request.model_type == ModelType.OPENAI:
                summary = self.openai_model.summarize(
                    request.text,
                    summary_type=request.summary_length.value,
                    language=request.language
                )
                model_used = f"OpenAI ({self.openai_model.model_name})"
                self.stats['openai_requests'] += 1
                
            elif request.model_type == ModelType.OLLAMA:
                summary = self.ollama_model.summarize(
                    request.text,
                    summary_type=request.summary_length.value,
                    language=request.language
                )
                model_used = f"Ollama ({self.ollama_model.model_name})"
                self.stats['ollama_requests'] += 1
            
            else:
                raise ValueError(f"Type de modèle non supporté: {request.model_type}")
            
            processing_time = time.time() - start_time
            
            # Mettre à jour les statistiques
            self.stats['total_requests'] += 1
            self.stats['total_processing_time'] += processing_time
            self.stats['average_processing_time'] = (
                self.stats['total_processing_time'] / self.stats['total_requests']
            )
            
            # Calculer le nombre de mots
            word_count = len(summary.split())
            
            return SummaryResponse(
                summary=summary,
                model_used=model_used,
                processing_time=processing_time,
                word_count=word_count,
                metadata={
                    'input_word_count': len(request.text.split()),
                    'compression_ratio': word_count / len(request.text.split()) if request.text else 0,
                    'requested_length': request.summary_length.value,
                    'model_type': request.model_type.value
                }
            )
            
        except Exception as e:
            logger.error(f"Erreur lors de la génération du résumé: {e}")
            raise
    
    def summarize_simple(self, 
                        text: str,
                        model_type: str = "auto",
                        summary_length: str = "long",
                        language: str = None) -> str:
        """
        Interface simplifiée pour le résumé
        
        Args:
            text: Texte à résumer
            model_type: Type de modèle ('led', 'openai', 'ollama', 'auto')
            summary_length: Longueur ('short', 'long')
            language: Langue (optionnel)
            
        Returns:
            str: Résumé généré
        """
        # Déterminer le modèle
        if model_type == "auto":
            model_enum = self.recommend_model(text)
        elif model_type == "led":
            # Legacy option: map to auto recommendation since LED is disabled
            model_enum = self.recommend_model(text)
        elif model_type == "openai":
            model_enum = ModelType.OPENAI
        elif model_type == "ollama":
            model_enum = ModelType.OLLAMA
        else:
            # Fallback vers auto
            model_enum = self.recommend_model(text)
        
        # Créer la requête
        request = SummaryRequest(
            text=text,
            model_type=model_enum,
            summary_length=SummaryLength.SHORT if summary_length == "short" else SummaryLength.LONG,
            language=language
        )
        
        # Générer le résumé
        response = self.summarize(request)
        return response.summary
    
    def batch_summarize(self, 
                       texts: List[str],
                       model_type: str = "auto",
                       summary_length: str = "long") -> List[SummaryResponse]:
        """
        Résumé en lot
        
        Args:
            texts: Liste des textes à résumer
            model_type: Type de modèle
            summary_length: Longueur des résumés
            
        Returns:
            List[SummaryResponse]: Liste des réponses
        """
        responses = []
        
        for i, text in enumerate(texts):
            try:
                response = self.summarize_simple(text, model_type, summary_length)
                responses.append(response)
                logger.info(f"Traité {i + 1}/{len(texts)} textes")
            except Exception as e:
                logger.error(f"Erreur pour le texte {i + 1}: {e}")
                error_response = SummaryResponse(
                    summary=f"Erreur: {str(e)}",
                    model_used="error",
                    processing_time=0.0,
                    word_count=0
                )
                responses.append(error_response)
        
        return responses
    
    def _get_model_recommendations(self, model_type: ModelType) -> Dict[str, Any]:
        """Get recommendations for a specific model"""
        recommendations = {
            ModelType.LED: {
                "best_for": ["Long documents (>2000 words)", "English content", "Offline use"],
                "pros": ["Free", "Offline", "Specialized for long texts", "GPU accelerated"],
                "cons": ["Slower", "Requires 8GB+ RAM", "Best for English"],
                "speed": "Slow (30-200s)",
                "quality": "★★★★☆",
                "cost": "Free"
            },
            ModelType.OPENAI: {
                "best_for": ["Quick summaries", "Multi-language", "High quality"],
                "pros": ["Very fast", "Excellent quality", "Multi-language", "Reliable"],
                "cons": ["Costs money", "Requires internet", "API limits"],
                "speed": "Fast (2-5s)",
                "quality": "★★★★★",
                "cost": "$$$ (paid API)"
            },
            ModelType.OLLAMA: {
                "best_for": ["Local deployment", "Cost-free", "Medium texts"],
                "pros": ["Free", "Local", "Fast CPU inference", "Multiple models"],
                "cons": ["Requires Ollama server", "Moderate quality", "Setup needed"],
                "speed": "Medium (10-30s)",
                "quality": "★★★★☆",
                "cost": "Free (local compute)"
            }
        }
        return recommendations.get(model_type, {})
    
    def get_stats(self) -> Dict[str, Any]:
        """Retourne les statistiques globales"""
        stats = self.stats.copy()
        
        # Ajouter les stats des modèles individuels
        if self._openai_model:
            stats['openai_usage'] = self._openai_model.get_usage_stats()
        
        if self._ollama_model and self._ollama_model is not False:
            stats['ollama_usage'] = self._ollama_model.get_usage_stats()
        
        if self._led_model:
            stats['led_info'] = self._led_model.get_model_info()
        
        return stats
    
    def reset_stats(self):
        """Remet à zéro les statistiques"""
        self.stats = {
            'total_requests': 0,
            'led_requests': 0,
            'openai_requests': 0,
            'ollama_requests': 0,
            'total_processing_time': 0.0,
            'average_processing_time': 0.0
        }
        
        if self._openai_model:
            self._openai_model.reset_usage_stats()
        
        if self._ollama_model and self._ollama_model is not False:
            self._ollama_model.reset_usage_stats()


# Fonction utilitaire globale
def create_model_manager(config_path: Optional[str] = None) -> ModelManager:
    """
    Crée une instance de ModelManager
    
    Args:
        config_path: Chemin vers la configuration
        
    Returns:
        ModelManager: Instance configurée
    """
    return ModelManager(config_path)


if __name__ == "__main__":
    # Test du gestionnaire de modèles
    manager = ModelManager()
    
    test_text = """
    L'intelligence artificielle transforme notre société de manière profonde. 
    Cette technologie révolutionnaire permet aux machines d'apprendre et de prendre 
    des décisions autonomes, ouvrant de nouvelles possibilités dans tous les secteurs.
    Cependant, elle soulève aussi des questions éthiques importantes qu'il faut adresser.
    """
    
    # Test de recommandation
    recommended = manager.recommend_model(test_text, "balanced")
    print(f"Modèle recommandé: {recommended}")
    
    # Test de résumé simple
    try:
        summary = manager.summarize_simple(test_text, "auto", "short")
        print(f"Résumé: {summary}")
        
        # Statistiques
        stats = manager.get_stats()
        print(f"Statistiques: {stats}")
        
    except Exception as e:
        print(f"Erreur lors du test: {e}")