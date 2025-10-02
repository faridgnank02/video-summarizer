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

from .led_model import LEDSummarizer
from .openai_model import OpenAISummarizer

logger = logging.getLogger(__name__)


class ModelType(Enum):
    """Types de modèles disponibles"""
    LED = "led"
    OPENAI = "openai"


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
        self._led_model = None
        self._openai_model = None
        
        # Statistiques globales
        self.stats = {
            'total_requests': 0,
            'led_requests': 0,
            'openai_requests': 0,
            'total_processing_time': 0.0,
            'average_processing_time': 0.0
        }
        
        logger.info("ModelManager initialisé")
    
    def _load_config(self) -> Dict[str, Any]:
        """Charge la configuration"""
        default_config = {
            'default_model': 'led',
            'auto_fallback': True,
            'models': {
                'led': {
                    'model_name': 'allenai/led-base-16384',
                    'device': 'auto'
                },
                'openai': {
                    'model_name': 'gpt-4',
                    'fallback_model': 'gpt-3.5-turbo'
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
    def led_model(self) -> Optional[LEDSummarizer]:
        """Accès lazy au modèle LED"""
        if self._led_model is None:
            logger.info("Chargement du modèle LED...")
            start_time = time.time()
            
            try:
                led_config = self.config['models']['led']
                self._led_model = LEDSummarizer(
                    model_name=led_config['model_name'],
                    config_path=self.config_path,
                    device=led_config.get('device', 'auto')
                )
                
                load_time = time.time() - start_time
                logger.info(f"Modèle LED chargé en {load_time:.2f}s")
                
            except Exception as e:
                logger.error(f"Impossible de charger le modèle LED: {e}")
                logger.info("Le modèle LED ne sera pas disponible")
                # Ne pas réessayer - marquer comme échec définitif
                self._led_model = False
                return None
        
        # Si le chargement a échoué précédemment
        if self._led_model is False:
            return None
            
        return self._led_model
    
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
                # Vérifier si le modèle LED peut être chargé
                led_model = self.led_model
                if led_model is None:
                    return False, "Modèle LED indisponible (erreur de chargement)"
                return True, ""
            
            elif model_type == ModelType.OPENAI:
                # Vérifier si la clé API OpenAI est configurée
                api_key = os.getenv('OPENAI_API_KEY')
                if not api_key:
                    return False, "Clé API OpenAI manquante (OPENAI_API_KEY)"
                
                _ = self.openai_model
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
        
        # Vérifier la disponibilité des modèles
        led_available, _ = self.is_model_available(ModelType.LED)
        openai_available, _ = self.is_model_available(ModelType.OPENAI)
        
        if priority == "speed":
            if openai_available:
                return ModelType.OPENAI
            elif led_available:
                return ModelType.LED
        
        elif priority == "quality":
            if led_available:
                return ModelType.LED
            elif openai_available:
                return ModelType.OPENAI
        
        else:  # balanced
            # Pour les textes courts, préférer OpenAI (plus rapide)
            if text_length < 1000 and openai_available:
                return ModelType.OPENAI
            # Pour les textes longs, préférer LED (spécialisé pour les longs textes)
            elif text_length >= 1000 and led_available:
                return ModelType.LED
            # Fallback
            elif openai_available:
                return ModelType.OPENAI
            elif led_available:
                return ModelType.LED
        
        # Par défaut, retourner LED si disponible
        return ModelType.LED if led_available else ModelType.OPENAI
    
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
            fallback_model = (ModelType.OPENAI if request.model_type == ModelType.LED 
                            else ModelType.LED)
            
            fallback_available, _ = self.is_model_available(fallback_model)
            if fallback_available:
                logger.warning(f"Modèle {request.model_type.value} indisponible, "
                             f"fallback vers {fallback_model.value}")
                request.model_type = fallback_model
            else:
                raise RuntimeError(f"Aucun modèle disponible. Erreur LED: {error_msg}")
        
        elif request.model_type == ModelType.LED:
            # Double vérification pour LED : essayer d'accéder au modèle
            led_model = self.led_model
            if led_model is None:
                logger.warning("Modèle LED indisponible, fallback vers OpenAI")
                openai_available, _ = self.is_model_available(ModelType.OPENAI)
                if openai_available:
                    request.model_type = ModelType.OPENAI
                else:
                    raise RuntimeError("Aucun modèle disponible")
        
        # Ajuster les longueurs selon la configuration
        length_config = self.config['summary_lengths'][request.summary_length.value]
        max_length = request.max_length or length_config['max_length']
        min_length = request.min_length or length_config['min_length']
        
        # Générer le résumé
        try:
            if request.model_type == ModelType.LED:
                summary = self.led_model.summarize(
                    request.text,
                    max_length=max_length,
                    min_length=min_length,
                    summary_type=request.summary_length.value
                )
                model_used = f"LED ({self.led_model.model_name})"
                self.stats['led_requests'] += 1
                
            else:  # OpenAI
                summary = self.openai_model.summarize(
                    request.text,
                    summary_type=request.summary_length.value,
                    language=request.language
                )
                model_used = f"OpenAI ({self.openai_model.model_name})"
                self.stats['openai_requests'] += 1
            
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
            model_type: Type de modèle ('led', 'openai', 'auto')
            summary_length: Longueur ('short', 'long')
            language: Langue (optionnel)
            
        Returns:
            str: Résumé généré
        """
        # Déterminer le modèle
        if model_type == "auto":
            model_enum = self.recommend_model(text)
        else:
            model_enum = ModelType.LED if model_type == "led" else ModelType.OPENAI
        
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
    
    def get_stats(self) -> Dict[str, Any]:
        """Retourne les statistiques globales"""
        stats = self.stats.copy()
        
        # Ajouter les stats des modèles individuels
        if self._openai_model:
            stats['openai_usage'] = self._openai_model.get_usage_stats()
        
        if self._led_model:
            stats['led_info'] = self._led_model.get_model_info()
        
        return stats
    
    def reset_stats(self):
        """Remet à zéro les statistiques"""
        self.stats = {
            'total_requests': 0,
            'led_requests': 0,
            'openai_requests': 0,
            'total_processing_time': 0.0,
            'average_processing_time': 0.0
        }
        
        if self._openai_model:
            self._openai_model.reset_usage_stats()


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