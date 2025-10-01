"""
Modèle OpenAI GPT pour le résumé rapide
Alternative rapide au modèle LED avec l'API OpenAI
"""

import os
import logging
import time
from typing import List, Dict, Any, Optional
import yaml
from pathlib import Path
from dotenv import load_dotenv

try:
    import openai
    from openai import OpenAI
except ImportError:
    openai = None
    OpenAI = None

# Charger les variables d'environnement depuis .env
load_dotenv()

logger = logging.getLogger(__name__)


class OpenAISummarizer:
    """Modèle OpenAI GPT pour le résumé rapide"""

    def __init__(self, 
                 api_key: Optional[str] = None,
                 model_name: str = "gpt-4",
                 config_path: Optional[str] = None):
        """
        Initialise le résumeur OpenAI
        
        Args:
            api_key: Clé API OpenAI (si None, utilise la variable d'environnement)
            model_name: Nom du modèle ('gpt-4', 'gpt-3.5-turbo')
            config_path: Chemin vers le fichier de configuration
        """
        if OpenAI is None:
            raise ImportError("La bibliothèque OpenAI n'est pas installée. Installez-la avec: pip install openai")

        # Configuration de l'API
        self.api_key = api_key or os.getenv('OPENAI_API_KEY')
        if not self.api_key:
            raise ValueError("Clé API OpenAI manquante. Assurez-vous que OPENAI_API_KEY est configurée dans .env.")

        self.client = OpenAI(api_key=self.api_key)
        self.model_name = model_name

        # Charger la configuration
        self.config = self._load_config(config_path)

        # Statistiques d'utilisation
        self.usage_stats = {
            'total_requests': 0,
            'total_tokens': 0,
            'total_cost': 0.0,
            'errors': 0
        }

        logger.info(f"OpenAI Summarizer initialisé avec le modèle: {model_name}")
    
    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """Charge la configuration depuis un fichier YAML"""
        default_config = {
            'model_name': 'gpt-4',
            'fallback_model': 'gpt-3.5-turbo',
            'max_tokens': 500,
            'temperature': 0.3,
            'prompts': {
                'short_french': """Résume ce texte en français en 2-3 phrases maximum, en gardant les points les plus importants:

{text}

Résumé:""",
                'long_french': """Résume ce texte en français en un paragraphe détaillé (5-8 phrases), en préservant les idées principales et les détails importants:

{text}

Résumé détaillé:""",
                'short_english': """Summarize this text in English in 2-3 sentences maximum, keeping the most important points:

{text}

Summary:""",
                'long_english': """Summarize this text in English in a detailed paragraph (5-8 sentences), preserving the main ideas and important details:

{text}

Detailed summary:"""
            }
        }
        
        if config_path and os.path.exists(config_path):
            try:
                with open(config_path, 'r', encoding='utf-8') as f:
                    file_config = yaml.safe_load(f)
                    openai_config = file_config.get('models', {}).get('openai', {})
                    default_config.update(openai_config)
            except Exception as e:
                logger.warning(f"Erreur lors du chargement de la config: {e}")
        
        return default_config
    
    def _detect_language(self, text: str) -> str:
        """Détection simple de la langue"""
        french_words = {'le', 'la', 'les', 'de', 'du', 'des', 'et', 'est', 'une', 'un', 'avec', 'dans'}
        english_words = {'the', 'and', 'is', 'are', 'of', 'to', 'in', 'for', 'with', 'on', 'at'}
        
        words = set(text.lower().split()[:100])  # Analyser les 100 premiers mots
        
        french_score = len(words.intersection(french_words))
        english_score = len(words.intersection(english_words))
        
        return 'french' if french_score > english_score else 'english'
    
    def _get_prompt(self, text: str, summary_type: str = "long", language: str = None) -> str:
        """Génère le prompt approprié selon le type et la langue"""
        if language is None:
            language = self._detect_language(text)
        
        prompt_key = f"{summary_type}_{language}"
        prompt_template = self.config['prompts'].get(prompt_key)
        
        if not prompt_template:
            # Fallback vers l'anglais si pas trouvé
            prompt_key = f"{summary_type}_english"
            prompt_template = self.config['prompts'].get(prompt_key)
        
        return prompt_template.format(text=text)
    
    def _calculate_cost(self, model: str, input_tokens: int, output_tokens: int) -> float:
        """Calcule le coût approximatif de la requête"""
        # Tarifs approximatifs (à jour en 2024)
        pricing = {
            'gpt-4': {'input': 0.03, 'output': 0.06},  # pour 1K tokens
            'gpt-3.5-turbo': {'input': 0.001, 'output': 0.002}
        }
        
        model_pricing = pricing.get(model, pricing['gpt-3.5-turbo'])
        
        input_cost = (input_tokens / 1000) * model_pricing['input']
        output_cost = (output_tokens / 1000) * model_pricing['output']
        
        return input_cost + output_cost
    
    def summarize(self, 
                  text: str, 
                  summary_type: str = "long",
                  language: str = None,
                  max_retries: int = 3) -> str:
        """
        Génère un résumé avec l'API OpenAI
        
        Args:
            text: Texte à résumer
            summary_type: Type de résumé ('short' ou 'long')
            language: Langue ('french' ou 'english', auto-détectée si None)
            max_retries: Nombre de tentatives max en cas d'erreur
            
        Returns:
            str: Résumé généré
        """
        if not text or not text.strip():
            return ""
        
        # Tronquer le texte si trop long (éviter les erreurs de tokens)
        max_chars = 12000  # Approximativement 3000 tokens
        if len(text) > max_chars:
            text = text[:max_chars] + "..."
            logger.warning("Texte tronqué pour respecter les limites de l'API")
        
        prompt = self._get_prompt(text, summary_type, language)
        
        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=self.config['max_tokens'],
                    temperature=self.config['temperature']
                )
                
                summary = response.choices[0].message.content.strip()
                
                # Mettre à jour les statistiques
                self.usage_stats['total_requests'] += 1
                if hasattr(response, 'usage'):
                    input_tokens = response.usage.prompt_tokens
                    output_tokens = response.usage.completion_tokens
                    self.usage_stats['total_tokens'] += input_tokens + output_tokens
                    
                    cost = self._calculate_cost(self.model_name, input_tokens, output_tokens)
                    self.usage_stats['total_cost'] += cost
                
                logger.info(f"Résumé généré avec succès ({summary_type}, {self.model_name})")
                return summary
                
            except Exception as e:
                logger.warning(f"Tentative {attempt + 1} échouée: {e}")
                
                if attempt == max_retries - 1:
                    # Dernière tentative, essayer le modèle de fallback
                    if self.model_name != self.config['fallback_model']:
                        logger.info(f"Essai avec le modèle de fallback: {self.config['fallback_model']}")
                        try:
                            response = self.client.chat.completions.create(
                                model=self.config['fallback_model'],
                                messages=[{"role": "user", "content": prompt}],
                                max_tokens=self.config['max_tokens'],
                                temperature=self.config['temperature']
                            )
                            
                            summary = response.choices[0].message.content.strip()
                            self.usage_stats['total_requests'] += 1
                            
                            logger.info(f"Résumé généré avec le modèle de fallback")
                            return summary
                            
                        except Exception as fallback_error:
                            logger.error(f"Erreur avec le modèle de fallback: {fallback_error}")
                    
                    self.usage_stats['errors'] += 1
                    return f"Erreur lors de la génération du résumé: {str(e)}"
                
                time.sleep(2 ** attempt)  # Backoff exponentiel
    
    def batch_summarize(self, 
                       texts: List[str], 
                       summary_type: str = "long",
                       language: str = None,
                       delay: float = 1.0) -> List[str]:
        """
        Génère des résumés pour une liste de textes
        
        Args:
            texts: Liste des textes à résumer
            summary_type: Type de résumé ('short' ou 'long')
            language: Langue cible
            delay: Délai entre les requêtes (pour respecter les rate limits)
            
        Returns:
            List[str]: Liste des résumés générés
        """
        summaries = []
        
        for i, text in enumerate(texts):
            try:
                summary = self.summarize(text, summary_type, language)
                summaries.append(summary)
                
                logger.info(f"Traité {i + 1}/{len(texts)} textes")
                
                # Délai entre les requêtes
                if i < len(texts) - 1:
                    time.sleep(delay)
                    
            except Exception as e:
                logger.error(f"Erreur pour le texte {i + 1}: {e}")
                summaries.append(f"Erreur: {str(e)}")
        
        return summaries
    
    def get_usage_stats(self) -> Dict[str, Any]:
        """Retourne les statistiques d'utilisation"""
        return self.usage_stats.copy()
    
    def reset_usage_stats(self):
        """Remet à zéro les statistiques d'utilisation"""
        self.usage_stats = {
            'total_requests': 0,
            'total_tokens': 0,
            'total_cost': 0.0,
            'errors': 0
        }
    
    def get_model_info(self) -> Dict[str, Any]:
        """Retourne les informations sur le modèle"""
        return {
            'model_name': self.model_name,
            'fallback_model': self.config['fallback_model'],
            'api_configured': bool(self.api_key),
            'config': self.config,
            'usage_stats': self.usage_stats
        }


# Fonction de compatibilité
def create_openai_summarizer(api_key: Optional[str] = None, 
                           model_name: str = "gpt-4") -> OpenAISummarizer:
    """
    Crée une instance d'OpenAISummarizer
    
    Args:
        api_key: Clé API OpenAI
        model_name: Nom du modèle à utiliser
        
    Returns:
        OpenAISummarizer: Instance configurée
    """
    return OpenAISummarizer(api_key=api_key, model_name=model_name)


if __name__ == "__main__":
    # Test du modèle OpenAI (nécessite une clé API)
    try:
        summarizer = OpenAISummarizer()
        
        test_text = """
        L'intelligence artificielle représente l'une des innovations technologiques les plus 
        significatives de notre époque. Cette technologie permet aux ordinateurs et aux systèmes 
        automatisés d'effectuer des tâches qui nécessitaient auparavant l'intelligence humaine, 
        comme la reconnaissance de formes, la prise de décision, et l'apprentissage à partir de données. 
        Les domaines d'application sont vastes : de la médecine à la finance, en passant par les 
        transports autonomes et les assistants virtuels. Cependant, cette révolution technologique 
        soulève également des préoccupations importantes concernant l'éthique, l'emploi, et la 
        protection de la vie privée des utilisateurs.
        """
        
        # Test résumé court
        short_summary = summarizer.summarize(test_text, summary_type="short")
        print(f"Résumé court: {short_summary}")
        
        # Statistiques
        stats = summarizer.get_usage_stats()
        print(f"Statistiques: {stats}")
        
    except Exception as e:
        print(f"Test impossible (clé API manquante ?): {e}")
        print("Configurez OPENAI_API_KEY pour tester ce module")