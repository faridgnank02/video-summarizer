"""
Ollama model integration for local LLM summarization
Supports Mistral, Llama, Gemma, Qwen and other models via Ollama API
"""

import os
import logging
import time
import requests
from typing import List, Dict, Any, Optional
from pathlib import Path
import yaml

logger = logging.getLogger(__name__)


class OllamaConnectionError(Exception):
    """Exception raised when Ollama server is not available"""
    pass


class OllamaSummarizer:
    """
    Ollama-based summarizer using local LLMs
    
    Supports multiple models:
    - Mistral 7B: General purpose, fast
    - Llama 3.1 8B: High quality, multilingual
    - Gemma 2 9B: Advanced comprehension
    - Qwen 2.5 7B: Optimized for long summaries
    """
    
    def __init__(self,
                 base_url: str = "http://localhost:11434",
                 model_name: Optional[str] = None,
                 config_path: Optional[str] = None):
        """
        Initialize Ollama summarizer
        
        Args:
            base_url: Ollama API base URL
            model_name: Model to use (e.g., 'gemma3:1b', 'llama3.1:8b'). If None, loads from config.
            config_path: Path to config file
        """
        self.base_url = base_url.rstrip('/')
        self.config = self._load_config(config_path)
        
        # Use config model_name if not provided
        self.model_name = model_name or self.config.get('model_name', 'gemma3:1b')
        
        # Track usage statistics
        self.usage_stats = {
            'total_requests': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'total_tokens_generated': 0,
            'total_processing_time': 0.0
        }
        
        # Check Ollama availability
        if not self._check_ollama_available():
            raise OllamaConnectionError(
                f"Ollama server not available at {base_url}. "
                f"Make sure Ollama is running: 'ollama serve'"
            )
        
        # Check if model is available
        available_models = self._get_available_models()
        if model_name not in available_models:
            logger.warning(f"Model {model_name} not found locally. Available: {available_models}")
            logger.info(f"Download with: ollama pull {model_name}")
        
        logger.info(f"Ollama Summarizer initialized with {model_name} at {base_url}")
    
    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        default_config = {
            'temperature': 0.3,
            'max_tokens': 800,  # Increased for long summaries
            'top_p': 0.9,
            'top_k': 40,
            'repeat_penalty': 1.1,
            'timeout': 120,
            'prompts': {
                'short_french': """Tu dois produire un résumé court en français. Commence directement par le contenu, sans dire "Voici", "Résumé" ou toute autre introduction.

Contraintes strictes:
- 2 à 3 phrases maximum
- Commence directement par le sujet principal
- Pas de formules comme "Ce texte parle de", "Voici un résumé", etc.

Texte à résumer:
{text}

Résumé (2-3 phrases):""",
                'long_french': """Tu dois produire un résumé détaillé en français. Commence directement par le contenu, sans dire "Voici", "Résumé" ou toute autre introduction.

Contraintes strictes:
- 5 à 8 phrases complètes
- Commence directement par le sujet principal
- Couvre tous les points importants du texte
- Pas de formules comme "Ce texte parle de", "Voici un résumé", etc.

Texte à résumer:
{text}

Résumé détaillé (5-8 phrases):""",
                'short_english': """You must produce a brief summary in English. Start directly with the content, without saying "Here's", "Summary" or any other introduction.

Strict requirements:
- 2 to 3 sentences maximum
- Start directly with the main subject
- No phrases like "This text is about", "Here's a summary", etc.

Text to summarize:
{text}

Summary (2-3 sentences):""",
                'long_english': """You must produce a detailed summary in English. Start directly with the content, without saying "Here's", "Summary" or any other introduction.

Strict requirements:
- 5 to 8 complete sentences
- Start directly with the main subject
- Cover all important points from the text
- No phrases like "This text is about", "Here's a summary", etc.

Text to summarize:
{text}

Detailed summary (5-8 sentences):"""
            }
        }
        
        if config_path and os.path.exists(config_path):
            try:
                with open(config_path, 'r', encoding='utf-8') as f:
                    file_config = yaml.safe_load(f)
                    ollama_config = file_config.get('models', {}).get('ollama', {})
                    default_config.update(ollama_config)
            except Exception as e:
                logger.warning(f"Error loading config from {config_path}: {e}")
        
        return default_config
    
    def _check_ollama_available(self) -> bool:
        """Check if Ollama server is running and accessible"""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            return response.status_code == 200
        except requests.exceptions.RequestException as e:
            logger.error(f"Ollama connection error: {e}")
            return False
    
    def _get_available_models(self) -> List[str]:
        """Get list of locally available models"""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            if response.status_code == 200:
                models = response.json().get('models', [])
                return [model['name'] for model in models]
            return []
        except Exception as e:
            logger.error(f"Error fetching available models: {e}")
            return []
    
    def _detect_language(self, text: str) -> str:
        """Simple language detection based on common words"""
        french_words = {'le', 'la', 'les', 'de', 'du', 'des', 'et', 'est', 'une', 'un', 'avec', 'dans'}
        english_words = {'the', 'and', 'is', 'are', 'of', 'to', 'in', 'for', 'with', 'on', 'at'}
        
        words = set(text.lower().split()[:100])  # Analyze first 100 words
        
        french_score = len(words.intersection(french_words))
        english_score = len(words.intersection(english_words))
        
        return 'french' if french_score > english_score else 'english'
    
    def _build_prompt(self, text: str, summary_type: str = "long", language: str = None) -> str:
        """Build appropriate prompt based on summary type and language"""
        if language is None:
            language = self._detect_language(text)
        
        prompt_key = f"{summary_type}_{language}"
        prompt_template = self.config['prompts'].get(prompt_key)
        
        if not prompt_template:
            # Fallback to English if language-specific prompt not found
            prompt_key = f"{summary_type}_english"
            prompt_template = self.config['prompts'].get(prompt_key)
        
        return prompt_template.format(text=text)
    
    def summarize(self,
                  text: str,
                  summary_type: str = "long",
                  language: str = None,
                  max_retries: int = 2) -> str:
        """
        Generate summary using Ollama
        
        Args:
            text: Text to summarize
            summary_type: 'short' or 'long'
            language: Target language ('french' or 'english', auto-detected if None)
            max_retries: Number of retries on failure
            
        Returns:
            Generated summary
        """
        if not text or not text.strip():
            return ""
        
        # Truncate very long texts to avoid timeouts
        max_chars = 12000  # ~3000 tokens
        if len(text) > max_chars:
            text = text[:max_chars] + "..."
            logger.warning(f"Text truncated to {max_chars} characters")
        
        # Build prompt
        prompt = self._build_prompt(text, summary_type, language)
        
        start_time = time.time()
        
        for attempt in range(max_retries):
            try:
                self.usage_stats['total_requests'] += 1
                
                # Adjust max_tokens based on summary type
                num_predict = 300 if summary_type == "short" else 800
                
                # Make request to Ollama
                response = requests.post(
                    f"{self.base_url}/api/generate",
                    json={
                        "model": self.model_name,
                        "prompt": prompt,
                        "stream": False,
                        "keep_alive": "5m",  # Keep model in memory for only 5 minutes
                        "options": {
                            "temperature": self.config.get('temperature', 0.3),
                            "num_predict": num_predict,  # Dynamic based on summary type
                            "top_p": self.config.get('top_p', 0.9),
                            "top_k": self.config.get('top_k', 40),
                            "repeat_penalty": self.config.get('repeat_penalty', 1.1),
                        }
                    },
                    timeout=self.config.get('timeout', 120)
                )
                
                if response.status_code == 200:
                    result = response.json()
                    logger.debug(f"Ollama full response: {str(result)[:500]}")
                    logger.debug(f"Ollama response keys: {list(result.keys())}")
                    
                    # Qwen3 peut mettre la réponse dans 'thinking' au lieu de 'response'
                    summary = result.get('response', '').strip()
                    if not summary and 'thinking' in result:
                        # Pour Qwen3, extraire la partie utile du thinking
                        thinking = result.get('thinking', '')
                        # Chercher après "summary" ou prendre le dernier paragraphe
                        if 'summary' in thinking.lower():
                            parts = thinking.lower().split('summary')
                            if len(parts) > 1:
                                summary = parts[-1].strip(': \n')[:500]
                        if not summary:
                            # Prendre les dernières phrases du thinking
                            sentences = thinking.split('.')
                            summary = '. '.join(sentences[-3:]).strip()
                    
                    # Post-processing: nettoyer les phrases introductives courantes
                    summary = self._clean_summary_response(summary, language)
                    
                    logger.debug(f"Response field length: {len(result.get('response', ''))}")
                    logger.debug(f"Summary after extraction: {repr(summary[:100])}")
                    
                    # Track statistics
                    processing_time = time.time() - start_time
                    self.usage_stats['successful_requests'] += 1
                    self.usage_stats['total_processing_time'] += processing_time
                    
                    # Estimate tokens (rough approximation)
                    estimated_tokens = len(summary.split()) * 1.3
                    self.usage_stats['total_tokens_generated'] += int(estimated_tokens)
                    
                    logger.info(f"Summary generated in {processing_time:.2f}s using {self.model_name}")
                    return summary
                else:
                    error_msg = f"Ollama API error: {response.status_code} - {response.text}"
                    logger.error(error_msg)
                    if attempt < max_retries - 1:
                        wait_time = 2 ** attempt  # Exponential backoff
                        logger.info(f"Retrying in {wait_time}s...")
                        time.sleep(wait_time)
                        continue
                    raise Exception(error_msg)
                    
            except requests.exceptions.Timeout:
                logger.error(f"Timeout after {self.config.get('timeout')}s")
                if attempt < max_retries - 1:
                    logger.info("Retrying with shorter timeout...")
                    self.config['timeout'] = max(60, self.config['timeout'] // 2)
                    continue
                self.usage_stats['failed_requests'] += 1
                raise Exception(f"Request timeout after {max_retries} attempts")
                
            except requests.exceptions.ConnectionError as e:
                logger.error(f"Connection error: {e}")
                self.usage_stats['failed_requests'] += 1
                raise OllamaConnectionError(
                    f"Cannot connect to Ollama at {self.base_url}. "
                    f"Make sure Ollama is running: 'ollama serve'"
                )
                
            except Exception as e:
                logger.error(f"Unexpected error during summarization: {e}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
                    continue
                self.usage_stats['failed_requests'] += 1
                raise
        
        # If all retries failed
        self.usage_stats['failed_requests'] += 1
        return f"Error: Failed to generate summary after {max_retries} attempts"
    
    def _clean_summary_response(self, summary: str, language: str) -> str:
        """
        Remove common introductory phrases from model responses
        
        Args:
            summary: Raw summary text from model
            language: Language of the summary ('fr' or 'en')
            
        Returns:
            Cleaned summary without meta-commentary
        """
        import re
        
        # Patterns to remove common intro phrases (more comprehensive)
        patterns = [
            # English variations
            r"^Here'?s? a (concise |brief |detailed |comprehensive )?summary.*?[.:][\s]*",
            r'^Here is a (concise |brief |detailed |comprehensive )?summary.*?[.:][\s]*',
            r'^This is a (concise |brief |detailed |comprehensive )?summary.*?[.:][\s]*',
            r'^The summary.*?[.:][\s]*',
            r'^Summary\s*[:\-][\s]*',
            r'^Résumé\s*[:\-][\s]*',
            # French variations
            r'^Voici (un |le )?résumé( concis| bref| détaillé| complet)?.*?[.:][\s]*',
            r'^Le résumé( concis| bref| détaillé| complet)?.*?[.:][\s]*',
            r"^C'est (un |le )?résumé.*?[.:][\s]*",
            # Meta phrases about sentences/phrases
            r',\s*focusing on.*?[.:][\s]*',
            r',\s*en \d+(-\d+)?\s+phrases?.*?[.:]',
            r',\s*in \d+(-\d+)?\s+sentences?.*?[.:]',
            # Format markers
            r'^\*\*Summary\*\*\s*[:\-]?[\s]*',
            r'^\*\*Résumé\*\*\s*[:\-]?[\s]*',
            # Common starting patterns to remove
            r'^(This text|The text|Ce texte|Le texte) (discusses?|is about|parle de|traite de).*?[.:][\s]*',
        ]
        
        for pattern in patterns:
            summary = re.sub(pattern, '', summary, flags=re.IGNORECASE)
        
        # Remove leading/trailing whitespace and markdown
        summary = summary.strip()
        summary = summary.strip('*')
        summary = summary.strip()
        
        # Remove multiple spaces
        summary = re.sub(r'\s+', ' ', summary)
        
        return summary
    
    def batch_summarize(self,
                       texts: List[str],
                       summary_type: str = "long",
                       language: str = None,
                       delay: float = 0.5) -> List[str]:
        """
        Generate summaries for multiple texts
        
        Args:
            texts: List of texts to summarize
            summary_type: 'short' or 'long'
            language: Target language
            delay: Delay between requests to avoid overload
            
        Returns:
            List of generated summaries
        """
        summaries = []
        
        for i, text in enumerate(texts):
            try:
                summary = self.summarize(text, summary_type, language)
                summaries.append(summary)
                
                logger.info(f"Processed {i + 1}/{len(texts)} texts")
                
                # Small delay between requests
                if i < len(texts) - 1:
                    time.sleep(delay)
                    
            except Exception as e:
                logger.error(f"Error processing text {i + 1}: {e}")
                summaries.append(f"Error: {str(e)}")
        
        return summaries
    
    def get_model_info(
    base_url: str = "http://localhost:11434",
    model_name: str = "gemma3:1b",
    config_path: Optional[str] = None
) -> Dict[str, Any]:
        """Get information about the current model"""
        try:
            response = requests.post(
                f"{self.base_url}/api/show",
                json={"name": self.model_name},
                timeout=10
            )
            
            if response.status_code == 200:
                model_info = response.json()
                return {
                    'model_name': self.model_name,
                    'base_url': self.base_url,
                    'model_info': model_info,
                    'available_models': self._get_available_models(),
                    'config': self.config,
                    'usage_stats': self.usage_stats
                }
            else:
                return {
                    'model_name': self.model_name,
                    'base_url': self.base_url,
                    'available_models': self._get_available_models(),
                    'config': self.config,
                    'usage_stats': self.usage_stats
                }
        except Exception as e:
            logger.error(f"Error fetching model info: {e}")
            return {
                'model_name': self.model_name,
                'base_url': self.base_url,
                'error': str(e),
                'usage_stats': self.usage_stats
            }
    
    def get_usage_stats(self) -> Dict[str, Any]:
        """Get usage statistics"""
        stats = self.usage_stats.copy()
        
        # Calculate averages
        if stats['successful_requests'] > 0:
            stats['avg_processing_time'] = (
                stats['total_processing_time'] / stats['successful_requests']
            )
            stats['avg_tokens_per_request'] = (
                stats['total_tokens_generated'] / stats['successful_requests']
            )
        else:
            stats['avg_processing_time'] = 0.0
            stats['avg_tokens_per_request'] = 0
        
        # Calculate success rate
        if stats['total_requests'] > 0:
            stats['success_rate'] = (
                stats['successful_requests'] / stats['total_requests']
            )
        else:
            stats['success_rate'] = 0.0
        
        return stats
    
    def reset_usage_stats(self):
        """Reset usage statistics"""
        self.usage_stats = {
            'total_requests': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'total_tokens_generated': 0,
            'total_processing_time': 0.0
        }


def create_ollama_summarizer(
    model_name: str = "mistral:7b",
    base_url: str = "http://localhost:11434"
) -> OllamaSummarizer:
    """
    Factory function to create OllamaSummarizer instance
    
    Args:
        model_name: Model to use
        base_url: Ollama server URL
        
    Returns:
        Configured OllamaSummarizer instance
    """
    return OllamaSummarizer(base_url=base_url, model_name=model_name)


if __name__ == "__main__":
    """Test Ollama integration"""
    print("🧪 Testing Ollama Summarizer...")
    
    try:
        # Initialize
        summarizer = OllamaSummarizer(model_name="gemma3:1b")
        
        # Test text
        test_text = """
        L'intelligence artificielle (IA) est une technologie révolutionnaire qui transforme 
        profondément notre société moderne. Cette discipline informatique vise à créer des 
        systèmes capables de réaliser des tâches qui nécessitent normalement l'intelligence 
        humaine, comme la reconnaissance de formes, la prise de décision, l'apprentissage 
        et la résolution de problèmes complexes.
        
        Les applications de l'IA sont désormais omniprésentes dans notre quotidien. 
        On la retrouve dans les assistants vocaux comme Siri ou Alexa, les systèmes 
        de recommandation de Netflix ou YouTube, les voitures autonomes, la médecine 
        de précision, et même dans les systèmes de traduction automatique.
        """
        
        # Test short summary
        print("\n📝 Testing SHORT summary (French)...")
        short_summary = summarizer.summarize(test_text, summary_type="short")
        print(f"Short summary: {short_summary}")
        
        # Test long summary
        print("\n📝 Testing LONG summary (French)...")
        long_summary = summarizer.summarize(test_text, summary_type="long")
        print(f"Long summary: {long_summary}")
        
        # Model info
        print("\n📊 Model Information:")
        info = summarizer.get_model_info()
        print(f"Model: {info['model_name']}")
        print(f"Available models: {info.get('available_models', [])}")
        
        # Usage stats
        print("\n📈 Usage Statistics:")
        stats = summarizer.get_usage_stats()
        for key, value in stats.items():
            print(f"  {key}: {value}")
        
        print("\n✅ Test completed successfully!")
        
    except OllamaConnectionError as e:
        print(f"\n❌ Connection Error: {e}")
        print("\n💡 Make sure Ollama is running:")
        print("   1. Install: curl -fsSL https://ollama.com/install.sh | sh")
        print("   2. Start: ollama serve")
        print("   3. Pull model: ollama pull mistral:7b")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
