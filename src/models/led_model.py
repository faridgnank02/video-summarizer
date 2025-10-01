"""
Modèle LED (Longformer Encoder-Decoder) pour le résumé
Basé sur le code du notebook avec améliorations
"""

import os
import torch
import logging
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import yaml

from transformers import (
    AutoTokenizer, 
    AutoModelForSeq2SeqLM,
    LEDTokenizer,
    LEDForConditionalGeneration
)

logger = logging.getLogger(__name__)


class LEDSummarizer:
    """Modèle LED pour le résumé de texte"""
    
    def __init__(self, 
                 model_name: str = "allenai/led-base-16384",
                 config_path: Optional[str] = None,
                 device: str = "auto"):
        """
        Initialise le modèle LED
        
        Args:
            model_name: Nom du modèle pré-entraîné ou chemin vers modèle local
            config_path: Chemin vers le fichier de configuration
            device: Device à utiliser ('cuda', 'cpu', 'auto')
        """
        self.model_name = model_name
        self.device = self._get_device(device)
        
        # Charger la configuration
        self.config = self._load_config(config_path)
        
        # Initialiser le tokenizer et le modèle
        self.tokenizer = None
        self.model = None
        self._load_model()
        
        logger.info(f"LED Summarizer initialisé avec le modèle: {model_name}")
        logger.info(f"Device utilisé: {self.device}")
    
    def _get_device(self, device: str) -> str:
        """Détermine le device à utiliser"""
        if device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return device
    
    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """Charge la configuration depuis un fichier YAML"""
        default_config = {
            'max_input_length': 7168,
            'max_output_length': 512,
            'generation_config': {
                'num_beams': 2,
                'max_length': 512,
                'min_length': 100,
                'length_penalty': 2.0,
                'early_stopping': True,
                'no_repeat_ngram_size': 3
            }
        }
        
        if config_path and os.path.exists(config_path):
            try:
                with open(config_path, 'r', encoding='utf-8') as f:
                    file_config = yaml.safe_load(f)
                    led_config = file_config.get('models', {}).get('led', {})
                    default_config.update(led_config)
            except Exception as e:
                logger.warning(f"Erreur lors du chargement de la config: {e}")
        
        return default_config
    
    def _load_model(self):
        """Charge le tokenizer et le modèle"""
        try:
            # Charger le tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            
            # Charger le modèle
            self.model = AutoModelForSeq2SeqLM.from_pretrained(
                self.model_name,
                gradient_checkpointing=True,
                use_cache=False
            )
            
            # Déplacer vers le device
            self.model.to(self.device)
            
            # Configurer les paramètres de génération
            self._configure_generation()
            
        except Exception as e:
            logger.error(f"Erreur lors du chargement du modèle: {e}")
            raise
    
    def _configure_generation(self):
        """Configure les paramètres de génération du modèle"""
        gen_config = self.config['generation_config']
        
        self.model.config.num_beams = gen_config.get('num_beams', 2)
        self.model.config.max_length = gen_config.get('max_length', 512)
        self.model.config.min_length = gen_config.get('min_length', 100)
        self.model.config.length_penalty = gen_config.get('length_penalty', 2.0)
        self.model.config.early_stopping = gen_config.get('early_stopping', True)
        self.model.config.no_repeat_ngram_size = gen_config.get('no_repeat_ngram_size', 3)
    
    def summarize(self, 
                  text: str, 
                  max_length: Optional[int] = None,
                  min_length: Optional[int] = None,
                  summary_type: str = "long") -> str:
        """
        Génère un résumé du texte donné
        
        Args:
            text: Texte à résumer
            max_length: Longueur maximale du résumé (optionnel)
            min_length: Longueur minimale du résumé (optionnel)
            summary_type: Type de résumé ('short' ou 'long')
            
        Returns:
            str: Résumé généré
        """
        if not text or not text.strip():
            return ""
        
        try:
            # Ajuster les longueurs selon le type de résumé
            if summary_type == "short":
                max_length = max_length or 200
                min_length = min_length or 50
            else:  # long
                max_length = max_length or 512
                min_length = min_length or 100
            
            # Tokeniser le texte d'entrée
            inputs_dict = self.tokenizer(
                text,
                padding="max_length",
                max_length=self.config['max_input_length'],
                return_tensors="pt",
                truncation=True
            )
            
            # Déplacer vers le device
            input_ids = inputs_dict.input_ids.to(self.device)
            attention_mask = inputs_dict.attention_mask.to(self.device)
            
            # Créer le masque d'attention globale
            global_attention_mask = torch.zeros_like(attention_mask)
            global_attention_mask[:, 0] = 1  # Attention globale sur le token <s>
            
            # Générer le résumé
            with torch.no_grad():
                predicted_ids = self.model.generate(
                    input_ids,
                    attention_mask=attention_mask,
                    global_attention_mask=global_attention_mask,
                    max_length=max_length,
                    min_length=min_length,
                    num_beams=self.model.config.num_beams,
                    length_penalty=self.model.config.length_penalty,
                    early_stopping=self.model.config.early_stopping,
                    no_repeat_ngram_size=self.model.config.no_repeat_ngram_size
                )
            
            # Décoder le résumé
            summary = self.tokenizer.batch_decode(
                predicted_ids, 
                skip_special_tokens=True
            )[0]
            
            return summary.strip()
            
        except Exception as e:
            logger.error(f"Erreur lors de la génération du résumé: {e}")
            return f"Erreur lors de la génération: {str(e)}"
    
    def batch_summarize(self, 
                       texts: List[str], 
                       batch_size: int = 1,
                       summary_type: str = "long") -> List[str]:
        """
        Génère des résumés pour une liste de textes
        
        Args:
            texts: Liste des textes à résumer
            batch_size: Taille du batch (recommandé: 1 pour LED)
            summary_type: Type de résumé ('short' ou 'long')
            
        Returns:
            List[str]: Liste des résumés générés
        """
        summaries = []
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            
            for text in batch_texts:
                summary = self.summarize(text, summary_type=summary_type)
                summaries.append(summary)
                
            logger.info(f"Traité {min(i + batch_size, len(texts))}/{len(texts)} textes")
        
        return summaries
    
    def get_model_info(self) -> Dict[str, Any]:
        """Retourne les informations sur le modèle"""
        return {
            'model_name': self.model_name,
            'device': self.device,
            'config': self.config,
            'tokenizer_vocab_size': len(self.tokenizer) if self.tokenizer else 0,
            'model_parameters': sum(p.numel() for p in self.model.parameters()) if self.model else 0
        }
    
    def save_model(self, save_path: str):
        """Sauvegarde le modèle fine-tuné"""
        try:
            Path(save_path).mkdir(parents=True, exist_ok=True)
            
            self.model.save_pretrained(save_path)
            self.tokenizer.save_pretrained(save_path)
            
            # Sauvegarder la configuration
            config_path = Path(save_path) / "led_config.yaml"
            with open(config_path, 'w', encoding='utf-8') as f:
                yaml.dump(self.config, f, default_flow_style=False)
            
            logger.info(f"Modèle sauvegardé dans: {save_path}")
            
        except Exception as e:
            logger.error(f"Erreur lors de la sauvegarde: {e}")
            raise
    
    def load_finetuned_model(self, model_path: str):
        """Charge un modèle fine-tuné"""
        try:
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Modèle non trouvé: {model_path}")
            
            self.model_name = model_path
            self._load_model()
            
            # Charger la configuration spécifique si elle existe
            config_path = Path(model_path) / "led_config.yaml"
            if config_path.exists():
                with open(config_path, 'r', encoding='utf-8') as f:
                    saved_config = yaml.safe_load(f)
                    self.config.update(saved_config)
                    self._configure_generation()
            
            logger.info(f"Modèle fine-tuné chargé depuis: {model_path}")
            
        except Exception as e:
            logger.error(f"Erreur lors du chargement du modèle fine-tuné: {e}")
            raise


# Fonction de compatibilité avec le code original
def create_led_summarizer(model_path: Optional[str] = None) -> LEDSummarizer:
    """
    Crée une instance de LEDSummarizer
    
    Args:
        model_path: Chemin vers un modèle fine-tuné (optionnel)
        
    Returns:
        LEDSummarizer: Instance configurée
    """
    summarizer = LEDSummarizer()
    
    if model_path and os.path.exists(model_path):
        summarizer.load_finetuned_model(model_path)
    
    return summarizer


if __name__ == "__main__":
    # Test du modèle LED
    print("Test du modèle LED...")
    
    summarizer = LEDSummarizer()
    
    test_text = """
    L'intelligence artificielle (IA) est une technologie révolutionnaire qui transforme notre monde.
    Elle permet aux machines d'apprendre, de raisonner et de prendre des décisions de manière autonome.
    Les applications de l'IA sont nombreuses : reconnaissance vocale, vision par ordinateur, 
    traduction automatique, et bien d'autres. Cependant, cette technologie soulève aussi des questions
    éthiques importantes concernant l'emploi, la vie privée et l'autonomie humaine.
    Il est crucial de développer l'IA de manière responsable pour maximiser ses bénéfices
    tout en minimisant les risques potentiels.
    """
    
    # Test résumé court
    short_summary = summarizer.summarize(test_text, summary_type="short")
    print(f"Résumé court: {short_summary}")
    
    # Test résumé long
    long_summary = summarizer.summarize(test_text, summary_type="long")
    print(f"Résumé long: {long_summary}")
    
    # Informations sur le modèle
    info = summarizer.get_model_info()
    print(f"Paramètres du modèle: {info['model_parameters']:,}")