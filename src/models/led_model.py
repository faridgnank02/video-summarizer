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
            if torch.cuda.is_available():
                return "cuda"
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                return "mps"  # Metal Performance Shaders sur Mac
            else:
                return "cpu"
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
            
            # Essayer différentes approches de chargement
            self.model = None
            
            # Approche 1: Chargement normal avec optimisations
            try:
                # Optimiser le dtype selon le device
                if self.device == "mps":
                    # MPS supporte mieux float32 pour LED
                    dtype = torch.float32
                elif self.device == "cuda":
                    dtype = torch.float16
                else:
                    dtype = torch.float32
                
                self.model = AutoModelForSeq2SeqLM.from_pretrained(
                    self.model_name,
                    gradient_checkpointing=True,
                    use_cache=False,
                    torch_dtype=dtype
                )
                logger.info(f"Modèle chargé avec optimisations (dtype={dtype}) pour {self.device}")
            except Exception as e1:
                logger.warning(f"Échec chargement optimisé: {e1}")
                
                # Approche 2: Chargement basique
                try:
                    self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)
                    logger.info("Modèle chargé en mode basique")
                except Exception as e2:
                    logger.error(f"Échec chargement basique: {e2}")
                    
                    # Approche 3: Forcer CPU
                    try:
                        self.device = "cpu"
                        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)
                        logger.info("Modèle chargé en mode CPU forcé")
                    except Exception as e3:
                        logger.error(f"Tous les modes de chargement ont échoué: {e3}")
                        raise e3
            
            if self.model is None:
                raise RuntimeError("Impossible de charger le modèle LED")
            
            # Déplacer vers le device (avec gestion d'erreur)
            try:
                self.model.to(self.device)
            except Exception as e:
                logger.warning(f"Erreur déplacement vers {self.device}: {e}")
                self.device = "cpu"
                self.model.to(self.device)
                logger.info("Fallback vers CPU réussi")
            
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
        
        # Validation et nettoyage préalable du texte
        cleaned_text = self._preprocess_input_text(text)
        if not self._is_text_valid_for_summarization(cleaned_text):
            return "Le texte fourni ne semble pas exploitable pour un résumé de qualité. Veuillez vérifier la transcription ou essayer un autre contenu."
        
        try:
            # Ajuster les longueurs selon le type de résumé
            if summary_type == "short":
                max_length = max_length or 200
                min_length = min_length or 50
            else:  # long
                max_length = max_length or 512
                min_length = min_length or 100
            
            # Tokeniser le texte d'entrée nettoyé
            inputs_dict = self.tokenizer(
                cleaned_text,
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
            
            # Nettoyer le résumé des artefacts de génération
            summary = self._clean_generated_text(summary)
            
            # Validation finale du résumé généré
            if not self._is_summary_coherent(summary):
                logger.warning("Résumé généré incohérent, tentative avec paramètres conservateurs")
                return self._generate_conservative_summary(cleaned_text, max_length, min_length)
            
            return summary.strip()
            
        except Exception as e:
            logger.error(f"Erreur lors de la génération du résumé: {e}")
            return f"Erreur lors de la génération: {str(e)}"
    
    def _preprocess_input_text(self, text: str) -> str:
        """
        Préprocessing robuste du texte d'entrée pour éviter les artifacts
        
        Args:
            text: Texte brut d'entrée
            
        Returns:
            str: Texte nettoyé et validé
        """
        import re
        
        # Nettoyage initial
        cleaned = text.strip()
        
        # Supprimer les artefacts de transcription YouTube courants
        artifacts_patterns = [
            r'\b[a-z]{1,2}\s+[a-z]{1,2}\s+[a-z]{1,2}\b',  # Séquences de petits mots isolés
            r'\b(le|la|de|du|et|à|un|une|ce|il|elle|vous|tu|je|ne|pas|pour|avec|sur|dans|par|mais|ou|et|donc|or|ni|car|que|qui|quoi|où|quand|comment|pourquoi)\s+\1+\b',  # Répétitions de mots outils
            r'\b[bcdfghjklmnpqrstvwxz]{3,}\b',  # Séquences de consonnes sans voyelles
            r'\b[aeiouàâäéèêëïîôöùûüÿ]{4,}\b',  # Séquences de voyelles trop longues
            r'\b\w{1,2}(\s+\w{1,2}){10,}\b',  # Séquences de mots très courts
        ]
        
        for pattern in artifacts_patterns:
            cleaned = re.sub(pattern, '', cleaned, flags=re.IGNORECASE)
        
        # Supprimer les caractères répétitifs suspects
        cleaned = re.sub(r'(.)\1{3,}', r'\1\1', cleaned)  # Maximum 2 répétitions
        
        # Nettoyer les espaces multiples
        cleaned = re.sub(r'\s+', ' ', cleaned).strip()
        
        # Filtrer les phrases trop courtes ou incohérentes
        sentences = re.split(r'[.!?]+', cleaned)
        valid_sentences = []
        
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) > 10 and self._is_sentence_coherent(sentence):
                valid_sentences.append(sentence)
        
        result = '. '.join(valid_sentences)
        if result:
            result += '.'
        
        return result
    
    def _is_text_valid_for_summarization(self, text: str) -> bool:
        """
        Valide si le texte est exploitable pour un résumé de qualité
        
        Args:
            text: Texte à valider
            
        Returns:
            bool: True si le texte est valide
        """
        if not text or len(text.strip()) < 50:
            return False
        
        words = text.split()
        if len(words) < 10:
            return False
        
        # Vérifier le ratio de mots cohérents
        coherent_words = 0
        total_meaningful_words = 0
        
        for word in words:
            if len(word) >= 2:  # Mots de 2+ lettres
                total_meaningful_words += 1
                if self._is_word_coherent(word):
                    coherent_words += 1
        
        coherence_ratio = coherent_words / total_meaningful_words if total_meaningful_words else 0
        
        # Vérifier aussi la longueur moyenne des mots (indicateur de qualité)
        avg_word_length = sum(len(w) for w in words) / len(words) if words else 0
        
        # Critères plus stricts
        return (coherence_ratio >= 0.75 and     # Au moins 75% de mots cohérents
                avg_word_length >= 3.5 and      # Mots assez longs en moyenne
                len([w for w in words if len(w) >= 4]) >= len(words) * 0.4)  # 40% mots longs
    
    def _is_sentence_coherent(self, sentence: str) -> bool:
        """Vérifie si une phrase semble cohérente"""
        words = sentence.split()
        if len(words) < 3:
            return False
        
        # Vérifier qu'il y a au moins quelques mots de plus de 3 lettres
        long_words = [w for w in words if len(w) > 3]
        return len(long_words) >= len(words) * 0.4
    
    def _is_word_coherent(self, word: str) -> bool:
        """Vérifie si un mot semble cohérent (pas un artifact)"""
        if len(word) < 2:
            return False
        
        # Vérifier qu'il y a des voyelles
        vowels = 'aeiouàâäéèêëïîôöùûüÿAEIOUÀÂÄÉÈÊËÏÎÔÖÙÛÜŸ'
        has_vowel = any(c in vowels for c in word)
        
        # Vérifier qu'il n'y a pas que des répétitions
        unique_chars = set(word.lower())
        
        return has_vowel and len(unique_chars) >= 2
    
    def _is_summary_coherent(self, summary: str) -> bool:
        """Vérifie si le résumé généré est cohérent"""
        if not summary or len(summary.strip()) < 20:
            return False
        
        words = summary.split()
        if len(words) < 5:
            return False
        
        # Vérifier la cohérence des mots
        coherent_words = sum(1 for word in words if self._is_word_coherent(word))
        coherence_ratio = coherent_words / len(words)
        
        # Vérifier qu'il n'y a pas trop de répétitions
        unique_words = set(word.lower() for word in words)
        diversity_ratio = len(unique_words) / len(words)
        
        return coherence_ratio >= 0.7 and diversity_ratio >= 0.3
    
    def _generate_conservative_summary(self, text: str, max_length: int, min_length: int) -> str:
        """
        Génère un résumé avec des paramètres très conservateurs
        """
        try:
            # Utiliser des paramètres très conservateurs
            inputs_dict = self.tokenizer(
                text,
                padding="max_length",
                max_length=min(2048, self.config['max_input_length']),  # Réduire la longueur
                return_tensors="pt",
                truncation=True
            )
            
            input_ids = inputs_dict.input_ids.to(self.device)
            attention_mask = inputs_dict.attention_mask.to(self.device)
            
            # Paramètres très conservateurs
            with torch.no_grad():
                predicted_ids = self.model.generate(
                    input_ids,
                    attention_mask=attention_mask,
                    max_length=min(max_length, 200),  # Limiter davantage
                    min_length=max(min_length, 30),
                    num_beams=1,  # Greedy decoding
                    length_penalty=1.0,  # Neutre
                    early_stopping=True,
                    no_repeat_ngram_size=4,  # Plus strict
                    do_sample=False,
                    temperature=1.0
                )
            
            summary = self.tokenizer.batch_decode(predicted_ids, skip_special_tokens=True)[0]
            
            # Si même cela échoue, extraire les premières phrases
            if not self._is_summary_coherent(summary):
                import re
                sentences = re.split(r'[.!?]+', text)
                valid_sentences = [s.strip() for s in sentences[:3] if len(s.strip()) > 20]
                if valid_sentences:
                    return '. '.join(valid_sentences) + '.'
                else:
                    return "Le contenu fourni ne permet pas de générer un résumé cohérent."
            
            return self._clean_generated_text(summary)
            
        except Exception as e:
            logger.error(f"Erreur génération conservative: {e}")
            return "Impossible de générer un résumé à partir de ce contenu."

    def _clean_generated_text(self, text: str) -> str:
        """
        Nettoie le texte généré des artefacts communs - Version améliorée
        
        Args:
            text: Texte généré brut
            
        Returns:
            str: Texte nettoyé
        """
        import re
        
        # Supprimer les caractères de contrôle et artefacts Unicode
        text = re.sub(r'[^\w\s\-.,;:!?\'"àâäéèêëïîôöùûüÿçñÀÂÄÉÈÊËÏÎÔÖÙÛÜŸÇÑ]', '', text)
        
        # Supprimer les artefacts spécifiques observés plus agressivement
        artifacts_to_remove = [
            r'\b(vai?s?|voi?s?|vu?s?|va?i?t?|vait?|vant?|vents?)\b',  # Artifacts "vai", "vois", etc.
            r'\b(mai?s?|moi?s?|mu?s?|maoi?s?)\b',  # Artifacts "mais", "mois", etc.
            r'\b(le?a?|la?e?|de?e?|du?u?)\b(?=\s+\1)',  # Répétitions d'articles
            r'\b[a-z]{1,2}(\s+[a-z]{1,2}){5,}\b',  # Séquences de petits mots
            r'\b\w*[àâäéèêëïîôöùûüÿ]{3,}\w*\b',  # Mots avec trop d'accents
        ]
        
        for pattern in artifacts_to_remove:
            text = re.sub(pattern, '', text, flags=re.IGNORECASE)
        
        # Corriger les espaces multiples
        text = re.sub(r'\s+', ' ', text)
        
        # Supprimer les répétitions de mots consécutifs (version plus stricte)
        words = text.split()
        cleaned_words = []
        
        for i, word in enumerate(words):
            # Éviter les répétitions dans une fenêtre de 3 mots
            should_keep = True
            for j in range(max(0, i-2), i):
                if j < len(cleaned_words) and self._is_similar_word(word, cleaned_words[j]):
                    should_keep = False
                    break
            
            if should_keep and len(word) > 1:
                cleaned_words.append(word)
        
        # Reconstituer le texte
        cleaned_text = ' '.join(cleaned_words)
        
        # Supprimer les séquences de caractères répétitifs
        cleaned_text = re.sub(r'(.)\1{2,}', r'\1', cleaned_text)
        
        # Corriger la ponctuation
        cleaned_text = re.sub(r'\s+([,.;:!?])', r'\1', cleaned_text)
        cleaned_text = re.sub(r'([.!?])\s*([A-Z])', r'\1 \2', cleaned_text)
        
        # Assurer que le texte se termine proprement
        cleaned_text = cleaned_text.strip()
        if cleaned_text and not cleaned_text[-1] in '.!?':
            cleaned_text += '.'
        
        return cleaned_text
    
    def _is_similar_word(self, word1: str, word2: str) -> bool:
        """
        Vérifie si deux mots sont similaires (pour éviter les répétitions)
        
        Args:
            word1, word2: Mots à comparer
            
        Returns:
            bool: True si les mots sont similaires
        """
        if not word1 or not word2:
            return False
        
        # Mots identiques en minuscules
        if word1.lower() == word2.lower():
            return True
        
        # Mots très courts et similaires
        if len(word1) <= 3 and len(word2) <= 3:
            return abs(len(word1) - len(word2)) <= 1 and word1[:2].lower() == word2[:2].lower()
        
        # Similarité par distance de Levenshtein simple
        if len(word1) > 3 and len(word2) > 3:
            common_chars = sum(1 for a, b in zip(word1.lower(), word2.lower()) if a == b)
            return common_chars / max(len(word1), len(word2)) > 0.8
        
        return False
    
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