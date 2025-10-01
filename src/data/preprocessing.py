"""
Module de préprocessing des données textuelles
Nettoyage, segmentation et préparation des transcripts
"""

import re
import logging
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass
import pandas as pd
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)


@dataclass
class ProcessedData:
    """Structure pour les données préprocessées"""
    text: str
    word_count: int
    sentence_count: int
    language: str
    segments: List[str]
    metadata: Dict[str, Any]


class TextCleaner:
    """Nettoyeur de texte avancé"""
    
    def __init__(self):
        # Patterns de nettoyage
        self.patterns = {
            'youtube_timestamps': r'\[\d+:\d+\]',  # [00:15]
            'speaker_labels': r'^[A-Z\s]+:',  # SPEAKER:
            'music_tags': r'\[Música\]|\[Music\]|\[♪\]',
            'applause_tags': r'\[Applaudissements\]|\[Applause\]|\[Rires\]|\[Laughter\]',
            'extra_spaces': r'\s+',
            'repeated_chars': r'(.)\1{3,}',  # Caractères répétés plus de 3 fois
            'urls': r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+',
            'email': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        }
    
    def clean_text(self, text: str) -> str:
        """
        Nettoie le texte selon les patterns définis
        
        Args:
            text: Texte à nettoyer
            
        Returns:
            str: Texte nettoyé
        """
        if not text:
            return ""
        
        # Appliquer les patterns de nettoyage
        cleaned_text = text
        
        # Supprimer les timestamps YouTube
        cleaned_text = re.sub(self.patterns['youtube_timestamps'], '', cleaned_text)
        
        # Supprimer les labels de speakers
        cleaned_text = re.sub(self.patterns['speaker_labels'], '', cleaned_text, flags=re.MULTILINE)
        
        # Supprimer les tags musicaux et autres annotations
        cleaned_text = re.sub(self.patterns['music_tags'], '', cleaned_text, flags=re.IGNORECASE)
        cleaned_text = re.sub(self.patterns['applause_tags'], '', cleaned_text, flags=re.IGNORECASE)
        
        # Supprimer les URLs
        cleaned_text = re.sub(self.patterns['urls'], '', cleaned_text)
        
        # Supprimer les emails
        cleaned_text = re.sub(self.patterns['email'], '', cleaned_text)
        
        # Nettoyer les caractères répétés
        cleaned_text = re.sub(self.patterns['repeated_chars'], r'\1\1', cleaned_text)
        
        # Nettoyer les espaces multiples
        cleaned_text = re.sub(self.patterns['extra_spaces'], ' ', cleaned_text)
        
        # Supprimer les espaces en début et fin
        cleaned_text = cleaned_text.strip()
        
        return cleaned_text
    
    def normalize_text(self, text: str) -> str:
        """Normalise le texte (casse, ponctuation, etc.)"""
        if not text:
            return ""
        
        # Normaliser les apostrophes
        text = text.replace(''', "'").replace(''', "'")
        
        # Normaliser les guillemets
        text = text.replace('"', '"').replace('"', '"')
        
        # Normaliser les tirets
        text = text.replace('—', '-').replace('–', '-')
        
        # Ajouter des espaces après la ponctuation si nécessaire
        text = re.sub(r'([.!?])([A-Z])', r'\1 \2', text)
        
        return text


class TextSegmenter:
    """Segmenteur de texte en phrases et paragraphes"""
    
    def __init__(self, max_segment_length: int = 1000):
        self.max_segment_length = max_segment_length
    
    def segment_by_sentences(self, text: str) -> List[str]:
        """Segmente le texte en phrases"""
        if not text:
            return []
        
        # Pattern pour détecter les fins de phrases
        sentence_pattern = r'(?<=[.!?])\s+'
        sentences = re.split(sentence_pattern, text)
        
        # Nettoyer les phrases vides
        sentences = [s.strip() for s in sentences if s.strip()]
        
        return sentences
    
    def segment_by_length(self, text: str) -> List[str]:
        """Segmente le texte en chunks de longueur maximale"""
        if not text:
            return []
        
        words = text.split()
        segments = []
        current_segment = []
        current_length = 0
        
        for word in words:
            word_length = len(word) + 1  # +1 pour l'espace
            
            if current_length + word_length > self.max_segment_length and current_segment:
                segments.append(' '.join(current_segment))
                current_segment = [word]
                current_length = word_length
            else:
                current_segment.append(word)
                current_length += word_length
        
        # Ajouter le dernier segment
        if current_segment:
            segments.append(' '.join(current_segment))
        
        return segments
    
    def smart_segment(self, text: str) -> List[str]:
        """Segmentation intelligente combinant phrases et longueur"""
        sentences = self.segment_by_sentences(text)
        segments = []
        current_segment = []
        current_length = 0
        
        for sentence in sentences:
            sentence_length = len(sentence)
            
            if current_length + sentence_length > self.max_segment_length and current_segment:
                segments.append(' '.join(current_segment))
                current_segment = [sentence]
                current_length = sentence_length
            else:
                current_segment.append(sentence)
                current_length += sentence_length + 1  # +1 pour l'espace
        
        # Ajouter le dernier segment
        if current_segment:
            segments.append(' '.join(current_segment))
        
        return segments


class LanguageDetector:
    """Détecteur de langue simple"""
    
    def __init__(self):
        # Mots indicateurs de langue
        self.french_indicators = {
            'le', 'la', 'les', 'de', 'du', 'des', 'et', 'est', 'une', 'un',
            'que', 'qui', 'avec', 'dans', 'pour', 'sur', 'par', 'ce', 'cette'
        }
        
        self.english_indicators = {
            'the', 'and', 'is', 'are', 'was', 'were', 'of', 'to', 'in', 'for',
            'with', 'on', 'at', 'by', 'this', 'that', 'these', 'those'
        }
    
    def detect_language(self, text: str) -> str:
        """
        Détecte la langue du texte (français ou anglais)
        
        Args:
            text: Texte à analyser
            
        Returns:
            str: Code de langue ('fr' ou 'en')
        """
        if not text:
            return 'unknown'
        
        words = set(text.lower().split())
        
        french_score = len(words.intersection(self.french_indicators))
        english_score = len(words.intersection(self.english_indicators))
        
        if french_score > english_score:
            return 'fr'
        elif english_score > french_score:
            return 'en'
        else:
            return 'unknown'


class TextPreprocessor:
    """Préprocesseur principal de texte"""
    
    def __init__(self, max_segment_length: int = 1000):
        self.cleaner = TextCleaner()
        self.segmenter = TextSegmenter(max_segment_length)
        self.language_detector = LanguageDetector()
    
    def preprocess(self, text: str, detect_language: bool = True) -> ProcessedData:
        """
        Préprocesse un texte complet
        
        Args:
            text: Texte à préprocesser
            detect_language: Si True, détecte automatiquement la langue
            
        Returns:
            ProcessedData: Données préprocessées
        """
        if not text:
            return ProcessedData(
                text="",
                word_count=0,
                sentence_count=0,
                language="unknown",
                segments=[],
                metadata={}
            )
        
        # Nettoyer et normaliser le texte
        cleaned_text = self.cleaner.clean_text(text)
        normalized_text = self.cleaner.normalize_text(cleaned_text)
        
        # Détecter la langue
        language = self.language_detector.detect_language(normalized_text) if detect_language else "unknown"
        
        # Segmenter le texte
        segments = self.segmenter.smart_segment(normalized_text)
        
        # Calculer les statistiques
        word_count = len(normalized_text.split())
        sentence_count = len(self.segmenter.segment_by_sentences(normalized_text))
        
        # Métadonnées
        metadata = {
            'original_length': len(text),
            'cleaned_length': len(normalized_text),
            'reduction_ratio': 1 - (len(normalized_text) / len(text)) if len(text) > 0 else 0,
            'avg_segment_length': sum(len(seg.split()) for seg in segments) / len(segments) if segments else 0,
            'num_segments': len(segments)
        }
        
        return ProcessedData(
            text=normalized_text,
            word_count=word_count,
            sentence_count=sentence_count,
            language=language,
            segments=segments,
            metadata=metadata
        )


class DatasetPreprocessor:
    """Préprocesseur pour datasets d'entraînement"""
    
    def __init__(self, test_size: float = 0.2, val_size: float = 0.1, random_state: int = 42):
        self.test_size = test_size
        self.val_size = val_size
        self.random_state = random_state
        self.preprocessor = TextPreprocessor()
    
    def prepare_training_data(self, 
                            transcripts: List[str], 
                            summaries: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Prépare les données pour l'entraînement
        
        Args:
            transcripts: Liste des transcripts
            summaries: Liste des résumés correspondants
            
        Returns:
            Tuple[DataFrame, DataFrame, DataFrame]: train, validation, test sets
        """
        # Créer un DataFrame
        df = pd.DataFrame({
            'transcript': transcripts,
            'summary': summaries
        })
        
        # Préprocesser les textes
        processed_transcripts = []
        processed_summaries = []
        
        for transcript, summary in zip(transcripts, summaries):
            proc_transcript = self.preprocessor.preprocess(transcript)
            proc_summary = self.preprocessor.preprocess(summary)
            
            # Filtrer les textes trop courts ou trop longs
            if (proc_transcript.word_count > 50 and proc_transcript.word_count < 2000 and
                proc_summary.word_count > 10 and proc_summary.word_count < 500):
                processed_transcripts.append(proc_transcript.text)
                processed_summaries.append(proc_summary.text)
        
        # Créer le DataFrame final
        final_df = pd.DataFrame({
            'transcript': processed_transcripts,
            'summary': processed_summaries
        })
        
        # Division train/test
        train_data, test_data = train_test_split(
            final_df, 
            test_size=self.test_size, 
            random_state=self.random_state
        )
        
        # Division validation à partir du test
        if self.val_size > 0:
            val_data, test_data = train_test_split(
                test_data,
                test_size=0.5,
                random_state=self.random_state
            )
        else:
            val_data = pd.DataFrame(columns=final_df.columns)
        
        logger.info(f"Données préparées: {len(train_data)} train, {len(val_data)} val, {len(test_data)} test")
        
        return train_data, val_data, test_data
    
    def filter_by_quality(self, df: pd.DataFrame, min_words: int = 50, max_words: int = 2000) -> pd.DataFrame:
        """Filtre les données selon des critères de qualité"""
        def word_count(text):
            return len(str(text).split())
        
        # Calculer le nombre de mots
        df['transcript_words'] = df['transcript'].apply(word_count)
        df['summary_words'] = df['summary'].apply(word_count)
        
        # Filtrer
        filtered_df = df[
            (df['transcript_words'] >= min_words) & 
            (df['transcript_words'] <= max_words) &
            (df['summary_words'] >= 10) &
            (df['summary_words'] <= 500)
        ].copy()
        
        # Supprimer les colonnes temporaires
        filtered_df = filtered_df.drop(['transcript_words', 'summary_words'], axis=1)
        
        logger.info(f"Filtrage: {len(df)} -> {len(filtered_df)} exemples")
        
        return filtered_df


if __name__ == "__main__":
    # Test du préprocesseur
    preprocessor = TextPreprocessor()
    
    test_text = """
    [00:15] Bonjour à tous ! Aujourd'hui nous allons parler de l'intelligence artificielle...
    [Musique] C'est vraiment fascinant !! Et nous verrons comment ça marche.
    """
    
    result = preprocessor.preprocess(test_text)
    print(f"Texte original: {test_text}")
    print(f"Texte nettoyé: {result.text}")
    print(f"Langue détectée: {result.language}")
    print(f"Nombre de mots: {result.word_count}")
    print(f"Segments: {len(result.segments)}")