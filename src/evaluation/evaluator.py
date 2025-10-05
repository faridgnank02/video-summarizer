"""
Automatic summary evaluation system
Simple and reliable metrics: BERTScore, Compression Quality, Sentence Coherence
"""

import os
import json
import time
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import logging
from dataclasses import dataclass, asdict
from datetime import datetime
import re

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import string
from collections import Counter

# Import spaCy pour NER si disponible
try:
    import spacy
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False

logger = logging.getLogger(__name__)

@dataclass
class EvaluationMetrics:
    """Summary evaluation metrics - Simplified and reliable version"""
    
    # Main metrics
    bert_score: float              # BERTScore - Contextual semantic similarity
    compression_quality: float     # Compression with quality control
    
    # Additional simple metrics
    word_overlap_ratio: float      # Hybrid NER + keyword overlap
    
    # Technical metrics
    processing_time: float
    model_used: str
    timestamp: str
    
    # Simplified global score
    overall_score: float
    
    @property
    def semantic_similarity(self) -> float:
        """Compatibility: redirects to bert_score"""
        return self.bert_score
    
    @property
    def coherence_score(self) -> float:
        """Compatibility: redirects to word_overlap for backward compatibility"""
        return self.word_overlap_ratio
    
    @property
    def compression_ratio(self) -> float:
        """Compatibility: simple ratio version"""
        return min(1.0, self.compression_quality * 1.2)  # Approximation

@dataclass
class EvaluationReport:
    """Complete evaluation report"""
    summary_id: str
    original_text: str
    generated_summary: str
    reference_summary: Optional[str]
    metrics: EvaluationMetrics
    model_config: Dict[str, Any]
    recommendations: List[str]

class SummaryEvaluator:
    """Automatic summary quality evaluator"""
    
    def __init__(self, 
                 load_models: bool = True,
                 cache_dir: Optional[str] = None):
        """
        Initialize the evaluator with metrics
        
        Args:
            load_models: Load evaluation models
            cache_dir: Cache directory for models
        """
        self.cache_dir = cache_dir or os.path.join(os.path.expanduser("~"), ".cache", "video_summarizer")
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # Initialize only necessary tools
        self.sentence_model = None
        self.nlp_model = None
        self.spacy_available = False
        
        # Stop words for multilingual support
        self.french_stopwords = {
            'le', 'la', 'les', 'un', 'une', 'des', 'de', 'du', 'et', 'ou', 'mais', 
            'donc', 'car', 'ni', 'or', 'ce', 'cette', 'ces', 'son', 'sa', 'ses',
            'dans', 'sur', 'avec', 'par', 'pour', 'en', 'à', 'au', 'aux', 'est', 
            'sont', 'était', 'ont', 'avoir', 'être', 'faire', 'dire', 'aller'
        }
        
        self.english_stopwords = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have',
            'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should'
        }
        
        if load_models:
            self._load_models()
    
    def _load_models(self):
        """Load necessary models for evaluation"""
        try:
            # Lighter multilingual model for BERTScore
            self.sentence_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
            logger.info("Multilingual model loaded for BERTScore")
            
        except Exception as e:
            logger.error(f"Error loading sentence model: {e}")
            self.sentence_model = None
        
        # Load spaCy model for NER if available
        if SPACY_AVAILABLE:
            try:
                self.nlp_model = spacy.load('en_core_web_sm')
                self.spacy_available = True
                logger.info("spaCy model loaded for NER")
            except OSError:
                logger.warning("spaCy model 'en_core_web_sm' not available. NER will be disabled.")
                self.spacy_available = False
        else:
            logger.warning("spaCy not installed. NER will be disabled.")
    
    def evaluate_summary(self,
                        original_text: str,
                        generated_summary: str,
                        reference_summary: Optional[str] = None,
                        model_name: str = "unknown",
                        processing_time: float = 0.0) -> EvaluationReport:
        """
        Evaluate a summary according to metrics
        
        Args:
            original_text: Original text
            generated_summary: Generated summary
            reference_summary: Reference summary (optional, ignored)
            model_name: Name of the model used
            processing_time: Processing time
            
        Returns:
            EvaluationReport: Complete evaluation report
        """
        start_time = time.time()
        
        # Generate unique ID for this summary
        summary_id = f"summary_{int(time.time())}_{hash(generated_summary) % 10000}"
        
        # Calculate metrics
        metrics = self._calculate_metrics(
            original_text, 
            generated_summary, 
            reference_summary,
            model_name,
            processing_time
        )
        
        # Create report (no recommendations)
        report = EvaluationReport(
            summary_id=summary_id,
            original_text=original_text,
            generated_summary=generated_summary,
            reference_summary=reference_summary,
            metrics=metrics,
            model_config={"model_name": model_name},
            recommendations=[]  # No recommendations
        )
        
        evaluation_time = time.time() - start_time
        logger.info(f"Evaluation completed in {evaluation_time:.2f}s")
        
        return report
    
    def _calculate_metrics(self,
                          original_text: str,
                          generated_summary: str,
                          reference_summary: Optional[str],
                          model_name: str,
                          processing_time: float) -> EvaluationMetrics:
        """Calculate all evaluation metrics"""

        # 1. Approximate BERTScore (contextual semantic similarity)
        bert_score = self._calculate_bert_score(original_text, generated_summary)
        
        # 2. Compression Ratio with quality control
        compression_quality = self._calculate_compression_quality(original_text, generated_summary)
        
        # Hybrid word overlap (NER + Keywords)
        word_overlap = self._calculate_word_overlap(original_text, generated_summary)
        
        # Global score (weighted average)
        overall = self._calculate_simple_overall(
            bert_score, compression_quality, word_overlap
        )
        
        return EvaluationMetrics(
            # Main metrics
            bert_score=bert_score,
            compression_quality=compression_quality,
            word_overlap_ratio=word_overlap,
            # Technical metrics
            processing_time=processing_time,
            model_used=model_name,
            timestamp=datetime.now().isoformat(),
            # Global score
            overall_score=overall
        )
    
    def _calculate_bert_score(self, original: str, summary: str) -> float:
        """
        Approximate BERTScore - Measures contextual semantic similarity
        
        Principle: Compares contextual representations of words/sentences
        between original text and summary. Closer to human evaluation
        than metrics based on word overlap.
        
        Score: 0-1 (0.6+ = good, 0.8+ = excellent)
        """
        if not self.sentence_model or not original.strip() or not summary.strip():
            return 0.5
        
        try:
            # Split into sentences for finer comparison
            original_sentences = [s.strip() for s in re.split(r'[.!?]+', original) if s.strip()]
            summary_sentences = [s.strip() for s in re.split(r'[.!?]+', summary) if s.strip()]
            
            if not original_sentences or not summary_sentences:
                return 0.0
            
            # Encode all sentences
            original_embeddings = self.sentence_model.encode(original_sentences)
            summary_embeddings = self.sentence_model.encode(summary_sentences)
            
            # For each summary sentence, find best match in original
            similarities = []
            for sum_emb in summary_embeddings:
                sum_emb = sum_emb.reshape(1, -1)
                sentence_similarities = cosine_similarity(sum_emb, original_embeddings)[0]
                similarities.append(max(sentence_similarities))
            
            # Final score = average of best similarities
            bert_score = np.mean(similarities)
            
            logger.debug(f"BERTScore: {bert_score:.3f}")
            return float(bert_score)
            
        except Exception as e:
            logger.warning(f"BERTScore error: {e}")
            return 0.5
    

    
    def _calculate_compression_quality(self, original: str, summary: str) -> float:
        """
        Compression Ratio with quality control
        
        Principle: A good summary should have an appropriate compression ratio
        (neither too short = info loss, nor too long = not synthetic enough).
        Combines compression ratio and information density.
        
        Score: 0-1 (0.7-0.9 = optimal depending on text length)
        """
        original_words = len(original.split())
        summary_words = len(summary.split())
        
        if original_words == 0 or summary_words == 0:
            return 0.0
        
        # Compression ratio
        compression_ratio = summary_words / original_words
        
        # Optimal ratios according to text length
        if original_words < 200:          # Short text
            optimal_range = (0.3, 0.7)
        elif original_words < 1000:       # Medium text
            optimal_range = (0.1, 0.3)
        else:                            # Long text
            optimal_range = (0.05, 0.15)
        
        # Score based on proximity to optimal ratio
        if optimal_range[0] <= compression_ratio <= optimal_range[1]:
            ratio_score = 1.0
        elif compression_ratio < optimal_range[0]:
            # Too compressed
            ratio_score = compression_ratio / optimal_range[0]
        else:
            # Not compressed enough
            ratio_score = max(0.0, 1.0 - (compression_ratio - optimal_range[1]) / optimal_range[1])
        
        # Bonus for information density
        density_bonus = self._calculate_information_density_simple(summary)
        
        # Final score
        compression_quality = 0.7 * ratio_score + 0.3 * density_bonus
        
        logger.debug(f"Compression: {compression_ratio:.3f} (optimal: {optimal_range}), "
                    f"score: {compression_quality:.3f}")
        
        return min(1.0, compression_quality)
    
    def _calculate_information_density_simple(self, text: str) -> float:
        """Simple measure of information density"""
        words = text.split()
        if not words:
            return 0.0
        
        # "Informative" words (not stopwords)
        stopwords = {
            'le', 'la', 'les', 'un', 'une', 'des', 'de', 'du', 'et', 'ou', 'mais', 
            'donc', 'car', 'ni', 'or', 'ce', 'cette', 'ces', 'son', 'sa', 'ses',
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for'
        }
        
        informative_words = [w for w in words if w.lower() not in stopwords and len(w) > 2]
        
        # Density = informative words / total words
        density = len(informative_words) / len(words)
        
        return min(1.0, density * 1.5)  # Boost since it's a simple approximation
    
    def _calculate_word_overlap(self, original: str, summary: str) -> float:
        """
        Hybrid word overlap using NER + Keywords
        
        Combines Named Entity Recognition for important entities
        with keyword overlap for comprehensive coverage.
        Falls back to keyword-only if NER unavailable.
        
        Score: 0-1 (higher = better overlap)
        """
        if not original.strip() or not summary.strip():
            return 0.0
        
        # Configuration des poids
        ner_weight = 0.6  # Poids des entités nommées
        keyword_weight = 0.4  # Poids des mots-clés
        
        # Composante NER
        ner_f1 = self._calculate_ner_overlap(original, summary)
        
        # Composante keyword overlap
        keyword_score = self._calculate_keyword_overlap(original, summary)
        
        # Score hybride pondéré
        if self.spacy_available and ner_f1 > 0:
            # Utiliser NER si disponible et pertinent
            hybrid_score = ner_weight * ner_f1 + keyword_weight * keyword_score
            logger.debug(f"Hybrid overlap: NER={ner_f1:.3f}, Keywords={keyword_score:.3f}, Combined={hybrid_score:.3f}")
        else:
            # Fallback vers keyword seulement
            hybrid_score = keyword_score
            logger.debug(f"Keyword-only overlap: {hybrid_score:.3f}")
        
        return min(1.0, hybrid_score)
    
    def _detect_language(self, text: str) -> str:
        """Détection simple de langue basée sur les mots-outils"""
        words = text.lower().split()
        french_count = sum(1 for word in words if word in self.french_stopwords)
        english_count = sum(1 for word in words if word in self.english_stopwords)
        
        return 'french' if french_count > english_count else 'english'
    
    def _extract_named_entities(self, text: str) -> list:
        """Extraction des entités nommées si spaCy disponible"""
        if not self.spacy_available or not self.nlp_model:
            return []
        
        try:
            doc = self.nlp_model(text)
            entities = []
            
            # Types d'entités importantes
            important_types = ['PERSON', 'ORG', 'GPE', 'DATE', 'MONEY', 'PRODUCT', 
                             'EVENT', 'FAC', 'LOC', 'TIME', 'PERCENT']
            
            for ent in doc.ents:
                if ent.label_ in important_types:
                    entities.append({
                        'text': ent.text.lower().strip(),
                        'label': ent.label_
                    })
            
            return entities
        except Exception as e:
            logger.warning(f"Erreur extraction NER: {e}")
            return []
    
    def _calculate_ner_overlap(self, original: str, summary: str) -> float:
        """Calcul de l'overlap des entités nommées"""
        original_entities = self._extract_named_entities(original)
        summary_entities = self._extract_named_entities(summary)
        
        if not original_entities and not summary_entities:
            return 1.0  # Si aucune entité, considérer comme parfait
        
        if not original_entities or not summary_entities:
            return 0.0
        
        original_set = set(ent['text'] for ent in original_entities)
        summary_set = set(ent['text'] for ent in summary_entities)
        
        overlapping = original_set.intersection(summary_set)
        
        precision = len(overlapping) / len(summary_set) if summary_set else 0
        recall = len(overlapping) / len(original_set) if original_set else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        return f1
    
    def _calculate_keyword_overlap(self, original: str, summary: str) -> float:
        """Calcul d'overlap de mots-clés amélioré"""
        # Préprocessing
        def preprocess_text(text):
            text = text.lower()
            # Gérer apostrophes françaises
            text = re.sub(r"\b[ldn]'", "", text)
            text = text.translate(str.maketrans('', '', string.punctuation))
            return text
        
        original_clean = preprocess_text(original)
        summary_clean = preprocess_text(summary)
        
        # Détecter la langue pour les stop words
        language = self._detect_language(original)
        stopwords = self.french_stopwords if language == 'french' else self.english_stopwords
        
        # Extraire les mots significatifs (3+ caractères, pas de stop words)
        original_words = set(w for w in original_clean.split() 
                           if len(w) >= 3 and w not in stopwords)
        summary_words = set(w for w in summary_clean.split() 
                          if len(w) >= 3 and w not in stopwords)
        
        if not summary_words:
            return 0.0
        
        overlap_words = original_words.intersection(summary_words)
        return len(overlap_words) / len(summary_words)
    
    def _simple_word_overlap(self, original: str, summary: str) -> float:
        """Fallback simple word overlap method (deprecated, kept for compatibility)"""
        return self._calculate_keyword_overlap(original, summary)
    

    
    def _calculate_simple_overall(self, bert_score: float, 
                                 compression_quality: float, word_overlap: float) -> float:
        """
        Global score with redistributed weights (after removing sentence coherence)
        """
        weights = {
            'bert_score': 0.50,          # Increased: Most reliable semantic metric
            'compression_quality': 0.20,  # Unchanged: Summary efficiency
            'word_overlap': 0.30,        # Increased: Hybrid NER+keyword overlap
        }
        
        overall = (
            weights['bert_score'] * bert_score +
            weights['compression_quality'] * compression_quality +
            weights['word_overlap'] * word_overlap
        )
        
        logger.debug(f"Overall score: BERT={bert_score:.3f}({weights['bert_score']}), "
                    f"Compression={compression_quality:.3f}({weights['compression_quality']}), "
                    f"WordOverlap={word_overlap:.3f}({weights['word_overlap']}) = {overall:.3f}")
        
        return min(1.0, overall)
    
    def _generate_recommendations(self, metrics: EvaluationMetrics, 
                                 original: str, summary: str) -> List[str]:
        """Generate recommendations - removed as requested"""
        return []  # No recommendations as requested
    
    def batch_evaluate(self, summaries: List[Dict[str, Any]]) -> List[EvaluationReport]:
        """Evaluate multiple summaries in batch"""
        reports = []
        
        for i, summary_data in enumerate(summaries):
            logger.info(f"Evaluation {i+1}/{len(summaries)}")
            
            try:
                report = self.evaluate_summary(
                    original_text=summary_data.get("original_text", ""),
                    generated_summary=summary_data.get("generated_summary", ""),
                    reference_summary=summary_data.get("reference_summary"),
                    model_name=summary_data.get("model_name", "unknown"),
                    processing_time=summary_data.get("processing_time", 0.0)
                )
                reports.append(report)
                
            except Exception as e:
                logger.error(f"Evaluation error {i+1}: {e}")
        
        return reports
    
    def save_report(self, report: EvaluationReport, filepath: str):
        """Save an evaluation report"""
        try:
            # Convert to dictionary for serialization
            report_dict = {
                "summary_id": report.summary_id,
                "original_text": report.original_text,
                "generated_summary": report.generated_summary,
                "reference_summary": report.reference_summary,
                "metrics": asdict(report.metrics),
                "model_config": report.model_config,
                "recommendations": report.recommendations
            }
            
            # Save as JSON
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(report_dict, f, indent=2, ensure_ascii=False)
                
            logger.info(f"Report saved: {filepath}")
            
        except Exception as e:
            logger.error(f"Error saving report: {e}")