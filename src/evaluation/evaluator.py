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

logger = logging.getLogger(__name__)

@dataclass
class EvaluationMetrics:
    """Summary evaluation metrics - Simplified and reliable version"""
    
    # Main metrics
    bert_score: float              # BERTScore - Contextual semantic similarity
    compression_quality: float     # Compression with quality control
    
    # Additional simple metrics
    word_overlap_ratio: float      # Important words overlap
    sentence_coherence: float      # True coherence between sentences
    
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
        """Compatibility: redirects to sentence_coherence"""
        return self.sentence_coherence
    
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
        
        if load_models:
            self._load_models()
    
    def _load_models(self):
        """Load necessary models for evaluation"""
        try:
            # Lighter multilingual model for BERTScore
            self.sentence_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
            logger.info("Multilingual model loaded for BERTScore")
            
        except Exception as e:
            logger.error(f"Error loading models: {e}")
            self.sentence_model = None
    
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
        
        # Simple metrics
        word_overlap = self._calculate_word_overlap(original_text, generated_summary)
        sentence_coherence = self._calculate_sentence_coherence(generated_summary)
        
        # Global score (weighted average)
        overall = self._calculate_simple_overall(
            bert_score, compression_quality, word_overlap, sentence_coherence
        )
        
        return EvaluationMetrics(
            # Main metrics
            bert_score=bert_score,
            compression_quality=compression_quality,
            word_overlap_ratio=word_overlap,
            sentence_coherence=sentence_coherence,
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
        Important words overlap
        
        Simple measure: percentage of important words from summary present in original
        """
        original_words = set(w.lower() for w in re.findall(r'\b\w{4,}\b', original))
        summary_words = set(w.lower() for w in re.findall(r'\b\w{4,}\b', summary))
        
        if not summary_words:
            return 0.0
        
        overlap = len(original_words & summary_words) / len(summary_words)
        return min(1.0, overlap)
    
    def _calculate_sentence_coherence(self, text: str) -> float:
        """
        Sentence coherence
        
        Measures thematic continuity between consecutive sentences
        """
        if not self.sentence_model:
            return 0.8  # Default score
        
        sentences = [s.strip() for s in re.split(r'[.!?]+', text) if s.strip()]
        
        if len(sentences) < 2:
            return 1.0  # Single sentence = coherent by default
        
        try:
            embeddings = self.sentence_model.encode(sentences)
            
            # Calculate similarity between consecutive sentences
            coherence_scores = []
            for i in range(len(embeddings) - 1):
                sim = cosine_similarity([embeddings[i]], [embeddings[i + 1]])[0][0]
                coherence_scores.append(max(0, sim))  # Avoid negative scores
            
            return float(np.mean(coherence_scores))
            
        except Exception as e:
            logger.warning(f"Coherence error: {e}")
            return 0.8
    
    def _calculate_simple_overall(self, bert_score: float, 
                                 compression_quality: float, word_overlap: float,
                                 sentence_coherence: float) -> float:
        """
        Global score with balanced weights
        
        Weights based on importance and reliability of metrics
        ROUGE-L weight (0.25) redistributed to sentence_coherence (0.10 -> 0.35)
        """
        weights = {
            'bert_score': 0.35,          # Most reliable metric
            'compression_quality': 0.20,  # Summary efficiency
            'word_overlap': 0.10,        # Basic verification
            'sentence_coherence': 0.35   # Fluidity + structure
        }
        
        overall = (
            weights['bert_score'] * bert_score +
            weights['compression_quality'] * compression_quality +
            weights['word_overlap'] * word_overlap +
            weights['sentence_coherence'] * sentence_coherence
        )
        
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