"""
Système d'évaluation automatique des résumés
Métriques ROUGE, BLEU, cohérence sémantique, et métriques personnalisées
"""

import os
import json
import time
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import logging
from dataclasses import dataclass, asdict
from datetime import datetime

import numpy as np
from rouge import Rouge
import spacy
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)

@dataclass
class EvaluationMetrics:
    """Métriques d'évaluation d'un résumé"""
    # Métriques ROUGE
    rouge_1_f: float
    rouge_1_p: float  
    rouge_1_r: float
    rouge_2_f: float
    rouge_2_p: float
    rouge_2_r: float
    rouge_l_f: float
    rouge_l_p: float
    rouge_l_r: float
    
    # Métriques sémantiques
    semantic_similarity: float
    coherence_score: float
    
    # Métriques de qualité
    compression_ratio: float
    readability_score: float
    factual_consistency: float
    
    # Métriques techniques
    processing_time: float
    model_used: str
    timestamp: str
    
    # Score global
    overall_score: float

@dataclass
class EvaluationReport:
    """Rapport d'évaluation complet"""
    summary_id: str
    original_text: str
    generated_summary: str
    reference_summary: Optional[str]
    metrics: EvaluationMetrics
    model_config: Dict[str, Any]
    recommendations: List[str]

class SummaryEvaluator:
    """Évaluateur automatique de qualité des résumés"""
    
    def __init__(self, 
                 load_models: bool = True,
                 cache_dir: Optional[str] = None):
        """
        Initialise l'évaluateur
        
        Args:
            load_models: Charger les modèles d'évaluation
            cache_dir: Répertoire de cache pour les modèles
        """
        self.cache_dir = cache_dir or os.path.join(os.path.expanduser("~"), ".cache", "video_summarizer")
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # Initialiser les outils d'évaluation
        self.rouge = Rouge()
        self.nlp = None
        self.sentence_model = None
        
        if load_models:
            self._load_models()
    
    def _load_models(self):
        """Charge les modèles nécessaires à l'évaluation"""
        try:
            # Modèle spaCy pour l'analyse linguistique
            try:
                self.nlp = spacy.load("fr_core_news_sm")
                logger.info("Modèle spaCy français chargé")
            except OSError:
                try:
                    self.nlp = spacy.load("en_core_web_sm")
                    logger.info("Modèle spaCy anglais chargé")
                except OSError:
                    logger.warning("Aucun modèle spaCy disponible")
            
            # Modèle de similarité sémantique
            self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
            logger.info("Modèle de similarité sémantique chargé")
            
        except Exception as e:
            logger.error(f"Erreur lors du chargement des modèles: {e}")
    
    def evaluate_summary(self,
                        original_text: str,
                        generated_summary: str,
                        reference_summary: Optional[str] = None,
                        model_name: str = "unknown",
                        processing_time: float = 0.0) -> EvaluationReport:
        """
        Évalue un résumé selon plusieurs métriques
        
        Args:
            original_text: Texte original
            generated_summary: Résumé généré
            reference_summary: Résumé de référence (optionnel)
            model_name: Nom du modèle utilisé
            processing_time: Temps de traitement
            
        Returns:
            EvaluationReport: Rapport d'évaluation complet
        """
        start_time = time.time()
        
        # Générer un ID unique pour ce résumé
        summary_id = f"summary_{int(time.time())}_{hash(generated_summary) % 10000}"
        
        # Calculer les métriques
        metrics = self._calculate_metrics(
            original_text, 
            generated_summary, 
            reference_summary,
            model_name,
            processing_time
        )
        
        # Générer des recommandations
        recommendations = self._generate_recommendations(metrics, original_text, generated_summary)
        
        # Créer le rapport
        report = EvaluationReport(
            summary_id=summary_id,
            original_text=original_text,
            generated_summary=generated_summary,
            reference_summary=reference_summary,
            metrics=metrics,
            model_config={"model_name": model_name},
            recommendations=recommendations
        )
        
        evaluation_time = time.time() - start_time
        logger.info(f"Évaluation terminée en {evaluation_time:.2f}s")
        
        return report
    
    def _calculate_metrics(self,
                          original_text: str,
                          generated_summary: str,
                          reference_summary: Optional[str],
                          model_name: str,
                          processing_time: float) -> EvaluationMetrics:
        """Calcule toutes les métriques d'évaluation"""
        
        # Métriques ROUGE (si résumé de référence disponible)
        rouge_scores = self._calculate_rouge_scores(generated_summary, reference_summary)
        
        # Métriques sémantiques
        semantic_sim = self._calculate_semantic_similarity(original_text, generated_summary)
        coherence = self._calculate_coherence(generated_summary)
        
        # Métriques de qualité
        compression = self._calculate_compression_ratio(original_text, generated_summary)
        readability = self._calculate_readability(generated_summary)
        factual_consistency = self._calculate_factual_consistency(original_text, generated_summary)
        
        # Score global
        overall_score = self._calculate_overall_score(rouge_scores, semantic_sim, coherence, readability)
        
        return EvaluationMetrics(
            # ROUGE
            rouge_1_f=rouge_scores.get('rouge-1', {}).get('f', 0.0),
            rouge_1_p=rouge_scores.get('rouge-1', {}).get('p', 0.0),
            rouge_1_r=rouge_scores.get('rouge-1', {}).get('r', 0.0),
            rouge_2_f=rouge_scores.get('rouge-2', {}).get('f', 0.0),
            rouge_2_p=rouge_scores.get('rouge-2', {}).get('p', 0.0),
            rouge_2_r=rouge_scores.get('rouge-2', {}).get('r', 0.0),
            rouge_l_f=rouge_scores.get('rouge-l', {}).get('f', 0.0),
            rouge_l_p=rouge_scores.get('rouge-l', {}).get('p', 0.0),
            rouge_l_r=rouge_scores.get('rouge-l', {}).get('r', 0.0),
            
            # Sémantique
            semantic_similarity=semantic_sim,
            coherence_score=coherence,
            
            # Qualité
            compression_ratio=compression,
            readability_score=readability,
            factual_consistency=factual_consistency,
            
            # Technique
            processing_time=processing_time,
            model_used=model_name,
            timestamp=datetime.now().isoformat(),
            
            # Global
            overall_score=overall_score
        )
    
    def _calculate_rouge_scores(self, generated: str, reference: Optional[str]) -> Dict[str, Any]:
        """Calcule les scores ROUGE"""
        if not reference or not generated.strip():
            return {}
        
        try:
            scores = self.rouge.get_scores(generated, reference)[0]
            return scores
        except Exception as e:
            logger.warning(f"Erreur calcul ROUGE: {e}")
            return {}
    
    def _calculate_semantic_similarity(self, original: str, summary: str) -> float:
        """Calcule la similarité sémantique entre texte original et résumé"""
        if not self.sentence_model or not original.strip() or not summary.strip():
            return 0.0
        
        try:
            # Encoder les textes
            original_embedding = self.sentence_model.encode([original])
            summary_embedding = self.sentence_model.encode([summary])
            
            # Calculer la similarité cosinus
            similarity = cosine_similarity(original_embedding, summary_embedding)[0][0]
            return float(similarity)
            
        except Exception as e:
            logger.warning(f"Erreur similarité sémantique: {e}")
            return 0.0
    
    def _calculate_coherence(self, text: str) -> float:
        """Évalue la cohérence du texte"""
        if not text.strip():
            return 0.0
        
        # Métriques simples de cohérence
        sentences = text.split('.')
        if len(sentences) < 2:
            return 0.5
        
        # Vérifier la continuité thématique (approximation simple)
        words = text.lower().split()
        unique_words = set(words)
        lexical_diversity = len(unique_words) / len(words) if words else 0
        
        # Score basé sur la diversité lexicale et la longueur des phrases
        avg_sentence_length = len(words) / len(sentences)
        coherence_score = min(1.0, (lexical_diversity * 0.5) + (min(avg_sentence_length, 20) / 40))
        
        return coherence_score
    
    def _calculate_compression_ratio(self, original: str, summary: str) -> float:
        """Calcule le ratio de compression"""
        original_words = len(original.split())
        summary_words = len(summary.split())
        
        if original_words == 0:
            return 0.0
        
        return summary_words / original_words
    
    def _calculate_readability(self, text: str) -> float:
        """Évalue la lisibilité du texte (approximation du score de Flesch)"""
        if not text.strip():
            return 0.0
        
        sentences = len([s for s in text.split('.') if s.strip()])
        words = len(text.split())
        syllables = sum(self._count_syllables(word) for word in text.split())
        
        if sentences == 0 or words == 0:
            return 0.0
        
        # Approximation du score de Flesch
        avg_sentence_length = words / sentences
        avg_syllables_per_word = syllables / words
        
        flesch_score = 206.835 - (1.015 * avg_sentence_length) - (84.6 * avg_syllables_per_word)
        
        # Normaliser entre 0 et 1
        return max(0.0, min(1.0, flesch_score / 100))
    
    def _count_syllables(self, word: str) -> int:
        """Compte approximativement les syllabes dans un mot"""
        word = word.lower()
        vowels = "aeiouy"
        syllable_count = 0
        previous_was_vowel = False
        
        for char in word:
            is_vowel = char in vowels
            if is_vowel and not previous_was_vowel:
                syllable_count += 1
            previous_was_vowel = is_vowel
        
        # Au moins une syllabe par mot
        return max(1, syllable_count)
    
    def _calculate_factual_consistency(self, original: str, summary: str) -> float:
        """Évalue la cohérence factuelle (approximation)"""
        if not self.nlp:
            return 0.5  # Score neutre si pas de modèle NLP
        
        try:
            # Extraire les entités nommées
            original_doc = self.nlp(original)
            summary_doc = self.nlp(summary)
            
            original_entities = set([ent.text.lower() for ent in original_doc.ents])
            summary_entities = set([ent.text.lower() for ent in summary_doc.ents])
            
            if not summary_entities:
                return 0.7  # Score neutre si pas d'entités dans le résumé
            
            # Calculer le pourcentage d'entités cohérentes
            consistent_entities = original_entities.intersection(summary_entities)
            consistency_score = len(consistent_entities) / len(summary_entities)
            
            return min(1.0, consistency_score)
            
        except Exception as e:
            logger.warning(f"Erreur cohérence factuelle: {e}")
            return 0.5
    
    def _calculate_overall_score(self, rouge_scores: Dict, semantic_sim: float, 
                               coherence: float, readability: float) -> float:
        """Calcule un score global pondéré"""
        # Poids pour chaque métrique
        weights = {
            'rouge': 0.3,
            'semantic': 0.25,
            'coherence': 0.25,
            'readability': 0.2
        }
        
        # Score ROUGE moyen
        rouge_f_scores = [
            rouge_scores.get('rouge-1', {}).get('f', 0),
            rouge_scores.get('rouge-2', {}).get('f', 0),
            rouge_scores.get('rouge-l', {}).get('f', 0)
        ]
        avg_rouge = np.mean([s for s in rouge_f_scores if s > 0]) if any(rouge_f_scores) else 0.5
        
        # Score global pondéré
        overall = (
            weights['rouge'] * avg_rouge +
            weights['semantic'] * semantic_sim +
            weights['coherence'] * coherence +
            weights['readability'] * readability
        )
        
        return min(1.0, max(0.0, overall))
    
    def _generate_recommendations(self, metrics: EvaluationMetrics, 
                                 original: str, summary: str) -> List[str]:
        """Génère des recommandations d'amélioration"""
        recommendations = []
        
        # Recommandations basées sur les métriques
        if metrics.compression_ratio > 0.8:
            recommendations.append("Le résumé pourrait être plus concis")
        elif metrics.compression_ratio < 0.1:
            recommendations.append("Le résumé pourrait être plus détaillé")
        
        if metrics.coherence_score < 0.6:
            recommendations.append("Améliorer la cohérence et la fluidité du résumé")
        
        if metrics.readability_score < 0.5:
            recommendations.append("Simplifier le vocabulaire pour améliorer la lisibilité")
        
        if metrics.semantic_similarity < 0.6:
            recommendations.append("Renforcer la fidélité sémantique au texte original")
        
        if metrics.processing_time > 60:
            recommendations.append("Optimiser les performances pour réduire le temps de traitement")
        
        if not recommendations:
            recommendations.append("Résumé de bonne qualité globale")
        
        return recommendations
    
    def batch_evaluate(self, summaries: List[Dict[str, Any]]) -> List[EvaluationReport]:
        """Évalue un lot de résumés"""
        reports = []
        
        for i, summary_data in enumerate(summaries):
            logger.info(f"Évaluation {i+1}/{len(summaries)}")
            
            report = self.evaluate_summary(
                original_text=summary_data.get('original_text', ''),
                generated_summary=summary_data.get('generated_summary', ''),
                reference_summary=summary_data.get('reference_summary'),
                model_name=summary_data.get('model_name', 'unknown'),
                processing_time=summary_data.get('processing_time', 0.0)
            )
            
            reports.append(report)
        
        return reports
    
    def save_report(self, report: EvaluationReport, filepath: str):
        """Sauvegarde un rapport d'évaluation"""
        report_data = {
            'summary_id': report.summary_id,
            'original_text': report.original_text,
            'generated_summary': report.generated_summary,
            'reference_summary': report.reference_summary,
            'metrics': asdict(report.metrics),
            'model_config': report.model_config,
            'recommendations': report.recommendations
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(report_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Rapport sauvegardé: {filepath}")

def create_evaluation_dataset():
    """Crée un jeu de données d'évaluation avec références"""
    dataset = [
        {
            "original_text": """
            L'intelligence artificielle (IA) représente l'une des révolutions technologiques 
            les plus importantes de notre époque. Cette technologie permet aux machines 
            d'apprendre, de raisonner et de prendre des décisions de manière autonome. 
            Les applications sont multiples : reconnaissance vocale, vision par ordinateur, 
            traduction automatique, véhicules autonomes, et diagnostic médical assisté. 
            Cependant, cette évolution soulève des questions éthiques sur l'avenir du travail, 
            la protection de la vie privée, et le contrôle de ces technologies puissantes.
            """,
            "reference_summary": "L'IA transforme notre époque en permettant aux machines d'apprendre et de décider de façon autonome, avec des applications variées mais soulevant des questions éthiques importantes.",
            "topic": "Intelligence Artificielle"
        },
        {
            "original_text": """
            Le réchauffement climatique constitue l'un des défis majeurs du 21ème siècle. 
            Les activités humaines, notamment les émissions de gaz à effet de serre, 
            provoquent une augmentation des températures moyennes globales. Les conséquences 
            incluent la fonte des glaciers, l'élévation du niveau des mers, des événements 
            météorologiques extrêmes plus fréquents, et des perturbations des écosystèmes. 
            La transition vers les énergies renouvelables et la réduction des émissions 
            sont essentielles pour limiter l'impact climatique.
            """,
            "reference_summary": "Le réchauffement climatique, causé par les émissions humaines, entraîne des changements dramatiques nécessitant une transition urgente vers les énergies renouvelables.",
            "topic": "Environnement"
        }
    ]
    
    return dataset

if __name__ == "__main__":
    # Test de l'évaluateur
    evaluator = SummaryEvaluator()
    
    # Données de test
    test_data = create_evaluation_dataset()[0]
    
    # Simulation d'un résumé généré
    generated_summary = "L'IA révolutionne notre monde en donnant aux machines des capacités d'apprentissage et de décision, mais pose des défis éthiques."
    
    # Évaluation
    report = evaluator.evaluate_summary(
        original_text=test_data["original_text"],
        generated_summary=generated_summary,
        reference_summary=test_data["reference_summary"],
        model_name="test_model"
    )
    
    print(f"Score global: {report.metrics.overall_score:.3f}")
    print(f"Similarité sémantique: {report.metrics.semantic_similarity:.3f}")
    print(f"ROUGE-1 F: {report.metrics.rouge_1_f:.3f}")
    print(f"Recommandations: {report.recommendations}")