"""
Module d'évaluation automatique des résumés
Métriques ROUGE, BLEU, cohérence sémantique, et métriques personnalisées
"""

from .evaluator import SummaryEvaluator, EvaluationMetrics, EvaluationReport

__all__ = ['SummaryEvaluator', 'EvaluationMetrics', 'EvaluationReport']