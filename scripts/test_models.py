#!/usr/bin/env python3
"""
Script de test pour vérifier la disponibilité des modèles
"""

import sys
import os
from pathlib import Path

# Ajouter le chemin du projet
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from models.model_manager import ModelManager, ModelType

def test_models():
    """Test de disponibilité des modèles"""
    print("🧪 Test de disponibilité des modèles")
    print("=" * 50)
    
    # Initialiser le manager
    config_path = project_root / "config" / "model_config.yaml"
    manager = ModelManager(str(config_path) if config_path.exists() else None)
    
    # Tester LED
    print("\n🔍 Test modèle LED:")
    led_available, led_msg = manager.is_model_available(ModelType.LED)
    if led_available:
        print("✅ LED disponible")
    else:
        print(f"❌ LED indisponible: {led_msg}")
    
    # Tester OpenAI
    print("\n🔍 Test modèle OpenAI:")
    openai_available, openai_msg = manager.is_model_available(ModelType.OPENAI)
    if openai_available:
        print("✅ OpenAI disponible")
    else:
        print(f"❌ OpenAI indisponible: {openai_msg}")
    
    # Test de résumé simple
    print("\n🚀 Test de résumé avec fallback automatique:")
    test_text = """
    Ceci est un texte de test pour vérifier le fonctionnement du système de résumé 
    avec fallback automatique. Le système devrait automatiquement basculer vers 
    OpenAI si LED n'est pas disponible sur ce système.
    """
    
    try:
        summary = manager.summarize_simple(
            text=test_text,
            model_type="led",  # Forcer LED pour tester le fallback
            summary_length="short"
        )
        print(f"✅ Résumé généré: {summary[:100]}...")
    except Exception as e:
        print(f"❌ Erreur lors du résumé: {e}")
    
    print(f"\n📊 Statistiques:")
    print(f"- Requêtes LED: {manager.stats['led_requests']}")
    print(f"- Requêtes OpenAI: {manager.stats['openai_requests']}")
    print(f"- Temps moyen: {manager.stats['average_processing_time']:.2f}s")

if __name__ == "__main__":
    test_models()