#!/usr/bin/env python3
"""
Tests d'intégration bout-en-bout pour Video Summarizer
Tests des workflows complets: ingestion → preprocessing → résumé
"""

import sys
import os
from pathlib import Path

# Ajouter le répertoire src au PYTHONPATH
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

def test_complete_text_workflow():
    """Test complet: texte → préprocessing → résumé"""
    print("🔄 Test workflow complet: Texte → Résumé")
    
    try:
        from data.ingestion import DataIngestion
        from data.preprocessing import TextPreprocessor
        from models.model_manager import ModelManager
        
        # Texte de test
        test_text = """
        L'intelligence artificielle transforme notre monde de manière spectaculaire.
        Cette technologie révolutionnaire permet aux machines d'apprendre, de raisonner
        et de prendre des décisions comme les humains. Des assistants vocaux aux voitures
        autonomes, l'IA est désormais omniprésente dans notre quotidien. Cependant, cette
        révolution soulève des questions éthiques importantes sur l'avenir du travail,
        la vie privée et le contrôle de ces technologies puissantes.
        """
        
        # 1. Ingestion
        ingestion = DataIngestion()
        video_data = ingestion.process_text_input(test_text, "Test IA")
        print(f"   ✅ Ingestion: {len(video_data.transcript.split())} mots")
        
        # 2. Préprocessing
        preprocessor = TextPreprocessor()
        processed = preprocessor.preprocess(video_data.transcript)
        print(f"   ✅ Préprocessing: {processed.word_count} mots, langue: {processed.language}")
        
        # 3. Résumé (si OpenAI disponible)
        manager = ModelManager()
        if os.getenv('OPENAI_API_KEY'):
            summary = manager.summarize_simple(
                processed.text,
                model_type="openai",
                summary_length="short"
            )
            print(f"   ✅ Résumé généré: {len(summary.split())} mots")
            print(f"   📝 Résumé: {summary[:100]}...")
        else:
            print("   ⚠️  OpenAI non configuré, résumé ignoré")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Erreur workflow: {e}")
        return False

def test_youtube_integration():
    """Test d'intégration YouTube (si connexion Internet)"""
    print("\n🎥 Test intégration YouTube")
    
    try:
        from data.ingestion import DataIngestion
        from data.preprocessing import TextPreprocessor
        
        # URL de test courte
        test_url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
        
        # 1. Ingestion YouTube
        ingestion = DataIngestion()
        video_data = ingestion.process_youtube_url(test_url)
        print(f"   ✅ YouTube: {video_data.title[:50]}...")
        print(f"   📊 Transcript: {len(video_data.transcript.split())} mots")
        
        # 2. Préprocessing
        preprocessor = TextPreprocessor()
        processed = preprocessor.preprocess(video_data.transcript)
        print(f"   ✅ Nettoyage: {processed.word_count} mots finaux")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Erreur YouTube: {e}")
        print("   💡 Vérifiez votre connexion Internet")
        return False

def test_model_availability():
    """Test de disponibilité des modèles"""
    print("\n🤖 Test disponibilité des modèles")
    
    try:
        from models.model_manager import ModelManager, ModelType
        
        manager = ModelManager()
        
        # Test LED
        led_available, led_msg = manager.is_model_available(ModelType.LED)
        print(f"   🎯 LED: {'✅' if led_available else '❌'} {led_msg}")
        
        # Test OpenAI
        openai_available, openai_msg = manager.is_model_available(ModelType.OPENAI)
        print(f"   ⚡ OpenAI: {'✅' if openai_available else '❌'} {openai_msg}")
        
        # Recommandation
        test_text = "Test de recommandation de modèle pour ce texte court."
        recommended = manager.recommend_model(test_text, "balanced")
        print(f"   🎯 Recommandé: {recommended.value}")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Erreur modèles: {e}")
        return False

def test_configuration_loading():
    """Test de chargement des configurations"""
    print("\n⚙️ Test chargement configuration")
    
    try:
        import yaml
        
        # Test model_config.yaml
        model_config_path = Path(__file__).parent.parent / "config" / "model_config.yaml"
        if model_config_path.exists():
            with open(model_config_path, 'r', encoding='utf-8') as f:
                model_config = yaml.safe_load(f)
                print(f"   ✅ model_config.yaml: {len(model_config)} sections")
        else:
            print("   ❌ model_config.yaml manquant")
        
        # Test app_config.yaml
        app_config_path = Path(__file__).parent.parent / "config" / "app_config.yaml"
        if app_config_path.exists():
            with open(app_config_path, 'r', encoding='utf-8') as f:
                app_config = yaml.safe_load(f)
                print(f"   ✅ app_config.yaml: {len(app_config)} sections")
        else:
            print("   ❌ app_config.yaml manquant")
        
        # Test .env
        env_path = Path(__file__).parent.parent / ".env"
        if env_path.exists():
            print("   ✅ Fichier .env trouvé")
        else:
            print("   ⚠️  Fichier .env manquant (optionnel)")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Erreur configuration: {e}")
        return False

def main():
    """Tests d'intégration complets"""
    print("🧪 Video Summarizer - Tests d'Intégration")
    print("=" * 50)
    
    tests = [
        ("Workflow texte complet", test_complete_text_workflow),
        ("Intégration YouTube", test_youtube_integration),
        ("Disponibilité modèles", test_model_availability),
        ("Configuration", test_configuration_loading)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n📋 {test_name}")
        print("-" * 30)
        result = test_func()
        results.append(result)
    
    # Résumé final
    print("\n" + "=" * 50)
    print("📊 RÉSUMÉ DES TESTS D'INTÉGRATION")
    print("=" * 50)
    
    passed = sum(results)
    total = len(results)
    
    print(f"✅ Tests réussis: {passed}/{total}")
    print(f"❌ Tests échoués: {total - passed}/{total}")
    
    if passed == total:
        print("🎉 Tous les tests d'intégration passent !")
        print("🚀 Votre Video Summarizer est prêt à l'utilisation !")
    elif passed >= total * 0.75:
        print("🟡 La plupart des tests passent. Quelques ajustements mineurs.")
    else:
        print("🔴 Plusieurs tests échouent. Vérifiez la configuration.")
    
    print("\n💡 Pour utiliser l'application :")
    print("   • Interface web: streamlit run src/ui/streamlit_app.py")
    print("   • Script de lancement: python scripts/launch.py")
    print("   • Configuration OpenAI: Ajoutez OPENAI_API_KEY dans .env")

if __name__ == "__main__":
    main()
