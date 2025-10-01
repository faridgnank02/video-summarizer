#!/usr/bin/env python3
"""
Test simple des fonctionnalités principales du Video Summarizer
sans interface Streamlit
"""

import sys
import os
from pathlib import Path

# Ajouter le répertoire src au PYTHONPATH
sys.path.insert(0, str(Path(__file__).parent / "src"))

def test_youtube_ingestion():
    """Test de l'ingestion YouTube"""
    print("🔍 Test de l'ingestion YouTube...")
    
    try:
        from data.ingestion import DataIngestion
        
        ingestion = DataIngestion()
        
        # Test avec une URL YouTube courte et populaire
        test_url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"  # Rick Roll
        
        print(f"📹 Extraction du transcript de: {test_url}")
        video_data = ingestion.process_youtube_url(test_url)
        
        print(f"✅ Titre: {video_data.title}")
        print(f"📊 Longueur: {len(video_data.transcript.split())} mots")
        print(f"🌍 Langue: {video_data.language}")
        print(f"📝 Extrait: {video_data.transcript[:100]}...")
        
        return video_data
        
    except Exception as e:
        print(f"❌ Erreur ingestion YouTube: {e}")
        return None

def test_preprocessing(video_data):
    """Test du préprocessing"""
    print("\n🔧 Test du préprocessing...")
    
    try:
        from data.preprocessing import TextPreprocessor
        
        preprocessor = TextPreprocessor()
        
        result = preprocessor.preprocess(video_data.transcript)
        
        print(f"✅ Texte préprocessé")
        print(f"📊 Mots: {result.word_count}")
        print(f"📄 Phrases: {result.sentence_count}")
        print(f"🧹 Réduction: {result.metadata['reduction_ratio']:.1%}")
        print(f"📝 Extrait nettoyé: {result.text[:100]}...")
        
        return result
        
    except Exception as e:
        print(f"❌ Erreur preprocessing: {e}")
        return None

def test_text_only_summarization():
    """Test de résumé avec texte direct (sans YouTube)"""
    print("\n📝 Test de résumé avec texte direct...")
    
    try:
        from data.ingestion import DataIngestion
        from data.preprocessing import TextPreprocessor
        from models.model_manager import ModelManager
        
        # Texte d'exemple
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
        
        Cependant, cette révolution technologique soulève également des questions 
        éthiques et sociétales importantes. Les préoccupations incluent l'impact 
        sur l'emploi, la protection de la vie privée, les biais dans les algorithmes, 
        et les risques liés à une automatisation excessive. Il est donc crucial de 
        développer l'IA de manière responsable et éthique.
        
        L'avenir de l'IA promet des avancées encore plus spectaculaires, avec le 
        développement de l'intelligence artificielle générale (AGI) qui pourrait 
        égaler ou surpasser les capacités cognitives humaines dans tous les domaines. 
        Cette perspective excitante nécessite une réflexion approfondie sur les 
        implications pour l'humanité.
        """
        
        # Ingestion
        ingestion = DataIngestion()
        video_data = ingestion.process_text_input(test_text, "L'Intelligence Artificielle")
        
        print(f"✅ Texte ingéré: {video_data.title}")
        print(f"📊 Longueur: {len(video_data.transcript.split())} mots")
        
        # Préprocessing
        preprocessor = TextPreprocessor()
        processed = preprocessor.preprocess(video_data.transcript)
        
        print(f"✅ Texte préprocessé: {processed.word_count} mots")
        
        # Gestionnaire de modèles
        print("🤖 Initialisation du gestionnaire de modèles...")
        manager = ModelManager()
        
        # Recommandation de modèle
        recommended = manager.recommend_model(processed.text, "balanced")
        print(f"🎯 Modèle recommandé: {recommended.value}")
        
        # Test de disponibilité
        from models.model_manager import ModelType
        
        led_available, led_msg = manager.is_model_available(ModelType.LED)
        openai_available, openai_msg = manager.is_model_available(ModelType.OPENAI)
        
        print(f"🔍 LED disponible: {'✅' if led_available else '❌'} {led_msg}")
        print(f"🔍 OpenAI disponible: {'✅' if openai_available else '❌'} {openai_msg}")
        
        # Essayer de générer un résumé si possible
        if openai_available:
            print("\n⚡ Test de résumé avec OpenAI...")
            try:
                summary = manager.summarize_simple(
                    processed.text,
                    model_type="openai",
                    summary_length="short"
                )
                print(f"✅ Résumé généré: {summary}")
            except Exception as e:
                print(f"❌ Erreur résumé OpenAI: {e}")
        
        elif led_available:
            print("\n🎯 Test de résumé avec LED...")
            print("⚠️  LED nécessite beaucoup de ressources, test ignoré")
        
        else:
            print("\n⚠️  Aucun modèle disponible pour le résumé")
            print("💡 Configurez OPENAI_API_KEY pour tester OpenAI")
        
        return True
        
    except Exception as e:
        print(f"❌ Erreur test résumé: {e}")
        return False

def main():
    """Fonction principale"""
    print("🎥 Video Summarizer - Test des Fonctionnalités")
    print("=" * 50)
    
    # Test 1: Ingestion YouTube (si Internet disponible)
    print("Test 1: Ingestion YouTube")
    video_data = test_youtube_ingestion()
    
    if video_data:
        # Test 2: Préprocessing
        print("\nTest 2: Préprocessing")
        processed_data = test_preprocessing(video_data)
    
    # Test 3: Résumé avec texte direct
    print("\nTest 3: Résumé avec texte direct")
    test_text_only_summarization()
    
    print("\n" + "=" * 50)
    print("🎉 Tests terminés !")
    print("\n💡 Pour utiliser l'interface web:")
    print("   1. Configurez OPENAI_API_KEY dans .env")
    print("   2. Lancez: streamlit run src/ui/streamlit_app.py")
    print("   3. Ou utilisez: python launch.py")

if __name__ == "__main__":
    main()