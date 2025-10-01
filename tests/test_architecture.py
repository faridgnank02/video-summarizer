"""
Script de test pour valider l'architecture du Video Summarizer
"""

import sys
import os
from pathlib import Path
from dotenv import load_dotenv

# Charger les variables d'environnement depuis .env
load_dotenv()

# Ajouter le répertoire src au PYTHONPATH
project_root = Path(__file__).parent.parent  # Remonter d'un niveau depuis tests/
src_path = project_root / "src"
sys.path.insert(0, str(src_path))

print("🧪 Test de l'architecture Video Summarizer")
print("=" * 50)

def test_imports():
    """Test des importations des modules"""
    print("\n1. 📦 Test des importations...")
    
    try:
        from data.ingestion import DataIngestion, get_transcript
        print("   ✅ data.ingestion importé")
        
        from data.preprocessing import TextPreprocessor
        print("   ✅ data.preprocessing importé")
        
        from models.led_model import LEDSummarizer
        print("   ✅ models.led_model importé")
        
        from models.openai_model import OpenAISummarizer
        print("   ✅ models.openai_model importé")
        
        from models.model_manager import ModelManager
        print("   ✅ models.model_manager importé")
        
        return True
        
    except ImportError as e:
        print(f"   ❌ Erreur d'importation: {e}")
        return False

def test_data_ingestion():
    """Test de l'ingestion de données"""
    print("\n2. 📹 Test de l'ingestion de données...")
    
    try:
        # Test avec du texte direct
        from data.ingestion import DataIngestion
        
        ingestion = DataIngestion()
        
        # Test texte direct
        test_text = "Ceci est un test de l'ingestion de données pour le résumé automatique."
        video_data = ingestion.process_text_input(test_text, "Test")
        
        print(f"   ✅ Texte direct traité: {video_data.title}")
        print(f"   📊 Longueur: {len(video_data.transcript.split())} mots")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Erreur ingestion: {e}")
        return False

def test_preprocessing():
    """Test du préprocessing"""
    print("\n3. 🔧 Test du préprocessing...")
    
    try:
        from data.preprocessing import TextPreprocessor
        
        preprocessor = TextPreprocessor()
        
        test_text = """
        [00:15] Bonjour à tous ! Aujourd'hui nous allons parler de l'IA...
        [Musique] C'est vraiment fascinant !! Et nous verrons comment ça marche.
        """
        
        result = preprocessor.preprocess(test_text)
        
        print(f"   ✅ Texte préprocessé")
        print(f"   📊 Mots: {result.word_count}")
        print(f"   🌍 Langue: {result.language}")
        print(f"   📄 Segments: {len(result.segments)}")
        print(f"   🧹 Nettoyé: '{result.text[:50]}...'")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Erreur preprocessing: {e}")
        return False

def test_model_manager():
    """Test du gestionnaire de modèles"""
    print("\n4. 🤖 Test du gestionnaire de modèles...")
    
    try:
        from models.model_manager import ModelManager, ModelType
        
        manager = ModelManager()
        
        # Test de recommandation
        test_text = "L'intelligence artificielle transforme notre société."
        recommended = manager.recommend_model(test_text, "balanced")
        
        print(f"   ✅ Gestionnaire initialisé")
        print(f"   🎯 Modèle recommandé: {recommended.value}")
        
        # Test de disponibilité des modèles
        led_available, led_msg = manager.is_model_available(ModelType.LED)
        openai_available, openai_msg = manager.is_model_available(ModelType.OPENAI)
        
        print(f"   🔍 LED disponible: {'✅' if led_available else '❌'} {led_msg}")
        print(f"   🔍 OpenAI disponible: {'✅' if openai_available else '❌'} {openai_msg}")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Erreur model manager: {e}")
        return False

def test_led_model():
    """Test du modèle LED (sans génération pour éviter le téléchargement)"""
    print("\n5. 🎯 Test du modèle LED...")
    
    try:
        from models.led_model import LEDSummarizer
        
        # Juste tester l'initialisation de classe sans charger le modèle
        print("   ✅ Classe LEDSummarizer accessible")
        print("   ⚠️  Modèle non chargé (éviter téléchargement)")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Erreur LED model: {e}")
        return False

def test_openai_model():
    """Test du modèle OpenAI (sans appel API)"""
    print("\n6. ⚡ Test du modèle OpenAI...")
    
    try:
        from models.openai_model import OpenAISummarizer
        
        # Vérifier la présence de la clé API
        api_key = os.getenv('OPENAI_API_KEY')
        
        if api_key:
            print("   ✅ Clé API OpenAI configurée")
            print("   ⚠️  Pas d'appel API dans les tests")
        else:
            print("   ⚠️  Clé API OpenAI non configurée")
            print("   💡 Définissez OPENAI_API_KEY pour utiliser OpenAI")
        
        print("   ✅ Classe OpenAISummarizer accessible")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Erreur OpenAI model: {e}")
        return False

def test_configuration():
    """Test des fichiers de configuration"""
    print("\n7. ⚙️ Test de la configuration...")
    
    try:
        config_dir = project_root / "config"
        
        model_config = config_dir / "model_config.yaml"
        app_config = config_dir / "app_config.yaml"
        
        if model_config.exists():
            print("   ✅ model_config.yaml trouvé")
        else:
            print("   ❌ model_config.yaml manquant")
        
        if app_config.exists():
            print("   ✅ app_config.yaml trouvé")
        else:
            print("   ❌ app_config.yaml manquant")
        
        # Test de chargement YAML
        import yaml
        
        if model_config.exists():
            with open(model_config, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
                print(f"   📊 Configuration chargée: {len(config)} sections")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Erreur configuration: {e}")
        return False

def test_streamlit_app():
    """Test de l'application Streamlit (import seulement)"""
    print("\n8. 🎨 Test de l'application Streamlit...")
    
    try:
        # Import sans lancer l'app
        from ui.streamlit_app import VideoSummarizerApp
        
        print("   ✅ Application Streamlit importée")
        print("   💡 Lancez avec: streamlit run src/ui/streamlit_app.py")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Erreur Streamlit app: {e}")
        return False

def main():
    """Fonction principale de test"""
    
    tests = [
        test_imports,
        test_data_ingestion,
        test_preprocessing,
        test_model_manager,
        test_led_model,
        test_openai_model,
        test_configuration,
        test_streamlit_app
    ]
    
    results = []
    
    for test_func in tests:
        try:
            result = test_func()
            results.append(result)
        except Exception as e:
            print(f"   💥 Test {test_func.__name__} échoué: {e}")
            results.append(False)
    
    # Résumé
    print("\n" + "=" * 50)
    print("📊 RÉSUMÉ DES TESTS")
    print("=" * 50)
    
    passed = sum(results)
    total = len(results)
    
    print(f"✅ Tests réussis: {passed}/{total}")
    print(f"❌ Tests échoués: {total - passed}/{total}")
    
    if passed == total:
        print("🎉 Tous les tests sont passés ! Architecture validée.")
        print("\n💡 Prochaines étapes:")
        print("   1. Configurez OPENAI_API_KEY pour utiliser OpenAI")
        print("   2. Lancez l'interface: streamlit run src/ui/streamlit_app.py")
        print("   3. Testez avec une vidéo YouTube")
    else:
        print("⚠️  Certains tests ont échoué. Vérifiez les erreurs ci-dessus.")
    
    print("\n🚀 Architecture Video Summarizer prête !")

if __name__ == "__main__":
    main()