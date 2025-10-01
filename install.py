#!/usr/bin/env python3
"""
Script d'installation automatique pour Video Summarizer
Configure l'environnement et installe toutes les dépendances
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path

def run_command(command, description, check=True):
    """Exécute une commande avec gestion d'erreur"""
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=check, capture_output=True, text=True)
        if result.stdout:
            print(f"   {result.stdout.strip()}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Erreur: {e}")
        if e.stderr:
            print(f"   {e.stderr.strip()}")
        return False

def check_python():
    """Vérifie la version de Python"""
    print("🐍 Vérification de Python...")
    
    if sys.version_info < (3, 8):
        print("❌ Python 3.8+ requis")
        print(f"   Version actuelle: {sys.version}")
        return False
    
    print(f"✅ Python {sys.version.split()[0]} OK")
    return True

def create_venv():
    """Crée l'environnement virtuel"""
    venv_path = Path("video-summarizer-env")
    
    if venv_path.exists():
        print("✅ Environnement virtuel déjà existant")
        return True
    
    return run_command(
        "python3 -m venv video-summarizer-env",
        "Création de l'environnement virtuel"
    )

def install_requirements():
    """Installe les dépendances"""
    pip_cmd = "video-summarizer-env/bin/pip"
    if os.name == 'nt':  # Windows
        pip_cmd = "video-summarizer-env\\Scripts\\pip"

    # Lire requirements.txt et exclure sqlite3
    with open("requirements.txt", "r") as f:
        requirements = [line.strip() for line in f if line.strip() and not line.startswith("sqlite3")]

    # Créer un fichier temporaire sans sqlite3
    temp_requirements = "temp_requirements.txt"
    with open(temp_requirements, "w") as f:
        f.write("\n".join(requirements))

    commands = [
        f"{pip_cmd} install --upgrade pip",
        f"{pip_cmd} install -r {temp_requirements}"
    ]

    for cmd in commands:
        if not run_command(cmd, f"Installation via pip"):
            return False

    # Supprimer le fichier temporaire
    if os.path.exists(temp_requirements):
        os.remove(temp_requirements)

    return True

def install_spacy_models():
    """Installe les modèles spaCy pour NER"""
    python_cmd = "video-summarizer-env/bin/python"
    if os.name == 'nt':  # Windows
        python_cmd = "video-summarizer-env\\Scripts\\python"
    
    models = ["fr_core_news_sm", "en_core_web_sm"]
    
    for model in models:
        run_command(
            f"{python_cmd} -m spacy download {model}",
            f"Téléchargement modèle spaCy {model}",
            check=False  # Continuer même si échec
        )

def setup_env_file():
    """Configure le fichier .env"""
    env_path = Path(".env")
    env_example_path = Path(".env.example")
    
    if env_path.exists():
        print("✅ Fichier .env déjà existant")
        return True
    
    if env_example_path.exists():
        shutil.copy(env_example_path, env_path)
        print("✅ Fichier .env créé depuis .env.example")
        print("💡 Éditez .env pour ajouter votre clé API OpenAI")
    else:
        # Créer un .env minimal
        with open(env_path, 'w') as f:
            f.write("# Configuration Video Summarizer\n")
            f.write("# OPENAI_API_KEY=sk-your-key-here\n")
            f.write("DEBUG=True\n")
            f.write("LOG_LEVEL=INFO\n")
        print("✅ Fichier .env minimal créé")
    
    return True

def test_installation():
    """Test rapide de l'installation"""
    print("\n🧪 Test de l'installation...")
    
    python_cmd = "video-summarizer-env/bin/python"
    if os.name == 'nt':  # Windows
        python_cmd = "video-summarizer-env\\Scripts\\python"
    
    # Test d'importation basique
    test_script = '''
import sys
sys.path.insert(0, "src")

try:
    from data.ingestion import DataIngestion
    from data.preprocessing import TextPreprocessor
    print("✅ Modules principaux importés")
except ImportError as e:
    print(f"❌ Erreur importation: {e}")
    sys.exit(1)

try:
    ingestion = DataIngestion()
    preprocessor = TextPreprocessor()
    print("✅ Classes instanciées")
except Exception as e:
    print(f"❌ Erreur instanciation: {e}")
    sys.exit(1)

print("🎉 Installation validée!")
'''
    
    with open("temp_test.py", "w") as f:
        f.write(test_script)
    
    success = run_command(
        f"{python_cmd} temp_test.py",
        "Test des importations"
    )
    
    # Nettoyer
    if Path("temp_test.py").exists():
        os.remove("temp_test.py")
    
    return success

def main():
    """Fonction principale d'installation"""
    print("🎥 Installation de Video Summarizer")
    print("=" * 40)
    
    # Vérifications préliminaires
    if not check_python():
        sys.exit(1)
    
    # Étapes d'installation
    steps = [
        (create_venv, "Environnement virtuel"),
        (install_requirements, "Dépendances Python"),
        (install_spacy_models, "Modèles spaCy"),
        (setup_env_file, "Configuration"),
        (test_installation, "Tests"),
    ]
    
    for step_func, step_name in steps:
        print(f"\n📦 {step_name}")
        if not step_func():
            print(f"❌ Échec: {step_name}")
            print("🛑 Installation interrompue")
            sys.exit(1)
    
    # Instructions finales
    print("\n" + "=" * 40)
    print("🎉 Installation terminée avec succès!")
    print("\n📋 Prochaines étapes:")
    print("1. Configurez votre clé API OpenAI dans .env")
    print("2. Lancez l'application:")
    
    if os.name == 'nt':  # Windows
        print("   video-summarizer-env\\Scripts\\activate")
        print("   python launch.py")
    else:  # Unix/Mac
        print("   source video-summarizer-env/bin/activate")
        print("   python launch.py")
    
    print("\n💡 Ou testez directement:")
    print("   python test_functionality.py")
    
    print("\n📚 Documentation complète dans README.md")

if __name__ == "__main__":
    main()