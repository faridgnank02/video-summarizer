#!/usr/bin/env python3
"""
Script de lancement Python pour Video Summarizer
Alternative au script bash pour Windows/Mac/Linux
"""

import os
import sys
import subprocess
from pathlib import Path

def main():
    print("🎥 Video Summarizer - Lancement")
    print("=" * 40)
    
    # Vérifier le répertoire
    if not Path("requirements.txt").exists():
        print("❌ Erreur: Lancez ce script depuis le répertoire du projet")
        sys.exit(1)
    
    # Vérifier l'environnement virtuel
    if "video-summarizer-env" not in sys.executable:
        print("⚠️  Environnement virtuel non activé")
        print("💡 Activez-le avec:")
        print("   source video-summarizer-env/bin/activate  # Linux/Mac")
        print("   video-summarizer-env\\Scripts\\activate     # Windows")
        
        # Essayer d'activer automatiquement (Unix seulement)
        if os.name != 'nt':
            venv_path = Path("video-summarizer-env/bin/python")
            if venv_path.exists():
                print("🔄 Tentative d'activation automatique...")
                os.execv(str(venv_path), [str(venv_path), __file__])
    
    # Vérifier Streamlit
    try:
        import streamlit
        print("✅ Streamlit disponible")
    except ImportError:
        print("❌ Streamlit non installé")
        print("💡 Installez avec: pip install streamlit")
        sys.exit(1)
    
    # Charger les variables d'environnement
    env_file = Path(".env")
    if env_file.exists():
        print("📝 Chargement de .env...")
        with open(env_file) as f:
            for line in f:
                if line.strip() and not line.startswith('#'):
                    key, value = line.strip().split('=', 1)
                    os.environ[key] = value
    
    # Vérifier OpenAI
    if os.getenv('OPENAI_API_KEY'):
        print("✅ Clé API OpenAI configurée")
    else:
        print("⚠️  OPENAI_API_KEY non configurée")
        print("💡 Définissez-la dans .env pour utiliser OpenAI GPT")
    
    # Lancer Streamlit
    print("\n🚀 Lancement de l'interface web...")
    print("📱 Accédez à: http://localhost:8501")
    print("🛑 Arrêtez avec Ctrl+C")
    print()
    
    try:
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", 
            "src/ui/streamlit_app.py"
        ], check=True)
    except KeyboardInterrupt:
        print("\n👋 Application arrêtée")
    except subprocess.CalledProcessError as e:
        print(f"❌ Erreur lors du lancement: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()