#!/usr/bin/env python3
"""
Script de lancement pour l'API Video Summarizer
Démarre l'API avec monitoring et configuration optimisée
"""

import os
import sys
import time
import logging
import argparse
from pathlib import Path

# Ajouter le répertoire src au PYTHONPATH
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

def setup_logging(log_level: str = "INFO"):
    """Configure le logging pour l'API"""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('logs/api.log', mode='a')
        ]
    )

def check_dependencies():
    """Vérifie les dépendances critiques"""
    print("🔍 Vérification des dépendances...")
    
    required_modules = [
        'fastapi', 'uvicorn', 'streamlit', 'torch', 'transformers',
        'psutil', 'sqlite3', 'yaml', 'rouge', 'spacy'
    ]
    
    missing = []
    
    for module in required_modules:
        try:
            __import__(module)
            print(f"   ✅ {module}")
        except ImportError:
            missing.append(module)
            print(f"   ❌ {module}")
    
    if missing:
        print(f"\n⚠️  Modules manquants: {', '.join(missing)}")
        print("💡 Installez avec: pip install -r requirements.txt")
        return False
    
    print("✅ Toutes les dépendances sont disponibles")
    return True

def create_directories():
    """Crée les répertoires nécessaires"""
    directories = [
        "logs",
        "data/cache",
        "metrics",
        "exports"
    ]
    
    print("📁 Création des répertoires...")
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"   📂 {directory}")

def check_environment():
    """Vérifie la configuration de l'environnement"""
    print("🔧 Vérification de l'environnement...")
    
    # Variables d'environnement critiques
    env_vars = {
        'OPENAI_API_KEY': 'Clé API OpenAI (optionnel)',
        'DEBUG': 'Mode debug (optionnel)',
        'LOG_LEVEL': 'Niveau de log (optionnel)'
    }
    
    for var, description in env_vars.items():
        value = os.getenv(var)
        if value:
            if var == 'OPENAI_API_KEY':
                masked_value = value[:8] + '*' * (len(value) - 8)
                print(f"   ✅ {var}: {masked_value}")
            else:
                print(f"   ✅ {var}: {value}")
        else:
            print(f"   ⚠️  {var}: Non définie ({description})")

def start_api_server(host: str = "0.0.0.0", 
                    port: int = 8000,
                    workers: int = 1,
                    log_level: str = "info",
                    reload: bool = False):
    """Démarre le serveur API"""
    
    print(f"🚀 Démarrage du serveur API sur {host}:{port}")
    print(f"📊 Workers: {workers}")
    print(f"🔄 Reload: {reload}")
    print(f"📝 Log level: {log_level}")
    print()
    
    try:
        import uvicorn
        from api.summarization import app
        
        # Configuration du serveur
        config = uvicorn.Config(
            app=app,
            host=host,
            port=port,
            workers=workers,
            log_level=log_level,
            reload=reload,
            access_log=True
        )
        
        server = uvicorn.Server(config)
        
        print("✅ API Video Summarizer démarrée")
        print(f"📍 Documentation: http://{host}:{port}/docs")
        print(f"🔍 Redoc: http://{host}:{port}/redoc")
        print(f"❤️  Health check: http://{host}:{port}/health")
        print()
        print("🛑 Arrêt avec Ctrl+C")
        print("=" * 50)
        
        server.run()
        
    except KeyboardInterrupt:
        print("\n👋 Arrêt du serveur API")
    except Exception as e:
        print(f"❌ Erreur lors du démarrage: {e}")
        return False
    
    return True

def start_monitoring_dashboard(port: int = 8501):
    """Démarre le dashboard de monitoring"""
    
    print(f"📊 Démarrage du dashboard de monitoring sur le port {port}")
    
    try:
        import subprocess
        import sys
        
        # Commande pour démarrer Streamlit
        cmd = [
            sys.executable, "-m", "streamlit", "run",
            "src/ui/monitoring_dashboard.py",
            "--server.port", str(port),
            "--server.address", "0.0.0.0"
        ]
        
        subprocess.run(cmd, check=True)
        
    except KeyboardInterrupt:
        print("\n👋 Arrêt du dashboard")
    except Exception as e:
        print(f"❌ Erreur dashboard: {e}")
        return False
    
    return True

def main():
    """Fonction principale"""
    parser = argparse.ArgumentParser(description="Video Summarizer API Server")
    
    parser.add_argument("--mode", choices=["api", "dashboard", "both"], 
                       default="api", help="Mode de démarrage")
    parser.add_argument("--host", default="0.0.0.0", help="Adresse d'écoute")
    parser.add_argument("--port", type=int, default=8000, help="Port API")
    parser.add_argument("--dashboard-port", type=int, default=8501, help="Port dashboard")
    parser.add_argument("--workers", type=int, default=1, help="Nombre de workers")
    parser.add_argument("--log-level", default="info", help="Niveau de log")
    parser.add_argument("--reload", action="store_true", help="Mode reload (dev)")
    parser.add_argument("--skip-checks", action="store_true", help="Ignorer les vérifications")
    
    args = parser.parse_args()
    
    print("🎥 Video Summarizer API - Enterprise Edition")
    print("=" * 50)
    
    # Vérifications préliminaires
    if not args.skip_checks:
        if not check_dependencies():
            sys.exit(1)
        
        create_directories()
        check_environment()
        print()
    
    # Configuration du logging
    setup_logging(args.log_level)
    
    # Démarrage selon le mode
    if args.mode == "api":
        success = start_api_server(
            host=args.host,
            port=args.port,
            workers=args.workers,
            log_level=args.log_level,
            reload=args.reload
        )
        
    elif args.mode == "dashboard":
        success = start_monitoring_dashboard(args.dashboard_port)
        
    elif args.mode == "both":
        print("🚀 Démarrage en mode complet (API + Dashboard)")
        print("💡 Utilisez des terminaux séparés en production")
        print()
        
        # Pour le développement, démarrer l'API en premier
        success = start_api_server(
            host=args.host,
            port=args.port,
            workers=args.workers,
            log_level=args.log_level,
            reload=args.reload
        )
    
    if not success:
        sys.exit(1)

if __name__ == "__main__":
    main()