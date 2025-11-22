#!/usr/bin/env python3
"""
Script de test rapide pour l'intégration Ollama
Vérifie que tout fonctionne correctement avant utilisation
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import requests
import time
from colorama import init, Fore, Style

# Initialiser colorama pour les couleurs
init()

def print_section(title):
    """Affiche un titre de section"""
    print(f"\n{Fore.CYAN}{'='*60}{Style.RESET_ALL}")
    print(f"{Fore.CYAN}{title}{Style.RESET_ALL}")
    print(f"{Fore.CYAN}{'='*60}{Style.RESET_ALL}\n")

def print_success(message):
    """Affiche un message de succès"""
    print(f"{Fore.GREEN}✓ {message}{Style.RESET_ALL}")

def print_error(message):
    """Affiche un message d'erreur"""
    print(f"{Fore.RED}✗ {message}{Style.RESET_ALL}")

def print_info(message):
    """Affiche un message d'information"""
    print(f"{Fore.YELLOW}ℹ {message}{Style.RESET_ALL}")

def check_ollama_server():
    """Vérifie que le serveur Ollama est accessible"""
    print_section("1. Vérification du serveur Ollama")
    
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code == 200:
            print_success("Serveur Ollama accessible sur http://localhost:11434")
            return True
        else:
            print_error(f"Serveur répond mais avec le code {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print_error("Serveur Ollama non accessible")
        print_info("Démarrez Ollama avec : ollama serve")
        return False
    except Exception as e:
        print_error(f"Erreur inattendue : {e}")
        return False

def list_available_models():
    """Liste les modèles disponibles sur Ollama"""
    print_section("2. Modèles disponibles")
    
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code == 200:
            data = response.json()
            models = data.get('models', [])
            
            if not models:
                print_error("Aucun modèle installé")
                print_info("Installez un modèle avec : ollama pull mistral:7b")
                return []
            
            print_success(f"Trouvé {len(models)} modèle(s) installé(s) :")
            model_names = []
            for model in models:
                name = model.get('name', 'unknown')
                size = model.get('size', 0) / (1024**3)  # Convert to GB
                print(f"  • {name} ({size:.2f} GB)")
                model_names.append(name)
            
            return model_names
        else:
            print_error("Impossible de lister les modèles")
            return []
    except Exception as e:
        print_error(f"Erreur lors de la récupération des modèles : {e}")
        return []

def test_ollama_model():
    """Teste la génération avec OllamaSummarizer"""
    print_section("3. Test de génération avec OllamaSummarizer")
    
    try:
        from src.models.ollama_model import OllamaSummarizer
        
        # Créer l'instance avec le modèle gemma3:1b
        print_info("Initialisation de OllamaSummarizer...")
        summarizer = OllamaSummarizer(model_name="gemma3:1b")

        # Texte de test
        test_text = """
        L'intelligence artificielle (IA) transforme radicalement notre monde. 
        Des assistants vocaux aux voitures autonomes, en passant par les 
        recommandations personnalisées, l'IA est partout. Les modèles de 
        langage comme GPT-4 et Mistral peuvent maintenant comprendre et 
        générer du texte de manière impressionnante. Cette révolution 
        technologique soulève aussi des questions éthiques importantes 
        concernant la vie privée, l'emploi et le contrôle de ces systèmes.
        """
        
        # Générer un résumé court
        print_info("Génération d'un résumé court...")
        start_time = time.time()
        summary = summarizer.summarize(test_text, summary_type="short")
        duration = time.time() - start_time
        
        print_success(f"Résumé généré en {duration:.2f} secondes")
        print(f"\n{Fore.MAGENTA}Résumé :{Style.RESET_ALL}")
        print(f"{summary}\n")
        
        # Obtenir les statistiques
        stats = summarizer.get_usage_stats()
        print_info(f"Statistiques : {stats['total_requests']} requête(s), "
                   f"{stats['total_tokens_generated']} token(s) générés")
        
        return True
    except ImportError as e:
        print_error(f"Impossible d'importer OllamaSummarizer : {e}")
        print_info("Vérifiez que le fichier src/models/ollama_model.py existe")
        return False
    except Exception as e:
        print_error(f"Erreur lors du test : {e}")
        import traceback
        traceback.print_exc()
        return False

def test_model_manager():
    """Teste l'intégration avec ModelManager"""
    print_section("4. Test d'intégration avec ModelManager")
    
    try:
        from src.models.model_manager import ModelManager, ModelType
        
        # Créer l'instance
        print_info("Initialisation de ModelManager...")
        manager = ModelManager()
        
        # Vérifier disponibilité
        is_available = manager.is_model_available(ModelType.OLLAMA)
        if is_available:
            print_success("Ollama est disponible dans ModelManager")
        else:
            print_error("Ollama n'est pas disponible dans ModelManager")
            return False
        
        # Texte de test
        test_text = """
        Python est un langage de programmation populaire, connu pour sa 
        simplicité et sa lisibilité. Il est largement utilisé en data science, 
        développement web, automatisation et intelligence artificielle.
        """
        
        # Générer un résumé avec Ollama
        print_info("Génération avec ModelManager (type=OLLAMA)...")
        start_time = time.time()
        summary = manager.summarize_simple(test_text, model_type="ollama")
        duration = time.time() - start_time
        
        print_success(f"Résumé généré en {duration:.2f} secondes")
        print(f"\n{Fore.MAGENTA}Résumé :{Style.RESET_ALL}")
        print(f"{summary}\n")
        
        # Obtenir les statistiques
        stats = manager.get_stats()
        print_info(f"Statistiques : {stats['ollama_requests']} requête(s) Ollama")
        
        return True
    except ImportError as e:
        print_error(f"Impossible d'importer ModelManager : {e}")
        return False
    except Exception as e:
        print_error(f"Erreur lors du test : {e}")
        import traceback
        traceback.print_exc()
        return False

def test_recommendation_system():
    """Teste le système de recommandation de modèles"""
    print_section("5. Test du système de recommandation")
    
    try:
        from src.models.model_manager import ModelManager
        
        manager = ModelManager()
        
        # Tester différentes priorités avec un texte de test
        priorities = ['cost', 'speed', 'quality', 'balanced']
        test_text = "Python est un langage de programmation populaire utilisé en data science."
        
        for priority in priorities:
            recommendation = manager.recommend_model(
                text=test_text,
                priority=priority
            )
            
            print(f"{Fore.CYAN}{priority.upper()}:{Style.RESET_ALL} {recommendation.value}")
        
        print_success("Système de recommandation fonctionnel")
        return True
    except Exception as e:
        print_error(f"Erreur lors du test de recommandation : {e}")
        return False

def print_summary(results):
    """Affiche un résumé des tests"""
    print_section("Résumé des tests")
    
    total = len(results)
    passed = sum(results.values())
    failed = total - passed
    
    print(f"Total : {total} tests")
    print_success(f"Réussis : {passed}")
    if failed > 0:
        print_error(f"Échoués : {failed}")
    
    print("\nDétails :")
    for test_name, result in results.items():
        status = "✓" if result else "✗"
        color = Fore.GREEN if result else Fore.RED
        print(f"  {color}{status} {test_name}{Style.RESET_ALL}")
    
    if passed == total:
        print(f"\n{Fore.GREEN}🎉 Tous les tests sont passés ! Ollama est prêt à l'emploi.{Style.RESET_ALL}")
    else:
        print(f"\n{Fore.YELLOW}⚠️  Certains tests ont échoué. Vérifiez les erreurs ci-dessus.{Style.RESET_ALL}")

def main():
    """Fonction principale"""
    print(f"\n{Fore.CYAN}{'='*60}{Style.RESET_ALL}")
    print(f"{Fore.CYAN}     Test d'intégration Ollama - Video Summarizer{Style.RESET_ALL}")
    print(f"{Fore.CYAN}{'='*60}{Style.RESET_ALL}")
    
    results = {}
    
    # Test 1 : Serveur Ollama
    results['Serveur Ollama'] = check_ollama_server()
    if not results['Serveur Ollama']:
        print_error("\n❌ Le serveur Ollama doit être démarré pour continuer")
        print_info("Commande : ollama serve")
        print_summary(results)
        return
    
    # Test 2 : Modèles disponibles
    models = list_available_models()
    results['Modèles installés'] = len(models) > 0
    if not results['Modèles installés']:
        print_error("\n❌ Aucun modèle installé")
        print_info("Commande : ollama pull mistral:7b")
        print_summary(results)
        return
    
    # Test 3 : OllamaSummarizer
    results['OllamaSummarizer'] = test_ollama_model()
    
    # Test 4 : ModelManager
    results['ModelManager'] = test_model_manager()
    
    # Test 5 : Système de recommandation
    results['Recommandations'] = test_recommendation_system()
    
    # Afficher le résumé
    print_summary(results)
    
    # Instructions suivantes
    if all(results.values()):
        print(f"\n{Fore.CYAN}Prochaines étapes :{Style.RESET_ALL}")
        print("  1. Lancer l'interface web : python scripts/launch.py")
        print("  2. Tester dans Streamlit avec Ollama comme modèle")
        print("  3. Comparer les performances avec LED et OpenAI")
        print("  4. Consulter la documentation : docs/OLLAMA_INTEGRATION.md")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print(f"\n\n{Fore.YELLOW}Test interrompu par l'utilisateur{Style.RESET_ALL}")
        sys.exit(0)
    except Exception as e:
        print(f"\n{Fore.RED}Erreur critique : {e}{Style.RESET_ALL}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
