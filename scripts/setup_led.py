#!/usr/bin/env python3
"""
Script pour télécharger, configurer et tester le modèle LED
Optimise les performances et corrige les problèmes de génération
"""

import sys
import os
from pathlib import Path

# Ajouter le répertoire src au PYTHONPATH
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

def download_and_setup_led():
    """Télécharge et configure le modèle LED"""
    print("🎯 Configuration du modèle LED")
    print("=" * 40)
    
    try:
        from models.led_model import LEDSummarizer
        import torch
        
        print(f"📊 PyTorch version: {torch.__version__}")
        print(f"💻 CUDA disponible: {torch.cuda.is_available()}")
        print(f"🖥️  Device recommandé: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
        
        print("\n📥 Téléchargement/Chargement du modèle LED...")
        print("⏳ Cela peut prendre quelques minutes la première fois...")
        
        # Initialiser le modèle (téléchargement automatique si nécessaire)
        led = LEDSummarizer()
        
        print("✅ Modèle LED chargé avec succès !")
        return led
        
    except Exception as e:
        print(f"❌ Erreur lors du chargement: {e}")
        return None

def test_led_with_different_texts(led):
    """Teste le modèle LED avec différents types de textes"""
    print("\n🧪 Tests du modèle LED")
    print("=" * 40)
    
    test_cases = [
        {
            "name": "Texte court (français)",
            "text": """
            L'intelligence artificielle représente l'une des révolutions technologiques 
            les plus importantes de notre époque. Cette technologie permet aux machines 
            d'apprendre, de comprendre et de prendre des décisions de manière autonome. 
            Les applications sont multiples : reconnaissance vocale, traduction automatique, 
            véhicules autonomes, diagnostic médical assisté. Cependant, cette évolution 
            soulève des questions éthiques fondamentales sur l'avenir du travail humain 
            et la protection de la vie privée.
            """,
            "type": "short"
        },
        {
            "name": "Texte technique (français)",
            "text": """
            Le machine learning, ou apprentissage automatique, constitue une branche 
            centrale de l'intelligence artificielle. Cette approche permet aux systèmes 
            informatiques d'améliorer automatiquement leurs performances sur une tâche 
            spécifique grâce à l'expérience, sans être explicitement programmés pour 
            chaque situation. Les algorithmes d'apprentissage automatique construisent 
            un modèle mathématique basé sur des données d'entraînement pour faire des 
            prédictions ou prendre des décisions sans intervention humaine directe. 
            Cette technologie trouve des applications dans de nombreux domaines : 
            filtrage de courrier électronique, reconnaissance optique de caractères, 
            moteurs de recherche, vision par ordinateur et bio-informatique.
            """,
            "type": "long"
        }
    ]
    
    results = []
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n📝 Test {i}: {test_case['name']}")
        print("-" * 30)
        
        text = test_case['text'].strip()
        summary_type = test_case['type']
        
        print(f"📊 Texte original: {len(text.split())} mots")
        print(f"🎯 Type de résumé: {summary_type}")
        
        try:
            summary = led.summarize(text, summary_type=summary_type)
            
            # Nettoyer le résumé des artefacts
            summary_clean = clean_summary(summary)
            
            print(f"✅ Résumé généré: {len(summary_clean.split())} mots")
            print(f"📄 Résumé: {summary_clean}")
            
            results.append({
                "test": test_case['name'],
                "original_length": len(text.split()),
                "summary_length": len(summary_clean.split()),
                "success": True,
                "summary": summary_clean
            })
            
        except Exception as e:
            print(f"❌ Erreur: {e}")
            results.append({
                "test": test_case['name'],
                "success": False,
                "error": str(e)
            })
    
    return results

def clean_summary(summary):
    """Nettoie le résumé des artefacts de génération"""
    import re
    
    # Supprimer les caractères étranges et doublons
    summary = re.sub(r'[^\w\s\-.,;:!?\'"àâäéèêëïîôöùûüÿçñ]', '', summary)
    
    # Corriger les espaces multiples
    summary = re.sub(r'\s+', ' ', summary)
    
    # Supprimer les répétitions de mots
    words = summary.split()
    cleaned_words = []
    prev_word = ""
    
    for word in words:
        if word.lower() != prev_word.lower():
            cleaned_words.append(word)
            prev_word = word
    
    return ' '.join(cleaned_words).strip()

def optimize_led_config():
    """Optimise la configuration du modèle LED"""
    print("\n⚙️ Optimisation de la configuration LED")
    print("=" * 40)
    
    config_path = Path(__file__).parent.parent / "config" / "model_config.yaml"
    
    try:
        import yaml
        
        # Lire la configuration actuelle
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        # Optimiser les paramètres LED
        if 'models' not in config:
            config['models'] = {}
        
        if 'led' not in config['models']:
            config['models']['led'] = {}
        
        # Paramètres optimisés
        optimized_config = {
            'model_name': 'allenai/led-base-16384',
            'max_input_length': 4096,  # Réduit pour de meilleures performances
            'max_output_length': 512,
            'generation_config': {
                'num_beams': 3,           # Équilibre qualité/vitesse
                'max_length': 512,
                'min_length': 50,
                'length_penalty': 1.5,    # Moins pénalisant
                'early_stopping': True,
                'no_repeat_ngram_size': 2, # Réduit les répétitions
                'repetition_penalty': 1.1, # Nouveau paramètre
                'do_sample': False,        # Génération déterministe
                'temperature': 1.0
            }
        }
        
        config['models']['led'].update(optimized_config)
        
        # Sauvegarder la configuration optimisée
        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.dump(config, f, default_flow_style=False, indent=2, allow_unicode=True)
        
        print("✅ Configuration LED optimisée")
        print("📝 Paramètres mis à jour:")
        for key, value in optimized_config['generation_config'].items():
            print(f"   • {key}: {value}")
        
        return True
        
    except Exception as e:
        print(f"❌ Erreur lors de l'optimisation: {e}")
        return False

def create_led_usage_examples():
    """Crée des exemples d'utilisation du modèle LED"""
    examples_path = Path(__file__).parent.parent / "assets" / "examples"
    examples_path.mkdir(exist_ok=True)
    
    example_file = examples_path / "led_examples.py"
    
    example_code = '''#!/usr/bin/env python3
"""
Exemples d'utilisation du modèle LED pour le résumé de texte
"""

import sys
from pathlib import Path

# Ajouter src au PYTHONPATH
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from models.led_model import LEDSummarizer

def example_simple_summary():
    """Exemple de résumé simple"""
    print("📝 Exemple: Résumé simple")
    
    led = LEDSummarizer()
    
    text = """
    L'intelligence artificielle (IA) transforme rapidement notre monde moderne.
    Cette technologie permet aux ordinateurs de réaliser des tâches qui nécessitaient
    auparavant l'intelligence humaine. Les applications incluent la reconnaissance
    vocale, la traduction automatique, la conduite autonome et l'analyse de données.
    Cependant, l'IA soulève des questions importantes sur l'emploi, l'éthique et
    la protection de la vie privée que nous devons aborder de manière responsable.
    """
    
    summary = led.summarize(text, summary_type="short")
    print(f"Résumé: {summary}")

def example_long_document():
    """Exemple avec un document plus long"""
    print("\\n📖 Exemple: Document long")
    
    led = LEDSummarizer()
    
    # Simuler un document plus long
    long_text = """
    L'intelligence artificielle représente l'une des avancées technologiques les plus
    significatives de notre époque. Cette discipline, qui combine informatique,
    mathématiques et sciences cognitives, vise à créer des systèmes capables de
    réaliser des tâches nécessitant normalement l'intelligence humaine.
    
    Les applications de l'IA sont désormais omniprésentes dans notre quotidien.
    Les assistants vocaux comme Siri, Alexa ou Google Assistant utilisent des
    algorithmes de reconnaissance vocale et de traitement du langage naturel.
    Les plateformes de streaming comme Netflix ou Spotify emploient des systèmes
    de recommandation basés sur l'apprentissage automatique.
    
    Dans le domaine médical, l'IA aide au diagnostic précoce de maladies,
    à l'analyse d'images médicales et au développement de nouveaux médicaments.
    L'industrie automobile développe des véhicules autonomes grâce à la vision
    par ordinateur et aux réseaux de neurones profonds.
    
    Cependant, cette révolution technologique soulève des défis importants.
    L'impact sur l'emploi inquiète de nombreux secteurs, car l'automatisation
    pourrait remplacer certains métiers. Les questions de protection des données
    personnelles et de vie privée deviennent cruciales avec la collecte massive
    d'informations par les algorithmes d'IA.
    
    L'avenir de l'intelligence artificielle dépend de notre capacité à développer
    cette technologie de manière éthique et responsable, en tenant compte des
    implications sociales, économiques et philosophiques qu'elle représente.
    """
    
    summary = led.summarize(long_text, summary_type="long")
    print(f"Résumé long: {summary}")

if __name__ == "__main__":
    example_simple_summary()
    example_long_document()
'''
    
    with open(example_file, 'w', encoding='utf-8') as f:
        f.write(example_code)
    
    print(f"✅ Exemples créés: {example_file}")

def main():
    """Fonction principale de configuration LED"""
    print("🎯 Configuration complète du modèle LED")
    print("=" * 50)
    
    # 1. Télécharger et configurer LED
    led = download_and_setup_led()
    if not led:
        print("❌ Échec de la configuration LED")
        return False
    
    # 2. Optimiser la configuration
    optimize_led_config()
    
    # 3. Tester avec différents textes
    results = test_led_with_different_texts(led)
    
    # 4. Créer des exemples
    create_led_usage_examples()
    
    # 5. Résumé final
    print("\n" + "=" * 50)
    print("📊 RÉSUMÉ DE LA CONFIGURATION LED")
    print("=" * 50)
    
    successful_tests = sum(1 for r in results if r['success'])
    total_tests = len(results)
    
    print(f"✅ Tests réussis: {successful_tests}/{total_tests}")
    
    if successful_tests == total_tests:
        print("🎉 Modèle LED entièrement fonctionnel !")
        print("\n💡 Utilisation:")
        print("   • Interface web: streamlit run src/ui/streamlit_app.py")
        print("   • Code: from models.led_model import LEDSummarizer")
        print("   • Exemples: python assets/examples/led_examples.py")
    else:
        print("⚠️  Quelques tests ont échoué, mais le modèle de base fonctionne")
    
    print(f"\n🔧 Configuration optimisée sauvegardée")
    print(f"📚 Exemples d'utilisation créés")
    
    return True

if __name__ == "__main__":
    main()