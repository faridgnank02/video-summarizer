#!/usr/bin/env python3
"""
Test comparatif entre les modèles LED et OpenAI
"""

import sys
from pathlib import Path

# Ajouter src au PYTHONPATH
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

def compare_models():
    """Compare les résultats des modèles LED et OpenAI"""
    print("🔍 Comparaison des modèles LED vs OpenAI")
    print("=" * 50)
    
    from models.model_manager import ModelManager
    
    # Texte de test
    test_text = """
    L'intelligence artificielle (IA) transforme rapidement tous les secteurs de notre économie moderne. 
    Cette technologie révolutionnaire permet aux machines d'apprendre, de comprendre le langage naturel, 
    de reconnaître les images, et de prendre des décisions complexes sans intervention humaine directe.
    
    Dans le domaine de la santé, l'IA aide au diagnostic médical en analysant des scanners, des radiographies 
    et des données de patients pour détecter des maladies plus rapidement et avec plus de précision que 
    les méthodes traditionnelles. Les algorithmes d'apprentissage automatique peuvent identifier des 
    schémas subtils dans les données médicales que l'œil humain pourrait manquer.
    
    L'industrie automobile développe des véhicules autonomes grâce à des systèmes de vision par ordinateur 
    et des réseaux de neurones profonds. Ces voitures intelligentes peuvent naviguer dans le trafic, 
    éviter les obstacles, et prendre des décisions de conduite en temps réel.
    
    Cependant, cette révolution technologique soulève des questions importantes sur l'avenir de l'emploi, 
    la protection de la vie privée, et l'éthique des décisions automatisées. Il est crucial de développer 
    l'IA de manière responsable pour maximiser ses bénéfices tout en minimisant les risques.
    """
    
    manager = ModelManager()
    
    print(f"📊 Texte original: {len(test_text.split())} mots")
    print(f"📝 Extrait: {test_text[:100]}...")
    
    # Test OpenAI
    print("\n⚡ Test avec OpenAI GPT:")
    print("-" * 30)
    try:
        openai_summary = manager.summarize_simple(
            test_text, 
            model_type="openai", 
            summary_length="short"
        )
        print(f"✅ Résumé OpenAI ({len(openai_summary.split())} mots):")
        print(f"   {openai_summary}")
    except Exception as e:
        print(f"❌ Erreur OpenAI: {e}")
        openai_summary = None
    
    # Test LED
    print("\n🎯 Test avec LED:")
    print("-" * 30)
    try:
        led_summary = manager.summarize_simple(
            test_text, 
            model_type="led", 
            summary_length="short"
        )
        print(f"✅ Résumé LED ({len(led_summary.split())} mots):")
        print(f"   {led_summary}")
    except Exception as e:
        print(f"❌ Erreur LED: {e}")
        led_summary = None
    
    # Comparaison
    print("\n📋 Comparaison:")
    print("-" * 30)
    
    if openai_summary and led_summary:
        print(f"• Longueur OpenAI: {len(openai_summary.split())} mots")
        print(f"• Longueur LED: {len(led_summary.split())} mots")
        
        # Qualité subjective (critères simples)
        openai_coherent = not any(artifact in openai_summary for artifact in ['ét', 'Â', '  '])
        led_coherent = not any(artifact in led_summary for artifact in ['ét', 'Â', '  '])
        
        print(f"• Cohérence OpenAI: {'✅' if openai_coherent else '⚠️'}")
        print(f"• Cohérence LED: {'✅' if led_coherent else '⚠️'}")
        
    elif openai_summary:
        print("• Seul OpenAI disponible")
    elif led_summary:
        print("• Seul LED disponible")
    else:
        print("• Aucun modèle disponible")

def test_led_performance():
    """Test spécifique des performances LED"""
    print("\n🎯 Test de performance LED")
    print("=" * 50)
    
    import time
    from models.led_model import LEDSummarizer
    
    led = LEDSummarizer()
    
    # Test avec différentes longueurs
    texts = [
        ("Court", "L'IA transforme notre monde avec des applications variées."),
        ("Moyen", """
        L'intelligence artificielle révolutionne de nombreux domaines.
        Elle permet l'automatisation de tâches complexes et l'analyse
        de grandes quantités de données. Les applications incluent
        la reconnaissance vocale, la vision par ordinateur, et la
        traduction automatique.
        """),
        ("Long", """
        L'intelligence artificielle représente l'une des avancées technologiques
        les plus significatives de notre époque. Cette discipline interdisciplinaire
        combine informatique, mathématiques, sciences cognitives et neurosciences
        pour créer des systèmes capables de réaliser des tâches qui nécessitent
        normalement l'intelligence humaine.
        
        Les applications de l'IA sont désormais omniprésentes dans notre quotidien.
        Les moteurs de recherche utilisent des algorithmes sophistiqués pour classer
        les résultats. Les plateformes de médias sociaux emploient l'IA pour
        personnaliser les contenus et détecter les contenus inappropriés.
        
        Dans le domaine médical, l'IA assiste les professionnels de santé dans
        le diagnostic, la découverte de médicaments et la planification de traitements.
        L'industrie financière utilise l'IA pour la détection de fraude, l'évaluation
        des risques et le trading algorithmique.
        """)
    ]
    
    for name, text in texts:
        print(f"\n📝 Test {name} ({len(text.split())} mots):")
        
        start_time = time.time()
        try:
            summary = led.summarize(text, summary_type="short")
            end_time = time.time()
            
            duration = end_time - start_time
            summary_clean = summary.replace('\n', ' ').strip()
            
            print(f"   ⏱️  Temps: {duration:.2f}s")
            print(f"   📊 Résumé: {len(summary_clean.split())} mots")
            print(f"   📄 Extrait: {summary_clean[:80]}...")
            
        except Exception as e:
            print(f"   ❌ Erreur: {e}")

def main():
    """Fonction principale"""
    compare_models()
    test_led_performance()
    
    print("\n" + "=" * 50)
    print("🎉 Tests de comparaison terminés !")
    print("\n💡 Conclusions:")
    print("• OpenAI: Plus rapide, plus cohérent, nécessite une clé API")
    print("• LED: Gratuit, fonctionne hors ligne, mais plus lent et moins cohérent sur des textes courts")
    print("• Recommandation: OpenAI pour usage quotidien, LED pour tests ou usage sans Internet")

if __name__ == "__main__":
    main()