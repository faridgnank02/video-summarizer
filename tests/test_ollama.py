#!/usr/bin/env python3
"""
Tests pour l'intégration Ollama
Vérifie la disponibilité, les performances et la qualité
"""

import sys
import time
from pathlib import Path

# Ajouter le répertoire src au PYTHONPATH
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from models.ollama_model import OllamaSummarizer, OllamaConnectionError
from models.model_manager import ModelManager, ModelType
from evaluation.evaluator import SummaryEvaluator


def test_ollama_connection():
    """Test 1: Vérifier la connexion à Ollama"""
    print("\n" + "="*60)
    print("TEST 1: Connexion à Ollama")
    print("="*60)
    
    try:
        summarizer = OllamaSummarizer()
        print("✅ Connexion à Ollama réussie")
        
        # Afficher les modèles disponibles
        models = summarizer._get_available_models()
        print(f"\n📦 Modèles disponibles ({len(models)}):")
        for model in models:
            print(f"   - {model}")
        
        if not models:
            print("\n⚠️  Aucun modèle trouvé. Téléchargez un modèle:")
            print("   ollama pull mistral:7b")
            print("   ollama pull llama3.1:8b")
            print("   ollama pull gemma2:9b")
            return False
        
        return True
        
    except OllamaConnectionError as e:
        print(f"❌ Erreur de connexion: {e}")
        print("\n💡 Démarrez Ollama avec: ollama serve")
        return False


def test_ollama_summarization():
    """Test 2: Génération de résumé avec Ollama"""
    print("\n" + "="*60)
    print("TEST 2: Génération de résumé")
    print("="*60)
    
    try:
        summarizer = OllamaSummarizer(model_name="mistral:7b")
        
        # Texte de test
        test_text = """
        L'intelligence artificielle (IA) transforme profondément notre société moderne.
        Cette technologie révolutionnaire permet aux machines d'apprendre, de raisonner 
        et de prendre des décisions de manière autonome. Les applications sont nombreuses : 
        reconnaissance vocale, vision par ordinateur, traduction automatique, assistants 
        virtuels, voitures autonomes, et bien d'autres domaines.
        
        Cependant, cette révolution technologique soulève également des questions éthiques 
        importantes concernant l'emploi, la vie privée, les biais algorithmiques et 
        l'autonomie humaine. Il est crucial de développer l'IA de manière responsable 
        pour maximiser ses bénéfices tout en minimisant les risques potentiels.
        
        L'avenir de l'IA promet des avancées encore plus spectaculaires, avec le 
        développement de l'intelligence artificielle générale (AGI) qui pourrait égaler 
        ou surpasser les capacités cognitives humaines dans tous les domaines.
        """
        
        print(f"\n📝 Texte original ({len(test_text.split())} mots)")
        print(f"   {test_text[:150]}...")
        
        # Test résumé court
        print("\n🔄 Génération résumé COURT...")
        start_time = time.time()
        short_summary = summarizer.summarize(test_text, summary_type="short")
        short_time = time.time() - start_time
        
        print(f"✅ Résumé court généré en {short_time:.2f}s ({len(short_summary.split())} mots)")
        print(f"   {short_summary}")
        
        # Test résumé long
        print("\n🔄 Génération résumé LONG...")
        start_time = time.time()
        long_summary = summarizer.summarize(test_text, summary_type="long")
        long_time = time.time() - start_time
        
        print(f"✅ Résumé long généré en {long_time:.2f}s ({len(long_summary.split())} mots)")
        print(f"   {long_summary}")
        
        # Statistiques
        stats = summarizer.get_usage_stats()
        print(f"\n📊 Statistiques:")
        print(f"   Requêtes totales: {stats['total_requests']}")
        print(f"   Succès: {stats['successful_requests']}")
        print(f"   Taux de réussite: {stats['success_rate']:.1%}")
        print(f"   Temps moyen: {stats['avg_processing_time']:.2f}s")
        
        return short_summary, long_summary
        
    except Exception as e:
        print(f"❌ Erreur: {e}")
        return None, None


def test_model_manager_integration():
    """Test 3: Intégration avec ModelManager"""
    print("\n" + "="*60)
    print("TEST 3: Intégration ModelManager")
    print("="*60)
    
    try:
        manager = ModelManager()
        
        # Vérifier la disponibilité
        print("\n🔍 Vérification disponibilité des modèles:")
        
        for model_type in ModelType:
            available, message = manager.is_model_available(model_type)
            status = "✅" if available else "❌"
            print(f"   {status} {model_type.value}: {message if message else 'Disponible'}")
        
        # Test de recommandation
        test_text = "Un texte de test de longueur moyenne pour tester la recommandation."
        
        print("\n🎯 Test de recommandation automatique:")
        recommended = manager.recommend_model(test_text, "balanced")
        print(f"   Modèle recommandé (balanced): {recommended.value}")
        
        recommended_cost = manager.recommend_model(test_text, "cost")
        print(f"   Modèle recommandé (cost): {recommended_cost.value}")
        
        recommended_speed = manager.recommend_model(test_text, "speed")
        print(f"   Modèle recommandé (speed): {recommended_speed.value}")
        
        # Test de résumé avec auto-selection
        print("\n🔄 Test résumé avec auto-sélection...")
        summary = manager.summarize_simple(
            test_text,
            model_type="auto",
            summary_length="short"
        )
        print(f"✅ Résumé généré: {summary[:100]}...")
        
        # Statistiques globales
        stats = manager.get_stats()
        print(f"\n📊 Statistiques globales:")
        print(f"   Total requêtes: {stats['total_requests']}")
        print(f"   LED: {stats['led_requests']}")
        print(f"   OpenAI: {stats['openai_requests']}")
        print(f"   Ollama: {stats['ollama_requests']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Erreur: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_quality_evaluation(summary_short, summary_long):
    """Test 4: Évaluation de la qualité"""
    print("\n" + "="*60)
    print("TEST 4: Évaluation de la qualité")
    print("="*60)
    
    if not summary_short or not summary_long:
        print("⏭️  Pas de résumés à évaluer")
        return
    
    try:
        evaluator = SummaryEvaluator(load_models=True)
        
        test_text = """
        L'intelligence artificielle transforme notre société de manière profonde.
        Cette technologie permet aux machines d'apprendre et de prendre des décisions.
        Les applications incluent la reconnaissance vocale, la vision par ordinateur,
        et les assistants virtuels. Cependant, elle soulève des questions éthiques
        importantes sur l'emploi, la vie privée et l'autonomie humaine.
        """
        
        print("\n📊 Évaluation résumé COURT:")
        report_short = evaluator.evaluate_summary(
            original_text=test_text,
            generated_summary=summary_short,
            model_name="Ollama (Mistral 7B)"
        )
        
        metrics_short = report_short.metrics
        print(f"   Score global: {metrics_short.overall_score:.3f}")
        print(f"   BERTScore: {metrics_short.bert_score:.3f}")
        print(f"   Compression: {metrics_short.compression_quality:.3f}")
        print(f"   Word Overlap: {metrics_short.word_overlap_ratio:.3f}")
        
        print("\n📊 Évaluation résumé LONG:")
        report_long = evaluator.evaluate_summary(
            original_text=test_text,
            generated_summary=summary_long,
            model_name="Ollama (Mistral 7B)"
        )
        
        metrics_long = report_long.metrics
        print(f"   Score global: {metrics_long.overall_score:.3f}")
        print(f"   BERTScore: {metrics_long.bert_score:.3f}")
        print(f"   Compression: {metrics_long.compression_quality:.3f}")
        print(f"   Word Overlap: {metrics_long.word_overlap_ratio:.3f}")
        
        # Interprétation
        def interpret_score(score):
            if score >= 0.8:
                return "🟢 Excellent"
            elif score >= 0.6:
                return "🟡 Bon"
            elif score >= 0.4:
                return "🟠 Acceptable"
            else:
                return "🔴 Faible"
        
        print(f"\n📈 Interprétation:")
        print(f"   Résumé court: {interpret_score(metrics_short.overall_score)}")
        print(f"   Résumé long: {interpret_score(metrics_long.overall_score)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Erreur d'évaluation: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_model_comparison():
    """Test 5: Comparaison des modèles disponibles"""
    print("\n" + "="*60)
    print("TEST 5: Comparaison des modèles")
    print("="*60)
    
    test_text = """
    Le changement climatique est l'un des plus grands défis de notre époque.
    Les activités humaines, notamment les émissions de gaz à effet de serre,
    sont la principale cause du réchauffement global. Les conséquences incluent
    la fonte des glaciers, la montée du niveau des mers, et des événements
    météorologiques extrêmes plus fréquents. Une action urgente est nécessaire.
    """
    
    try:
        manager = ModelManager()
        results = {}
        
        # Tester chaque modèle disponible
        for model_type in [ModelType.OLLAMA, ModelType.OPENAI, ModelType.LED]:
            available, _ = manager.is_model_available(model_type)
            
            if available:
                print(f"\n🧪 Test {model_type.value}...")
                try:
                    start_time = time.time()
                    summary = manager.summarize_simple(
                        test_text,
                        model_type=model_type.value,
                        summary_length="short"
                    )
                    processing_time = time.time() - start_time
                    
                    results[model_type.value] = {
                        'summary': summary,
                        'time': processing_time,
                        'words': len(summary.split())
                    }
                    
                    print(f"   ✅ Temps: {processing_time:.2f}s")
                    print(f"   📝 Longueur: {len(summary.split())} mots")
                    print(f"   {summary[:100]}...")
                    
                except Exception as e:
                    print(f"   ❌ Erreur: {e}")
            else:
                print(f"\n⏭️  {model_type.value}: Non disponible")
        
        # Résumé comparatif
        if results:
            print("\n" + "="*60)
            print("📊 RÉSUMÉ COMPARATIF")
            print("="*60)
            
            for model, data in sorted(results.items(), key=lambda x: x[1]['time']):
                print(f"\n{model.upper()}:")
                print(f"   Vitesse: {data['time']:.2f}s")
                print(f"   Longueur: {data['words']} mots")
                print(f"   Résumé: {data['summary'][:80]}...")
        
        return True
        
    except Exception as e:
        print(f"❌ Erreur: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Fonction principale de test"""
    print("\n" + "="*60)
    print("🧪 TESTS D'INTÉGRATION OLLAMA")
    print("="*60)
    print("\nCes tests vérifient:")
    print("1. ✓ Connexion au serveur Ollama")
    print("2. ✓ Génération de résumés")
    print("3. ✓ Intégration avec ModelManager")
    print("4. ✓ Évaluation de la qualité")
    print("5. ✓ Comparaison des modèles")
    
    results = {
        'connection': False,
        'summarization': False,
        'integration': False,
        'evaluation': False,
        'comparison': False
    }
    
    # Test 1: Connexion
    results['connection'] = test_ollama_connection()
    
    if results['connection']:
        # Test 2: Résumé
        summary_short, summary_long = test_ollama_summarization()
        results['summarization'] = (summary_short is not None)
        
        # Test 3: Intégration
        results['integration'] = test_model_manager_integration()
        
        # Test 4: Évaluation
        if summary_short and summary_long:
            results['evaluation'] = test_quality_evaluation(summary_short, summary_long)
        
        # Test 5: Comparaison
        results['comparison'] = test_model_comparison()
    
    # Résumé final
    print("\n" + "="*60)
    print("📊 RÉSUMÉ DES TESTS")
    print("="*60)
    
    total_tests = len(results)
    passed_tests = sum(1 for v in results.values() if v)
    
    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"   {status}: {test_name}")
    
    print(f"\n🎯 Résultat: {passed_tests}/{total_tests} tests réussis")
    
    if passed_tests == total_tests:
        print("\n🎉 Tous les tests sont passés ! Ollama est prêt à l'emploi.")
    elif results['connection']:
        print("\n💡 Ollama est connecté mais certains tests ont échoué.")
        print("   Vérifiez les logs ci-dessus pour plus de détails.")
    else:
        print("\n⚠️  Ollama n'est pas disponible.")
        print("\n💡 Pour démarrer:")
        print("   1. Installez Ollama: curl -fsSL https://ollama.com/install.sh | sh")
        print("   2. Démarrez le serveur: ollama serve")
        print("   3. Téléchargez un modèle: ollama pull mistral:7b")
        print("   4. Relancez les tests: python tests/test_ollama.py")
    
    return passed_tests == total_tests


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
