#!/usr/bin/env python3
"""
Test de vérification des modèles spaCy
Vérifie que les modèles nécessaires pour l'évaluation hybride sont installés
"""

import sys
import os

def test_spacy_models():
    """Test l'installation des modèles spaCy requis"""
    print("🔍 Vérification des modèles spaCy...")
    
    # Liste des modèles requis
    required_models = [
        ("en_core_web_sm", "Anglais - Reconnaissance d'entités nommées"),
        ("fr_core_news_sm", "Français - Reconnaissance d'entités nommées (optionnel)")
    ]
    
    results = []
    
    for model_name, description in required_models:
        try:
            import spacy
            nlp = spacy.load(model_name)
            
            # Test rapide du modèle
            if model_name.startswith("en_"):
                test_text = "Apple Inc. was founded by Steve Jobs in California."
            else:
                test_text = "Apple Inc. a été fondée par Steve Jobs en Californie."
            
            doc = nlp(test_text)
            entities = [ent.text for ent in doc.ents]
            
            results.append(f"✅ {model_name}: {description}")
            results.append(f"   Test: {len(entities)} entités détectées - {entities}")
            
        except OSError:
            results.append(f"❌ {model_name}: NON INSTALLÉ - {description}")
            results.append(f"   Solution: python -m spacy download {model_name}")
            
        except ImportError:
            results.append(f"⚠️  spaCy non installé - pip install spacy")
            break
    
    return results

def test_evaluation_system():
    """Test que le système d'évaluation fonctionne avec spaCy"""
    print("\n🧪 Test du système d'évaluation hybride...")
    
    try:
        # Ajouter le répertoire src au path
        sys.path.insert(0, 'src')
        
        from evaluation.evaluator import SummaryEvaluator
        
        evaluator = SummaryEvaluator(load_models=False)
        
        # Test avec du contenu ayant des entités nommées
        original = "Microsoft Corporation was founded by Bill Gates and Paul Allen in 1975 in Albuquerque, New Mexico."
        summary = "Microsoft was founded by Bill Gates and Paul Allen in 1975."
        
        report = evaluator.evaluate_summary(original, summary, model_name="test")
        
        results = []
        results.append(f"✅ Évaluateur initialisé")
        results.append(f"✅ spaCy disponible: {evaluator.spacy_available}")
        results.append(f"✅ Score global: {report.metrics.overall_score:.3f}")
        results.append(f"✅ Word Overlap (NER+Keywords): {report.metrics.word_overlap_ratio:.3f}")
        
        # Test des composants individuels
        if hasattr(evaluator, '_calculate_ner_overlap'):
            ner_score = evaluator._calculate_ner_overlap(original, summary)
            results.append(f"✅ Composant NER: {ner_score:.3f}")
        
        if hasattr(evaluator, '_calculate_keyword_overlap'):
            keyword_score = evaluator._calculate_keyword_overlap(original, summary)
            results.append(f"✅ Composant Keywords: {keyword_score:.3f}")
        
        return results
        
    except Exception as e:
        return [f"❌ Erreur système d'évaluation: {e}"]

def check_requirements():
    """Vérifie que les dépendances nécessaires sont installées"""
    print("\n📦 Vérification des dépendances...")
    
    dependencies = [
        ("spacy", "spaCy - Traitement du langage naturel"),
        ("sentence_transformers", "Sentence Transformers - Embeddings sémantiques"),
        ("sklearn", "Scikit-learn - Machine Learning"),
        ("numpy", "NumPy - Calculs numériques")
    ]
    
    results = []
    
    for dep, description in dependencies:
        try:
            __import__(dep)
            results.append(f"✅ {dep}: {description}")
        except ImportError:
            results.append(f"❌ {dep}: MANQUANT - {description}")
            results.append(f"   Solution: pip install {dep}")
    
    return results

def main():
    """Test principal"""
    print("=" * 60)
    print("🧪 TEST D'INSTALLATION DES MODÈLES SPACY")
    print("=" * 60)
    
    # Test des dépendances
    dep_results = check_requirements()
    for result in dep_results:
        print(f"  {result}")
    
    # Test des modèles spaCy
    print("\n" + "=" * 60)
    print("📋 MODÈLES SPACY REQUIS")
    print("=" * 60)
    
    spacy_results = test_spacy_models()
    for result in spacy_results:
        print(f"  {result}")
    
    # Test du système d'évaluation
    print("\n" + "=" * 60)
    print("🎯 SYSTÈME D'ÉVALUATION HYBRIDE")
    print("=" * 60)
    
    eval_results = test_evaluation_system()
    for result in eval_results:
        print(f"  {result}")
    
    # Résumé et recommandations
    print("\n" + "=" * 60)
    print("📝 RÉSUMÉ ET RECOMMANDATIONS")
    print("=" * 60)
    
    # Compter les erreurs
    all_results = dep_results + spacy_results + eval_results
    errors = [r for r in all_results if r.startswith("❌")]
    warnings = [r for r in all_results if r.startswith("⚠️")]
    
    if not errors and not warnings:
        print("🎉 PARFAIT ! Tous les modèles spaCy sont installés et fonctionnels")
        print("✅ Le système d'évaluation hybride NER+Keywords est opérationnel")
        print("✅ Toutes les dépendances sont satisfaites")
    elif errors:
        print("⚠️  ACTIONS REQUISES :")
        print("\n🔧 Commandes d'installation manquantes :")
        for error in errors:
            if "spacy download" in error:
                print(f"   {error}")
            elif "pip install" in error:
                print(f"   {error}")
        
        print("\n💡 Pour installer tous les modèles spaCy d'un coup :")
        print("   python -m spacy download en_core_web_sm")
        print("   python -m spacy download fr_core_news_sm")
    else:
        print("✅ Installation correcte avec quelques avertissements mineurs")
    
    print("\n📚 Pour plus d'informations :")
    print("   - Guide d'installation : docs/QUICKSTART.md")
    print("   - Documentation technique : docs/TECHNICAL_DOCUMENTATION.md")
    
    return len(errors) == 0

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)