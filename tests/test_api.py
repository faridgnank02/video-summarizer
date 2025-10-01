#!/usr/bin/env python3
"""
Tests complets de l'API Video Summarizer Enterprise
"""

import asyncio
import json
import time
from typing import Dict, Any
import requests
import sys
from pathlib import Path

# Configuration de l'API
API_BASE_URL = "http://localhost:8000"
HEADERS = {"Content-Type": "application/json"}

class APITester:
    """Testeur pour l'API Video Summarizer"""
    
    def __init__(self, base_url: str = API_BASE_URL):
        self.base_url = base_url
        self.session = requests.Session()
        self.session.headers.update(HEADERS)
    
    def test_health_check(self) -> bool:
        """Test du health check"""
        print("🏥 Test du health check...")
        
        try:
            response = self.session.get(f"{self.base_url}/health")
            
            if response.status_code == 200:
                data = response.json()
                print(f"   ✅ Status: {data.get('status', 'unknown')}")
                print(f"   📊 Uptime: {data.get('uptime', 0):.2f}s")
                
                models = data.get('models_status', {})
                for model, available in models.items():
                    status = "✅" if available else "❌"
                    print(f"   🤖 {model}: {status}")
                
                return True
            else:
                print(f"   ❌ Status code: {response.status_code}")
                return False
                
        except Exception as e:
            print(f"   ❌ Erreur: {e}")
            return False
    
    def test_text_summarization(self) -> bool:
        """Test du résumé de texte"""
        print("\n📝 Test du résumé de texte...")
        
        payload = {
            "text": """
            L'intelligence artificielle (IA) représente l'une des révolutions technologiques 
            les plus importantes de notre époque. Cette technologie permet aux machines 
            d'apprendre, de raisonner et de prendre des décisions de manière autonome. 
            Les applications sont multiples : reconnaissance vocale, vision par ordinateur, 
            traduction automatique, véhicules autonomes, et diagnostic médical assisté. 
            Cependant, cette évolution soulève des questions éthiques importantes sur 
            l'avenir du travail, la protection de la vie privée, et le contrôle de ces 
            technologies puissantes. Il est crucial de développer l'IA de manière 
            responsable pour maximiser ses bénéfices tout en minimisant les risques.
            """,
            "model_type": "auto",
            "summary_length": "short",
            "user_id": "test_user",
            "evaluate": True
        }
        
        try:
            start_time = time.time()
            response = self.session.post(f"{self.base_url}/api/v1/summarize/text", json=payload)
            response_time = time.time() - start_time
            
            if response.status_code == 200:
                data = response.json()
                
                if data.get('success'):
                    result = data.get('data', {})
                    print(f"   ✅ Résumé généré en {response_time:.2f}s")
                    print(f"   📊 Modèle: {result.get('model_used', 'unknown')}")
                    print(f"   📏 Compression: {result.get('compression_ratio', 0):.2f}")
                    print(f"   📝 Résumé: {result.get('summary', '')[:100]}...")
                    
                    # Évaluation
                    if 'evaluation' in result:
                        eval_data = result['evaluation']
                        print(f"   🎯 Score: {eval_data.get('overall_score', 0):.3f}")
                        print(f"   🔗 Similarité: {eval_data.get('semantic_similarity', 0):.3f}")
                    
                    return True
                else:
                    print(f"   ❌ Échec: {data.get('message', 'Erreur inconnue')}")
                    return False
            else:
                print(f"   ❌ Status code: {response.status_code}")
                print(f"   📄 Réponse: {response.text}")
                return False
                
        except Exception as e:
            print(f"   ❌ Erreur: {e}")
            return False
    
    def test_youtube_summarization(self) -> bool:
        """Test du résumé YouTube"""
        print("\n🎥 Test du résumé YouTube...")
        
        payload = {
            "url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ",  # Rick Roll - vidéo courte
            "model_type": "auto",
            "summary_length": "short",
            "user_id": "test_user",
            "evaluate": False
        }
        
        try:
            start_time = time.time()
            response = self.session.post(f"{self.base_url}/api/v1/summarize/youtube", json=payload)
            response_time = time.time() - start_time
            
            if response.status_code == 200:
                data = response.json()
                
                if data.get('success'):
                    result = data.get('data', {})
                    video_info = result.get('video_info', {})
                    
                    print(f"   ✅ Résumé YouTube généré en {response_time:.2f}s")
                    print(f"   🎬 Titre: {video_info.get('title', 'Inconnu')[:50]}...")
                    print(f"   🌍 Langue: {video_info.get('language', 'unknown')}")
                    print(f"   📊 Modèle: {result.get('model_used', 'unknown')}")
                    print(f"   📝 Résumé: {result.get('summary', '')[:100]}...")
                    
                    return True
                else:
                    print(f"   ❌ Échec: {data.get('message', 'Erreur inconnue')}")
                    return False
            else:
                print(f"   ❌ Status code: {response.status_code}")
                return False
                
        except Exception as e:
            print(f"   ❌ Erreur: {e}")
            return False
    
    def test_batch_summarization(self) -> bool:
        """Test du résumé en lot"""
        print("\n📦 Test du résumé en lot...")
        
        payload = {
            "items": [
                {
                    "text": "L'intelligence artificielle transforme notre monde avec des applications variées dans tous les secteurs."
                },
                {
                    "text": "Le machine learning permet aux ordinateurs d'apprendre à partir de données sans programmation explicite."
                },
                {
                    "text": "La robotique combine mécanique, électronique et informatique pour créer des machines autonomes."
                }
            ],
            "model_type": "auto",
            "summary_length": "short",
            "user_id": "test_user"
        }
        
        try:
            start_time = time.time()
            response = self.session.post(f"{self.base_url}/api/v1/summarize/batch", json=payload)
            response_time = time.time() - start_time
            
            if response.status_code == 200:
                data = response.json()
                
                if data.get('success'):
                    result = data.get('data', {})
                    stats = result.get('summary_stats', {})
                    
                    print(f"   ✅ Lot traité en {response_time:.2f}s")
                    print(f"   📊 Éléments: {stats.get('total_items', 0)}")
                    print(f"   ✅ Réussis: {stats.get('successful', 0)}")
                    print(f"   ❌ Échoués: {stats.get('failed', 0)}")
                    print(f"   📏 Compression moyenne: {stats.get('avg_compression_ratio', 0):.2f}")
                    
                    return stats.get('successful', 0) > 0
                else:
                    print(f"   ❌ Échec: {data.get('message', 'Erreur inconnue')}")
                    return False
            else:
                print(f"   ❌ Status code: {response.status_code}")
                return False
                
        except Exception as e:
            print(f"   ❌ Erreur: {e}")
            return False
    
    def test_evaluation_endpoint(self) -> bool:
        """Test de l'endpoint d'évaluation"""
        print("\n🎯 Test de l'évaluation...")
        
        payload = {
            "original_text": "L'intelligence artificielle transforme notre société moderne avec des applications dans de nombreux domaines.",
            "generated_summary": "L'IA révolutionne la société avec des applications variées.",
            "reference_summary": "L'intelligence artificielle change notre monde moderne.",
            "model_name": "test_model"
        }
        
        try:
            start_time = time.time()
            response = self.session.post(f"{self.base_url}/api/v1/evaluate", json=payload)
            response_time = time.time() - start_time
            
            if response.status_code == 200:
                data = response.json()
                
                if data.get('success'):
                    result = data.get('data', {})
                    metrics = result.get('metrics', {})
                    
                    print(f"   ✅ Évaluation terminée en {response_time:.2f}s")
                    print(f"   🎯 Score global: {result.get('overall_score', 0):.3f}")
                    print(f"   🔗 Similarité sémantique: {metrics.get('semantic_similarity', 0):.3f}")
                    print(f"   📊 Cohérence: {metrics.get('coherence_score', 0):.3f}")
                    
                    recommendations = result.get('recommendations', [])
                    if recommendations:
                        print(f"   💡 Recommandations: {len(recommendations)}")
                    
                    return True
                else:
                    print(f"   ❌ Échec: {data.get('message', 'Erreur inconnue')}")
                    return False
            else:
                print(f"   ❌ Status code: {response.status_code}")
                return False
                
        except Exception as e:
            print(f"   ❌ Erreur: {e}")
            return False
    
    def test_metrics_endpoint(self) -> bool:
        """Test de l'endpoint de métriques"""
        print("\n📊 Test des métriques...")
        
        try:
            response = self.session.get(f"{self.base_url}/api/v1/metrics?hours=1&metric_type=system")
            
            if response.status_code == 200:
                data = response.json()
                
                if data.get('success'):
                    result = data.get('data', {})
                    current = result.get('current', {})
                    historical = result.get('historical', [])
                    
                    print(f"   ✅ Métriques récupérées")
                    print(f"   📈 Données historiques: {len(historical)} points")
                    
                    system = current.get('system', {})
                    if system:
                        print(f"   💻 CPU: {system.get('cpu_percent', 0):.1f}%")
                        print(f"   🧠 Mémoire: {system.get('memory_percent', 0):.1f}%")
                    
                    health = current.get('health', {})
                    if health:
                        print(f"   ❤️  Santé: {health.get('score', 0)}/100")
                    
                    return True
                else:
                    print(f"   ❌ Échec: {data.get('message', 'Erreur inconnue')}")
                    return False
            else:
                print(f"   ❌ Status code: {response.status_code}")
                return False
                
        except Exception as e:
            print(f"   ❌ Erreur: {e}")
            return False
    
    def test_models_info(self) -> bool:
        """Test de l'endpoint d'information sur les modèles"""
        print("\n🤖 Test des informations modèles...")
        
        try:
            response = self.session.get(f"{self.base_url}/api/v1/models")
            
            if response.status_code == 200:
                data = response.json()
                
                if data.get('success'):
                    result = data.get('data', {})
                    models = result.get('models', {})
                    
                    print(f"   ✅ Informations récupérées")
                    print(f"   🎯 Recommandation auto: {result.get('recommendation_engine', False)}")
                    
                    for model_name, model_info in models.items():
                        available = model_info.get('available', False)
                        status = "✅" if available else "❌"
                        print(f"   {status} {model_name}: {model_info.get('status', 'unknown')}")
                    
                    return True
                else:
                    print(f"   ❌ Échec: {data.get('message', 'Erreur inconnue')}")
                    return False
            else:
                print(f"   ❌ Status code: {response.status_code}")
                return False
                
        except Exception as e:
            print(f"   ❌ Erreur: {e}")
            return False
    
    def run_all_tests(self) -> Dict[str, bool]:
        """Exécute tous les tests"""
        
        print("🧪 Tests complets de l'API Video Summarizer")
        print("=" * 60)
        
        tests = [
            ("Health Check", self.test_health_check),
            ("Résumé de texte", self.test_text_summarization),
            ("Résumé YouTube", self.test_youtube_summarization),
            ("Résumé en lot", self.test_batch_summarization),
            ("Évaluation", self.test_evaluation_endpoint),
            ("Métriques", self.test_metrics_endpoint),
            ("Info modèles", self.test_models_info)
        ]
        
        results = {}
        
        for test_name, test_func in tests:
            print(f"\n📋 {test_name}")
            print("-" * 40)
            
            try:
                result = test_func()
                results[test_name] = result
                
                if result:
                    print(f"   🎉 {test_name}: RÉUSSI")
                else:
                    print(f"   💥 {test_name}: ÉCHOUÉ")
                    
            except Exception as e:
                print(f"   💥 {test_name}: ERREUR - {e}")
                results[test_name] = False
        
        # Résumé final
        print("\n" + "=" * 60)
        print("📊 RÉSUMÉ DES TESTS")
        print("=" * 60)
        
        passed = sum(results.values())
        total = len(results)
        
        print(f"✅ Tests réussis: {passed}/{total}")
        print(f"❌ Tests échoués: {total - passed}/{total}")
        
        if passed == total:
            print("🎉 Tous les tests sont passés ! API entièrement fonctionnelle.")
        elif passed >= total * 0.75:
            print("🟡 La plupart des tests passent. Quelques ajustements mineurs.")
        else:
            print("🔴 Plusieurs tests échouent. Vérifiez la configuration de l'API.")
        
        print(f"\n💡 Pour utiliser l'API :")
        print(f"   • Documentation: {self.base_url}/docs")
        print(f"   • Health check: {self.base_url}/health")
        print(f"   • Métriques: {self.base_url}/api/v1/metrics")
        
        return results

def main():
    """Fonction principale"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Test de l'API Video Summarizer")
    parser.add_argument("--url", default=API_BASE_URL, help="URL de base de l'API")
    parser.add_argument("--test", help="Test spécifique à exécuter")
    
    args = parser.parse_args()
    
    tester = APITester(args.url)
    
    if args.test:
        # Test spécifique
        test_methods = {
            "health": tester.test_health_check,
            "text": tester.test_text_summarization,
            "youtube": tester.test_youtube_summarization,
            "batch": tester.test_batch_summarization,
            "evaluation": tester.test_evaluation_endpoint,
            "metrics": tester.test_metrics_endpoint,
            "models": tester.test_models_info
        }
        
        if args.test in test_methods:
            success = test_methods[args.test]()
            sys.exit(0 if success else 1)
        else:
            print(f"❌ Test '{args.test}' non reconnu")
            print(f"💡 Tests disponibles: {', '.join(test_methods.keys())}")
            sys.exit(1)
    else:
        # Tous les tests
        results = tester.run_all_tests()
        success_rate = sum(results.values()) / len(results)
        sys.exit(0 if success_rate >= 0.75 else 1)

if __name__ == "__main__":
    main()