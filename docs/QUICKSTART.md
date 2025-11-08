# 🚀 Démarrage Rapide - Video Summarizer

Guide express pour installer et utiliser le Video Summarizer en 5 minutes.

## ✨ Nouveautés de Cette Version

Dans cette nouvelle version du projet, j'ai décidé de **remplacer complètement le modèle LED par Ollama** (avec Qwen3/Gemma3). Cette décision était motivée par plusieurs facteurs clés :

- 🐌 **LED était lent** : 30-200s par résumé vs 3-10s avec Ollama
- 💾 **LED consommait trop de mémoire** : 8-16GB RAM vs 2-4GB avec Ollama
- 🆓 **Explorer les LLMs locaux** : Je voulais tester les capacités des modèles de langage locaux modernes
- 🔒 **Confidentialité** : Traitement 100% local, zéro coût API
- ⚡ **Performance** : Ollama offre une inférence beaucoup plus rapide avec une qualité comparable

Le système d'évaluation automatique a également été simplifié pour une expérience utilisateur plus fluide et rapide.

## ⚡ Installation Express

```bash
# 1. Installation automatique
python install.py

# 2. Activation environnement
source video-summarizer-env/bin/activate

# 3. Installation Ollama (pour modèle local)
curl -fsSL https://ollama.com/install.sh | sh
ollama serve
ollama pull qwen3:1b  # Rapide et efficace

# 4. Configuration optionnelle OpenAI
echo "OPENAI_API_KEY=sk-votre-clé" >> .env

# 5. Lancement
python launch.py
```

## 🎯 Utilisation Immédiate

### Interface Web (Recommandé)

1. Ouvrez votre navigateur sur `http://localhost:8501`
2. Collez une URL YouTube dans l'onglet "YouTube"
3. Choisissez le modèle Ollama (gratuit, local) ou OpenAI (rapide, cloud)
4. Sélectionnez la longueur : Court (2-3 phrases) ou Long (5-8 phrases)
5. Cliquez "Générer le résumé" et attendez

### Test Rapide en Ligne de Commande

```python
# Test avec Ollama
python -c "
from src.models.ollama_model import OllamaSummarizer
ollama = OllamaSummarizer(model_name='qwen3:1b')
print(ollama.summarize('Votre long texte ici...', summary_type='short'))
"
```

## 📊 Comparaison des Modèles

| Modèle | Vitesse | Mémoire | Coût | Qualité |
|--------|---------|---------|------|---------|
| **Ollama (Qwen3)** | 3-10s | 2-4GB | 🆓 Gratuit | ⭐⭐⭐⭐ |
| **Ollama (Gemma3)** | 3-10s | 2-4GB | 🆓 Gratuit | ⭐⭐⭐⭐ |
| **OpenAI GPT-4** | 2-5s | 0GB | 💰 Payant | ⭐⭐⭐⭐⭐ |

**Pourquoi Qwen3/Gemma3 ?**
- Optimisés pour suivre des instructions précises
- Beaucoup plus rapides que l'ancien modèle LED (3-10s vs 30-200s)
- Consomment beaucoup moins de mémoire (2-4GB vs 8-16GB)
- Excellente qualité pour leur taille (1B paramètres)

## 🔧 Résolution Express

| Problème | Solution |
|----------|----------|
| ModuleNotFoundError | `pip install -r requirements.txt` |
| Ollama non trouvé | `curl -fsSL https://ollama.com/install.sh \| sh` puis `ollama serve` |
| Mémoire insuffisante | Utilisez qwen3:1b au lieu de mistral:7b |
| Pas de clé OpenAI | Utilisez uniquement le modèle Ollama (gratuit) |
| Résumés de mauvaise qualité | Essayez gemma3:1b ou ajustez la température dans config/model_config.yaml |

## 📚 Exemples Rapides

### YouTube

```python
from src.data.ingestion import DataIngestion
ingestion = DataIngestion()
result = ingestion.ingest_youtube("https://youtube.com/watch?v=xxx")
```

### Texte Direct

```python
from src.models.model_manager import ModelManager
manager = ModelManager()
summary = manager.summarize_simple(
    text="Votre texte long...",
    model_type="ollama",  # ou "openai"
    summary_length="short"  # ou "long"
)
print(summary)
```

## 🎬 Demo URLs YouTube

Testez avec ces vidéos populaires :

- Conférence TED
- Tuto tech
- Documentaire

## 📞 Aide Rapide

- 🐛 Bugs ? Vérifiez `test_functionality.py`
- 📖 Doc complète ? Lisez `README.md`
- 🔍 Architecture ? Consultez `TECHNICAL_DOCUMENTATION.md`
- 💡 Migration LED → Ollama ? Voir `LED_REMOVAL_SUMMARY.md`

---

⚡ **En moins de 5 minutes, résumez vos premières vidéos !** 🎥
