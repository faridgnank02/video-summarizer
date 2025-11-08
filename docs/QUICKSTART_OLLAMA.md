# 🚀 Guide de Démarrage Rapide - Ollama Integration

## Installation en 5 minutes

### 1️⃣ Installer Ollama

```bash
# macOS ou Linux
curl -fsSL https://ollama.com/install.sh | sh

# Vérifier l'installation
ollama --version
```

### 2️⃣ Démarrer le serveur

```bash
# Terminal 1 : Démarrer Ollama
ollama serve
```

### 3️⃣ Télécharger un modèle

```bash
# Terminal 2 : Télécharger Mistral (recommandé pour démarrer)
ollama pull mistral:7b

# Vérifier que le modèle est installé
ollama list
```

### 4️⃣ Tester l'intégration

```bash
# Activer l'environnement virtuel
source video-summarizer-env/bin/activate

# Lancer le script de test
python scripts/test_ollama_integration.py
```

Si tous les tests passent ✅, vous êtes prêt !

### 5️⃣ Lancer l'application

```bash
# Démarrer l'interface web
python scripts/launch.py
```

Ouvrir `http://localhost:8501` et sélectionner **Ollama** comme modèle.

---

## Test Rapide (ligne de commande)

```python
# Test simple en Python
from src.models.ollama_model import OllamaSummarizer

summarizer = OllamaSummarizer()
summary = summarizer.summarize(
    "Python est un langage de programmation populaire...",
    summary_type="short"
)
print(summary)
```

---

## Dépannage Express

### ❌ "Ollama server not available"

```bash
# Vérifier si Ollama tourne
curl http://localhost:11434/api/tags

# Si erreur, démarrer Ollama
ollama serve
```

### ❌ "Model not found"

```bash
# Lister les modèles installés
ollama list

# Télécharger le modèle manquant
ollama pull mistral:7b
```

### ❌ Résumés trop lents

```yaml
# Modifier config/model_config.yaml
models:
  ollama:
    max_tokens: 300  # Réduire de 500 à 300
    temperature: 0.2  # Plus déterministe
```

---
