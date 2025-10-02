# 🚀 Démarrage Rapide - Video Summarizer

Guide express pour installer et utiliser le Video Summarizer en 5 minutes.

## ⚡ Installation Express

```bash
# 1. Installation automatique
python install.py

# 2. Activation environnement
source video-summarizer-env/bin/activate

# 3. Configuration optionnelle OpenAI
echo "OPENAI_API_KEY=sk-votre-clé" >> .env

# 4. Lancement
python launch.py
```

## 🎯 Utilisation Immédiate

### Interface Web (Recommandé)
1. Ouvrez votre navigateur sur `http://localhost:8501`
2. Collez une URL YouTube dans l'onglet "YouTube"
3. Choisissez le modèle LED (gratuit) ou OpenAI (rapide)
4. Sélectionnez la longueur : Court ou Moyen
5. Cliquez "Générer le résumé" et attendez

### Test Rapide en Ligne de Commande
```python
# Test avec texte simple
python -c "
from src.models.led_model import LEDSummarizer
led = LEDSummarizer()
print(led.summarize('Votre long texte ici...'))
"
```

## � Optimisation GPU M1 (MacBook Pro/Air)

**✅ GPU M1 automatiquement détecté et utilisé !**

### Performance GPU M1
- 🔥 **1.5x plus rapide** que le CPU
- ⚡ **Accélération Metal Performance Shaders (MPS)**
- 🧠 **Moins de consommation mémoire**
- 🌡️ **Moins de chauffe**

### Vérification GPU
```bash
# Tester que le GPU M1 est utilisé
python3 -c "
import torch
from src.models.led_model import LEDSummarizer
led = LEDSummarizer()
print(f'Device: {led.device}')  # Doit afficher 'mps'
"
```

## �🔧 Résolution Express

| Problème | Solution |
|----------|----------|
| ModuleNotFoundError | `pip install -r requirements.txt` |
| Mémoire insuffisante | Utilisez OpenAI au lieu de LED |
| GPU M1 non utilisé | Vérifiez `config/model_config.yaml` → `device: auto` |
| Erreur YouTube | `pip install --upgrade youtube-transcript-api` |
| Pas de clé OpenAI | Utilisez uniquement le modèle LED |

## 📚 Exemples Rapides

### YouTube
```python
from src.data.ingestion import DataIngestion
ingestion = DataIngestion()
result = ingestion.ingest_youtube("https://youtube.com/watch?v=xxx")
```

### Texte Direct
```python
from src.models.model_manager import ModelManager, SummaryRequest
manager = ModelManager()
request = SummaryRequest(text="...", model_type="led")
response = manager.summarize(request)
```

## 🎬 Demo URLs YouTube

Testez avec ces vidéos populaires :
- Conférence TED : `https://www.youtube.com/watch?v=...`
- Tuto tech : `https://www.youtube.com/watch?v=...`
- Documentaire : `https://www.youtube.com/watch?v=...`

## 📞 Aide Rapide

- 🐛 Bugs ? Vérifiez `test_functionality.py`
- 📖 Doc complète ? Lisez `README.md`
- 🔍 Architecture ? Consultez `PROJECT_SUMMARY.md`

---
⚡ **En moins de 5 minutes, résumez vos premières vidéos !** 🎥