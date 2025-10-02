# Video Summarizer 🎥📝

Un outil professionnel de résumé de vidéos alimenté par l'IA, conçu pour extraire et synthétiser automatiquement le contenu de vidéos YouTube et de textes longs.

## Fonctionnalités principales

- **Résumé multi-sources** : Vidéos YouTube (via URL), texte direct, fichiers locaux
- **Modèles IA avancés** : LED fine-tuné pour la qualité, OpenAI GPT pour la rapidité
- **Sélection automatique** : Recommandation intelligente du meilleur modèle selon le contenu
- **Interface moderne** : Application web Streamlit + API REST
- **Résumés adaptatifs** : Longueur court/long selon vos besoins
- **Historique intégré** : Sauvegarde et export des résumés

## 🏗️ Architecture

```
src/
├── data/           # Ingestion et préprocessing
├── models/         # Modèles IA (LED, OpenAI)
├── ui/            # Interface Streamlit
├── api/           # API REST
├── training/      # Entraînement des modèles
└── monitoring/    # Suivi des performances

config/            # Configuration YAML
tests/            # Tests unitaires et fonctionnels
```

## 📋 Prérequis

- Python 3.8+
- GPU recommandé (pour le modèle LED)
- Clé API OpenAI (optionnel)

## ⚡ Installation rapide

1. **Clonez le repository**
```bash
git clone <votre-repo>
cd summarizer
```

2. **Créez l'environnement virtuel**
```bash
python -m venv video-summarizer-env
source video-summarizer-env/bin/activate  # macOS/Linux
# video-summarizer-env\Scripts\activate   # Windows
```

3. **Installez les dépendances**
```bash
pip install -r requirements.txt
```

4. **Configuration (optionnel)**
```bash
cp .env.example .env
# Ajoutez votre OPENAI_API_KEY dans .env
```

## 🎯 Utilisation

### Interface Web (Recommandé)

Lancez l'application Streamlit :

```bash
streamlit run src/ui/streamlit_app.py
```

Ouvrez http://localhost:8501 et :
1. Collez une URL YouTube ou du texte
2. Choisissez la longueur du résumé
3. Laissez l'IA choisir le meilleur modèle (ou sélectionnez manuellement)
4. Obtenez votre résumé en quelques secondes !

### API REST

```bash
uvicorn src.api.summarization:app --reload
```

Documentation interactive : http://localhost:8000/docs

### Usage programmatique

```python
from src.models.model_manager import ModelManager

# Initialisation
manager = ModelManager()

# Résumé automatique (recommandation de modèle)
summary = manager.summarize_simple(
    text="Votre texte long...",
    model_type="auto",  # ou "led", "openai"
    summary_length="short"  # ou "long"
)

print(f"Résumé : {summary.summary}")
print(f"Modèle utilisé : {summary.model_used}")
print(f"Temps : {summary.processing_time:.2f}s")
```

## 🤖 Modèles disponibles

| Modèle | Avantages | Inconvénients | Recommandé pour |
|--------|-----------|---------------|-----------------|
| **LED Fine-tuné** | Qualité élevée, spécialisé longs textes | Plus lent (5-10s), GPU requis | Textes académiques, rapports |
| **OpenAI GPT** | Très rapide (2-3s), multi-langues | Coût par usage, dépendance API | Usage quotidien, prototypage |

L'option `"auto"` sélectionne automatiquement le meilleur modèle selon :
- Longueur du texte
- Disponibilité des modèles
- Configuration utilisateur

## 📊 Tests et validation

```bash
# Tests d'architecture
python test_architecture.py

# Tests fonctionnels complets
python test_functionality.py

# Tests unitaires spécifiques
pytest tests/ -v
```

## ⚙️ Configuration avancée

### Modèles personnalisés

Modifiez `config/model_config.yaml` :

```yaml
led:
  model_name: "allenai/led-base-16384"
  max_length: 1024
  min_length: 50
  
openai:
  model: "gpt-3.5-turbo"
  max_tokens: 500
  temperature: 0.3
```

### Interface utilisateur

Personnalisez `config/app_config.yaml` :

```yaml
streamlit:
  page_title: "Mon Résumeur"
  sidebar_options: ["Historique", "Paramètres"]
  
ui:
  default_summary_length: "short"
  show_word_count: true
```

## 🔧 Dépannage

### Problèmes courants

**Erreur GPU (modèle LED)**
```bash
# Utilisez la version CPU
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

**Clé API OpenAI manquante**
- Ajoutez `OPENAI_API_KEY=sk-xxx` dans votre fichier `.env`
- Ou utilisez uniquement le modèle LED avec `model_type="led"`

**Vidéo YouTube inaccessible**
- Vérifiez que la vidéo est publique
- Testez avec une autre URL
- Utilisez le texte direct en alternative

### Logs et monitoring

Les logs sont disponibles dans `logs/` avec différents niveaux de verbosité configurables.

## Déploiement

### Docker

```bash
docker build -t video-summarizer .
docker run -p 8501:8501 video-summarizer
```

### Cloud

Le projet est compatible avec :
- **Streamlit Cloud** (interface web)
- **Heroku/Railway** (API)
- **AWS/GCP** (déploiement complet)

## Contribution

1. Fork le project
2. Créez votre branche (`git checkout -b feature/nouvelle-fonctionnalite`)
3. Commit vos changes (`git commit -m 'Ajout nouvelle fonctionnalité'`)
4. Push sur la branche (`git push origin feature/nouvelle-fonctionnalite`)
5. Ouvrez une Pull Request

## 📄 Licence

Ce projet est sous licence MIT. Voir `LICENSE` pour plus de détails.

## 🙏 Remerciements

- [Hugging Face](https://huggingface.co/) pour les modèles Transformers
- [OpenAI](https://openai.com/) pour l'API GPT
- [Streamlit](https://streamlit.io/) pour l'interface utilisateur
- Communauté open-source pour les outils et bibliothèques

---

