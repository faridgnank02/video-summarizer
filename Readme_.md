# 🎥 Video Summarizer - Outil Professionnel de Résumé Vidéo

Un outil avancé de résumé de vidéos utilisant l'intelligence artificielle, développé à partir d'un notebook de fine-tuning LED et transformé en application complète.

## 🚀 Fonctionnalités

- **📹 Multi-sources** : YouTube, fichiers vidéo locaux, texte direct
- **🤖 Dual Models** : LED fine-tuné (qualité) + OpenAI GPT (rapidité)
- **📏 Longueurs flexibles** : Résumés courts (50-200 mots) ou longs (200-500 mots)
- **🌍 Multi-langues** : Français et anglais avec détection automatique
- **🔍 Named Entity Recognition** : Extraction automatique d'entités
- **📊 Monitoring** : Métriques ROUGE, temps de traitement, coûts
- **🎨 Interface moderne** : Streamlit avec design professionnel
- **⚡ API REST** : FastAPI pour intégrations

## 📋 Architecture

```
video-summarizer/
├── src/
│   ├── data/                    # Ingestion et préprocessing
│   │   ├── ingestion.py         # YouTube, fichiers locaux
│   │   ├── preprocessing.py     # Nettoyage, segmentation
│   │   └── datasets.py          # Gestion datasets
│   ├── models/                  # Modèles de ML
│   │   ├── led_model.py         # Modèle LED fine-tuné
│   │   ├── openai_model.py      # Modèle OpenAI GPT
│   │   ├── ner_model.py         # Named Entity Recognition
│   │   └── model_manager.py     # Orchestration modèles
│   ├── training/                # Entraînement
│   │   ├── trainer.py           # Fine-tuning LED
│   │   └── evaluation.py        # Métriques ROUGE
│   ├── api/                     # API REST
│   │   ├── summarization.py     # Endpoints résumé
│   │   └── video_processing.py  # Traitement vidéo
│   ├── monitoring/              # Monitoring
│   │   ├── metrics.py           # Collecte métriques
│   │   └── logging.py           # Système logs
│   └── ui/                      # Interface utilisateur
│       ├── streamlit_app.py     # Application Streamlit
│       └── components/          # Composants UI
├── config/                      # Configuration
│   ├── model_config.yaml        # Config modèles
│   └── app_config.yaml          # Config application
├── models/                      # Modèles entraînés
├── data/                        # Données d'entraînement
└── requirements.txt             # Dépendances
```

## 🛠️ Installation

### 1. Cloner et configurer l'environnement

```bash
cd /path/to/video-summarizer
source video-summarizer-env/bin/activate  # Environnement déjà créé
```

### 2. Installer les dépendances

```bash
pip install -r requirements.txt
```

### 3. Configurer les variables d'environnement

```bash
cp .env.example .env
# Éditer .env avec vos clés API
```

Variables requises :
- `OPENAI_API_KEY` : Clé API OpenAI (pour le modèle rapide)
- `WANDB_API_KEY` : Weights & Biases (optionnel, pour le monitoring)

### 4. Télécharger les modèles spaCy (pour NER)

```bash
python -m spacy download fr_core_news_sm
python -m spacy download en_core_web_sm
```

## 🚀 Utilisation

### Interface Streamlit (Recommandé)

```bash
streamlit run src/ui/streamlit_app.py
```

Accédez à `http://localhost:8501`

### API REST

```bash
uvicorn src.api.summarization:app --reload
```

Documentation API : `http://localhost:8000/docs`

### Utilisation programmatique

```python
from src.models.model_manager import ModelManager

# Initialiser le gestionnaire
manager = ModelManager()

# Résumé automatique (modèle recommandé)
summary = manager.summarize_simple(
    text="Votre texte ici...",
    model_type="auto",  # ou "led", "openai"
    summary_length="long"  # ou "short"
)

print(summary)
```

## 🤖 Modèles Disponibles

### LED Fine-tuné (Qualité)
- **Avantages** : Qualité élevée, compréhension contextuelle, spécialisé pour longs textes
- **Inconvénients** : Plus lent (~5-10s), nécessite GPU pour de meilleures performances
- **Usage** : Résumés de haute qualité, documents longs

### OpenAI GPT (Rapidité)
- **Avantages** : Très rapide (~2-3s), qualité excellente, multi-langues natif
- **Inconvénients** : Coût par utilisation, dépendance API externe
- **Usage** : Résumés rapides, prototypage, production légère

## 📊 Monitoring et Métriques

### Métriques collectées
- **ROUGE-1, ROUGE-2, ROUGE-L** : Qualité des résumés
- **Temps de traitement** : Performance des modèles
- **Coûts** : Utilisation API OpenAI
- **Taux d'erreur** : Fiabilité du système

### Visualisation
- Dashboard Streamlit intégré
- Export des métriques
- Comparaison des modèles

## 🔧 Configuration

### Modèles (`config/model_config.yaml`)

```yaml
models:
  led:
    model_name: "allenai/led-base-16384"
    max_input_length: 7168
    max_output_length: 512
    generation_config:
      num_beams: 2
      length_penalty: 2.0
  
  openai:
    model_name: "gpt-4"
    fallback_model: "gpt-3.5-turbo"
    temperature: 0.3

summary_lengths:
  short:
    min_length: 50
    max_length: 200
  long:
    min_length: 200
    max_length: 500
```

### Application (`config/app_config.yaml`)

```yaml
ui:
  title: "🎥 Résumeur de Vidéos IA"
  max_file_size: 500  # MB
  supported_formats: ["mp4", "avi", "mov", "mkv"]

monitoring:
  enable_mlflow: true
  mlflow_uri: "sqlite:///mlflow.db"
```

## 📈 Fine-tuning du modèle LED

### Préparer les données

```python
from src.training.trainer import LEDTrainer

trainer = LEDTrainer()
trainer.prepare_data(
    dataset_name="potsawee/podcast_summary_assessment",
    sample_size=1000
)
```

### Lancer l'entraînement

```python
trainer.train(
    num_epochs=3,
    batch_size=1,
    learning_rate=5e-5
)
```

### Évaluer le modèle

```python
from src.training.evaluation import ModelEvaluator

evaluator = ModelEvaluator()
metrics = evaluator.evaluate_model(
    model_path="./models/led_finetuned",
    test_data=test_dataset
)
```

## 🎨 Interface Utilisateur

### Fonctionnalités UI
- **Upload vidéos** : Drag & drop, support multi-formats
- **URLs YouTube** : Extraction automatique de transcripts
- **Sélection modèle** : Choix entre qualité et rapidité
- **Longueur résumé** : Court ou long
- **Entités nommées** : Mise en évidence automatique
- **Export** : PDF, TXT, JSON
- **Historique** : Sauvegarde des résumés

### Composants
- Barre de progression en temps réel
- Métriques de qualité
- Comparaison de modèles
- Visualisation des entités

## 🔍 Named Entity Recognition

```python
from src.models.ner_model import NERExtractor

ner = NERExtractor()
entities = ner.extract_entities(text)

# Résultat : {
#   "PERSON": ["John Doe", "Marie Dupont"],
#   "ORG": ["Google", "Microsoft"],
#   "GPE": ["Paris", "New York"]
# }
```

## 🧪 Tests

```bash
# Tests unitaires
pytest tests/

# Tests de modèles
python -m pytest tests/test_models.py -v

# Tests d'intégration
python -m pytest tests/test_integration.py -v
```

## 🚀 Déploiement

### Docker

```bash
# Construire l'image
docker build -t video-summarizer .

# Lancer le conteneur
docker run -p 8501:8501 -p 8000:8000 video-summarizer
```

### Docker Compose

```bash
docker-compose up -d
```

## 📝 Changelog

### Version 1.0.0
- ✅ Architecture modulaire complète
- ✅ Dual models (LED + OpenAI)
- ✅ Interface Streamlit avancée
- ✅ API REST avec FastAPI
- ✅ Monitoring et métriques
- ✅ Named Entity Recognition
- ✅ Multi-langues (FR/EN)
- ✅ Configuration flexible

## 🤝 Contribution

1. Fork le repository
2. Créer une branche feature (`git checkout -b feature/nouvelle-fonctionnalite`)
3. Commit les changes (`git commit -am 'Ajouter nouvelle fonctionnalité'`)
4. Push vers la branche (`git push origin feature/nouvelle-fonctionnalite`)
5. Créer une Pull Request

## 📄 Licence

MIT License - voir le fichier [LICENSE](LICENSE) pour plus de détails.

## 🆘 Support

- **Issues** : [GitHub Issues](https://github.com/votre-username/video-summarizer/issues)
- **Documentation** : [Wiki du projet](https://github.com/votre-username/video-summarizer/wiki)
- **Email** : votre.email@example.com

## 🙏 Remerciements

- Modèle LED original : [allenai/led-base-16384](https://huggingface.co/allenai/led-base-16384)
- Dataset : [potsawee/podcast_summary_assessment](https://huggingface.co/datasets/potsawee/podcast_summary_assessment)
- OpenAI pour l'API GPT
- Communauté Hugging Face

---

**Développé avec ❤️ pour démocratiser l'accès au résumé automatique de vidéos**