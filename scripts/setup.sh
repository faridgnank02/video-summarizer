#!/bin/bash

# Script de lancement pour Video Summarizer
# Ce script active l'environnement virtuel et lance l'interface Streamlit

echo "🎥 Lancement du Video Summarizer..."
echo "=================================="

# Vérifier que nous sommes dans le bon répertoire
if [ ! -f "requirements.txt" ]; then
    echo "❌ Erreur: Ce script doit être lancé depuis le répertoire du projet"
    exit 1
fi

# Activer l'environnement virtuel
if [ -d "video-summarizer-env" ]; then
    echo "🔄 Activation de l'environnement virtuel..."
    source video-summarizer-env/bin/activate
else
    echo "❌ Erreur: Environnement virtuel 'video-summarizer-env' non trouvé"
    echo "Créez-le avec: python3 -m venv video-summarizer-env"
    exit 1
fi

# Vérifier que Streamlit est installé
if ! command -v streamlit &> /dev/null; then
    echo "❌ Streamlit non installé. Installation..."
    pip install streamlit
fi

# Configurer les variables d'environnement si .env existe
if [ -f ".env" ]; then
    echo "📝 Chargement des variables d'environnement..."
    export $(grep -v '^#' .env | xargs)
fi

# Vérifier la clé API OpenAI (optionnel)
if [ -z "$OPENAI_API_KEY" ]; then
    echo "⚠️  OPENAI_API_KEY non configurée"
    echo "💡 Pour utiliser OpenAI GPT, définissez OPENAI_API_KEY dans .env"
else
    echo "✅ Clé API OpenAI configurée"
fi

# Lancer l'application Streamlit
echo "🚀 Lancement de l'interface web..."
echo "📱 L'application sera disponible sur: http://localhost:8501"
echo ""

streamlit run src/ui/streamlit_app.py