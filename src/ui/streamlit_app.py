"""
Interface Streamlit pour le Video Summarizer
Application web moderne pour le résumé de vidéos
"""

import streamlit as st
import os
import sys
import logging
import time
from pathlib import Path
from typing import Optional, Dict, Any

# Ajouter le répertoire src au PYTHONPATH
sys.path.append(str(Path(__file__).parent.parent))

try:
    from data.ingestion import DataIngestion, VideoData
    from data.preprocessing import TextPreprocessor
    from models.model_manager import ModelManager, ModelType, SummaryLength
except ImportError as e:
    st.error(f"Erreur d'importation des modules: {e}")
    st.stop()

# Configuration de la page
st.set_page_config(
    page_title="🎥 Résumeur de Vidéos IA",
    page_icon="🎥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class VideoSummarizerApp:
    """Application Streamlit pour le résumé de vidéos"""
    
    def __init__(self):
        self.ingestion = DataIngestion()
        self.preprocessor = TextPreprocessor()
        self.model_manager = None
        
        # État de l'application
        if 'summary_history' not in st.session_state:
            st.session_state.summary_history = []
        
        if 'current_video_data' not in st.session_state:
            st.session_state.current_video_data = None
    
    def initialize_models(self):
        """Initialise le gestionnaire de modèles (lazy loading)"""
        if self.model_manager is None:
            with st.spinner("🔄 Initialisation des modèles..."):
                try:
                    config_path = Path(__file__).parent.parent.parent / "config" / "model_config.yaml"
                    self.model_manager = ModelManager(str(config_path) if config_path.exists() else None)
                    st.success("✅ Modèles initialisés avec succès!")
                except Exception as e:
                    st.error(f"❌ Erreur lors de l'initialisation des modèles: {e}")
                    return False
        return True
    
    def render_header(self):
        """Affiche l'en-tête de l'application"""
        st.title("🎥 Résumeur de Vidéos IA")
        st.markdown("""
        **Transformez vos vidéos en résumés intelligents** avec deux modèles au choix :
        - 🎯 **LED Fine-tuné** : Qualité maximale pour résumés détaillés
        - ⚡ **OpenAI GPT** : Vitesse optimale pour résumés rapides
        """)
        st.divider()
    
    def render_sidebar(self):
        """Affiche la barre latérale avec les paramètres"""
        st.sidebar.header("⚙️ Paramètres")
        
        # Sélection du modèle
        model_option = st.sidebar.selectbox(
            "🤖 Modèle de résumé",
            ["Auto (Recommandé)", "LED (Qualité)", "OpenAI (Rapidité)"],
            help="Auto choisit automatiquement le meilleur modèle selon le contexte"
        )
        
        # Sélection de la longueur
        length_option = st.sidebar.selectbox(
            "📏 Longueur du résumé",
            ["Long (200-500 mots)", "Court (50-200 mots)"],
            help="Longueur approximative du résumé généré"
        )
        
        # Langue
        language_option = st.sidebar.selectbox(
            "🌍 Langue",
            ["Auto-détection", "Français", "Anglais"],
            help="Langue du résumé généré"
        )
        
        # Informations sur les modèles
        with st.sidebar.expander("ℹ️ Informations sur les modèles"):
            st.markdown("""
            **LED Fine-tuné:**
            - ✅ Qualité élevée
            - ✅ Textes longs
            - ⏱️ Plus lent (~5-10s)
            
            **OpenAI GPT:**
            - ✅ Très rapide (~2-3s)
            - ✅ Multi-langues
            - 💰 Coût par utilisation
            """)
        
        return {
            'model': model_option,
            'length': length_option,
            'language': language_option
        }
    
    def render_video_input(self):
        """Affiche les options d'entrée vidéo"""
        st.header("📹 Source Vidéo")
        
        # Tabs pour différentes sources
        tab1, tab2, tab3 = st.tabs(["🔗 YouTube", "📁 Fichier Local", "📝 Texte Direct"])
        
        video_data = None
        
        with tab1:
            st.subheader("URL YouTube")
            youtube_url = st.text_input(
                "Entrez l'URL de la vidéo YouTube :",
                placeholder="https://www.youtube.com/watch?v=...",
                help="Collez l'URL complète de la vidéo YouTube"
            )
            
            if st.button("📥 Extraire le transcript", key="youtube"):
                if youtube_url:
                    try:
                        with st.spinner("🔄 Extraction du transcript..."):
                            video_data = self.ingestion.process_youtube_url(youtube_url)
                            st.success(f"✅ Transcript extrait : {video_data.title}")
                    except Exception as e:
                        st.error(f"❌ Erreur : {e}")
                else:
                    st.warning("⚠️ Veuillez entrer une URL YouTube")
        
        with tab2:
            st.subheader("Fichier Vidéo Local")
            uploaded_file = st.file_uploader(
                "Choisissez un fichier vidéo",
                type=['mp4', 'avi', 'mov', 'mkv', 'webm'],
                help="Formats supportés : MP4, AVI, MOV, MKV, WebM"
            )
            
            if uploaded_file and st.button("🎙️ Transcrire l'audio", key="local"):
                st.warning("🚧 Fonctionnalité en développement (nécessite Whisper)")
                # TODO: Implémenter la transcription locale avec Whisper
        
        with tab3:
            st.subheader("Texte Direct")
            direct_text = st.text_area(
                "Collez votre texte ici :",
                height=200,
                placeholder="Collez le transcript ou le texte que vous souhaitez résumer...",
                help="Texte brut à résumer directement"
            )
            
            custom_title = st.text_input(
                "Titre (optionnel) :",
                placeholder="Titre de votre texte"
            )
            
            if st.button("📝 Utiliser ce texte", key="direct"):
                if direct_text.strip():
                    video_data = self.ingestion.process_text_input(
                        direct_text, 
                        custom_title or "Texte personnalisé"
                    )
                    st.success("✅ Texte prêt pour le résumé")
                else:
                    st.warning("⚠️ Veuillez entrer du texte")
        
        return video_data
    
    def render_video_info(self, video_data: VideoData):
        """Affiche les informations sur la vidéo"""
        st.header("📊 Informations sur le contenu")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("📝 Titre", value="", delta=video_data.title)
        
        with col2:
            word_count = len(video_data.transcript.split())
            st.metric("📊 Mots", word_count)
        
        with col3:
            st.metric("🌍 Langue", video_data.language.upper())
        
        with col4:
            if video_data.duration:
                duration_min = video_data.duration // 60
                st.metric("⏱️ Durée", f"{duration_min}min")
            else:
                st.metric("📄 Source", video_data.source)
        
        # Prévisualisation du transcript
        with st.expander("👁️ Prévisualiser le transcript"):
            preview_length = min(500, len(video_data.transcript))
            st.text_area(
                "Transcript (premiers 500 caractères) :",
                video_data.transcript[:preview_length] + ("..." if len(video_data.transcript) > preview_length else ""),
                height=150,
                disabled=True
            )
    
    def render_summary_generation(self, video_data: VideoData, params: Dict[str, str]):
        """Affiche la section de génération de résumé"""
        st.header("🎯 Génération du Résumé")
        
        if not self.initialize_models():
            return
        
        # Configuration des paramètres
        model_type = "auto"
        if "LED" in params['model']:
            model_type = "led"
        elif "OpenAI" in params['model']:
            model_type = "openai"
        
        summary_length = "short" if "Court" in params['length'] else "long"
        
        language = None
        if params['language'] != "Auto-détection":
            language = "french" if params['language'] == "Français" else "english"
        
        # Bouton de génération
        if st.button("🚀 Générer le Résumé", type="primary", use_container_width=True):
            try:
                start_time = time.time()
                
                with st.spinner("🔄 Génération en cours..."):
                    # Préprocessing
                    processed_data = self.preprocessor.preprocess(video_data.transcript)
                    
                    # Génération du résumé
                    summary = self.model_manager.summarize_simple(
                        text=processed_data.text,
                        model_type=model_type,
                        summary_length=summary_length,
                        language=language
                    )
                
                processing_time = time.time() - start_time
                
                # Affichage du résumé
                st.success(f"✅ Résumé généré en {processing_time:.1f}s")
                
                # Métriques du résumé
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("📊 Mots", len(summary.split()))
                with col2:
                    compression_ratio = len(summary.split()) / len(video_data.transcript.split()) * 100
                    st.metric("📉 Compression", f"{compression_ratio:.1f}%")
                with col3:
                    st.metric("⏱️ Temps", f"{processing_time:.1f}s")
                
                # Résumé
                st.subheader("📋 Résumé")
                st.markdown(f"**{video_data.title}**")
                st.write(summary)
                
                # Sauvegarder dans l'historique
                summary_data = {
                    'title': video_data.title,
                    'summary': summary,
                    'model_type': model_type,
                    'length': summary_length,
                    'processing_time': processing_time,
                    'timestamp': time.strftime("%Y-%m-%d %H:%M:%S")
                }
                st.session_state.summary_history.append(summary_data)
                
                # Options d'export
                self.render_export_options(summary_data)
                
            except Exception as e:
                st.error(f"❌ Erreur lors de la génération : {e}")
                logger.error(f"Erreur résumé: {e}")
    
    def render_export_options(self, summary_data: Dict[str, Any]):
        """Affiche les options d'export"""
        st.subheader("💾 Export")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # Export TXT
            txt_content = f"""Titre: {summary_data['title']}
Date: {summary_data['timestamp']}
Modèle: {summary_data['model_type']}
Longueur: {summary_data['length']}

Résumé:
{summary_data['summary']}"""
            
            st.download_button(
                "📄 Télécharger TXT",
                txt_content,
                file_name=f"resume_{summary_data['timestamp'].replace(':', '-')}.txt",
                mime="text/plain"
            )
        
        with col2:
            # Export JSON
            import json
            json_content = json.dumps(summary_data, indent=2, ensure_ascii=False)
            
            st.download_button(
                "📊 Télécharger JSON",
                json_content,
                file_name=f"resume_{summary_data['timestamp'].replace(':', '-')}.json",
                mime="application/json"
            )
        
        with col3:
            # Copier dans le presse-papier (avec JavaScript)
            if st.button("📋 Copier"):
                st.write("Sélectionnez le texte ci-dessus et copiez-le (Ctrl+C)")
    
    def render_history(self):
        """Affiche l'historique des résumés"""
        if st.session_state.summary_history:
            st.header("📚 Historique des Résumés")
            
            for i, item in enumerate(reversed(st.session_state.summary_history)):
                with st.expander(f"📄 {item['title']} - {item['timestamp']}"):
                    col1, col2 = st.columns([3, 1])
                    
                    with col1:
                        st.write(item['summary'])
                    
                    with col2:
                        st.metric("Modèle", item['model_type'])
                        st.metric("Longueur", item['length'])
                        st.metric("Temps", f"{item['processing_time']:.1f}s")
            
            # Bouton pour vider l'historique
            if st.button("🗑️ Vider l'historique"):
                st.session_state.summary_history = []
                st.rerun()
    
    def render_stats(self):
        """Affiche les statistiques globales"""
        if self.model_manager:
            st.header("📈 Statistiques")
            
            try:
                stats = self.model_manager.get_stats()
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("📊 Total Requêtes", stats.get('total_requests', 0))
                
                with col2:
                    st.metric("🎯 LED", stats.get('led_requests', 0))
                
                with col3:
                    st.metric("⚡ OpenAI", stats.get('openai_requests', 0))
                
                with col4:
                    avg_time = stats.get('average_processing_time', 0)
                    st.metric("⏱️ Temps Moyen", f"{avg_time:.1f}s")
                
                # Graphique simple des requêtes
                if stats.get('total_requests', 0) > 0:
                    import matplotlib.pyplot as plt
                    
                    fig, ax = plt.subplots(figsize=(8, 4))
                    models = ['LED', 'OpenAI']
                    requests = [stats.get('led_requests', 0), stats.get('openai_requests', 0)]
                    
                    ax.bar(models, requests, color=['#1f77b4', '#ff7f0e'])
                    ax.set_ylabel('Nombre de requêtes')
                    ax.set_title('Utilisation des modèles')
                    
                    st.pyplot(fig)
                    
            except Exception as e:
                st.error(f"Erreur lors du chargement des statistiques: {e}")
    
    def run(self):
        """Lance l'application Streamlit"""
        self.render_header()
        
        # Barre latérale
        params = self.render_sidebar()
        
        # Contenu principal
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Entrée vidéo
            video_data = self.render_video_input()
            
            # Si on a des données vidéo
            if video_data:
                st.session_state.current_video_data = video_data
            
            # Afficher les infos et générer le résumé
            if st.session_state.current_video_data:
                self.render_video_info(st.session_state.current_video_data)
                self.render_summary_generation(st.session_state.current_video_data, params)
        
        with col2:
            # Historique et statistiques
            self.render_history()
            
            # Statistiques (si des modèles sont chargés)
            if self.model_manager:
                self.render_stats()


def main():
    """Point d'entrée principal"""
    try:
        app = VideoSummarizerApp()
        app.run()
    except Exception as e:
        st.error(f"Erreur critique : {e}")
        logger.error(f"Erreur critique: {e}")


if __name__ == "__main__":
    main()