"""
Streamlit Interface for Video Summarizer
Modern web application for video summarization
"""

import streamlit as st
import os
import sys
import logging
import time
import requests
import gc
from pathlib import Path
from typing import Optional, Dict, Any

# Add src directory to PYTHONPATH
sys.path.append(str(Path(__file__).parent.parent))

try:
    from data.ingestion import DataIngestion, VideoData
    from data.preprocessing import TextPreprocessor
    from models.model_manager import ModelManager, ModelType, SummaryLength
    from monitoring.metrics import MetricsCollector
except ImportError as e:
    st.error(f"Module import error: {e}")
    st.info("Make sure all dependencies are installed")
    st.stop()

# Page configuration
st.set_page_config(
    page_title="🎥 AI Video Summarizer",
    page_icon="🎥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Logging configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class VideoSummarizerApp:
    """Streamlit application for video summarization"""
    
    def __init__(self):
        self.ingestion = DataIngestion()
        self.preprocessor = TextPreprocessor()
        
        # Initialize monitoring
        try:
            self.metrics_collector = MetricsCollector()
        except Exception as e:
            st.warning(f"Monitoring unavailable: {e}")
            self.metrics_collector = None
        
        # Application state
        if 'summary_history' not in st.session_state:
            st.session_state.summary_history = []
        
        if 'current_video_data' not in st.session_state:
            st.session_state.current_video_data = None
    
    @property
    def model_manager(self):
        """Get cached model manager from session state"""
        if 'model_manager' not in st.session_state:
            st.session_state.model_manager = None
        return st.session_state.model_manager
    
    @model_manager.setter
    def model_manager(self, value):
        """Set model manager in session state"""
        st.session_state.model_manager = value
    
    def initialize_models(self):
        """Initialize model manager (cached in session state)"""
        if self.model_manager is None:
            with st.spinner("🔄 Initializing models..."):
                try:
                    config_path = Path(__file__).parent.parent.parent / "config" / "model_config.yaml"
                    self.model_manager = ModelManager(str(config_path) if config_path.exists() else None)
                    logger.info("✅ ModelManager cached in session state")
                    st.success("✅ Models initialized successfully!")
                except Exception as e:
                    st.error(f"❌ Error initializing models: {e}")
                    logger.error(f"Model initialization failed: {e}")
                    return False
        return True
    
    def cleanup_unused_models(self, active_model: str):
        """
        Free memory from unused models
        
        Args:
            active_model: The model currently being used ('led', 'openai', 'ollama')
        """
        if self.model_manager is None:
            return
        
        try:
            # Unload Ollama model if not in use
            if active_model != 'ollama' and self.model_manager._ollama_model:
                logger.info("🧹 Unloading Ollama model to free memory")
                self._unload_ollama_model()
                self.model_manager._ollama_model = None
            
            # LED support removed in this distribution; nothing to unload for LED
                
            # Force garbage collection
            import gc
            gc.collect()
            
        except Exception as e:
            logger.warning(f"Cleanup warning: {e}")
    
    def _unload_ollama_model(self):
        """Request Ollama to unload the current model from memory"""
        try:
            if self.model_manager._ollama_model and self.model_manager._ollama_model is not False:
                model_name = self.model_manager._ollama_model.model_name
                base_url = self.model_manager._ollama_model.base_url
                
                # Send keep_alive=0 to unload model
                response = requests.post(
                    f"{base_url}/api/generate",
                    json={
                        "model": model_name,
                        "keep_alive": 0  # Immediately unload
                    },
                    timeout=5
                )
                logger.info(f"✅ Ollama model {model_name} unloaded")
        except Exception as e:
            logger.warning(f"Could not unload Ollama model: {e}")
    
    def _clear_all_models(self):
        """Clear all loaded models from memory"""
        if self.model_manager is None:
            return
        
        try:
            # Unload Ollama
            self._unload_ollama_model()
            
            # Clear model references
            self.model_manager._ollama_model = None
            self.model_manager._openai_model = None
            
            # Force garbage collection
            gc.collect()
            
            logger.info("🧹 All models cleared from memory")
        except Exception as e:
            logger.warning(f"Error clearing models: {e}")
    
    def render_header(self):
        """Display main header"""
        st.title("🎥 Video Summarizer")
    st.markdown("""
    **Transform your videos into intelligent summaries** with two model options:
    - 🆕 **Ollama** : Local LLM (Gemma3, Qwen) - Fast, free, offline
    - ⚡ **OpenAI GPT** : Fast abstractive summaries with enhanced coherence evaluation
        
    *Choose your source, configure your preferences and get professional summaries in just a few clicks!*
    """)
    
    def render_sidebar(self):
        """Display sidebar with settings"""
        st.sidebar.header("⚙️ Settings")
        
        # Model selection with availability check
        model_options = ["Auto (Recommended)"]
        
        # Check model availability
        if self.model_manager:
            from models.model_manager import ModelType
            openai_available, openai_msg = self.model_manager.is_model_available(ModelType.OPENAI)
            ollama_available, ollama_msg = self.model_manager.is_model_available(ModelType.OLLAMA)
            
            if ollama_available:
                model_options.append("Ollama (Local - Free)")
            else:
                model_options.append("Ollama (Unavailable)")
                
            if openai_available:
                model_options.append("OpenAI (Speed)")
            else:
                model_options.append("OpenAI (Unavailable)")
        else:
            model_options.extend(["Ollama (Local - Free)", "OpenAI (Speed)"])
        
        model_option = st.sidebar.selectbox(
            "🤖 Summary Model",
            model_options,
            help="Auto automatically selects the best available model"
        )
        
        # Summary length selection
        length_option = st.sidebar.selectbox(
            "📏 Summary Length",
            ["Long (200-500 words)", "Short (50-200 words)"],
            help="Approximate length of the generated summary"
        )
        
        # Language
        language_option = st.sidebar.selectbox(
            "🌍 Language",
            ["Auto-detect", "English", "French", "Spanish", "German"],
            help="Language of the generated summary"
        )
        
        # Memory Management Section
        with st.sidebar.expander("🧹 Memory Management"):
            st.markdown("""
            **Clear model cache to free RAM/GPU memory**
            
            Use this if the app becomes slow or crashes.
            """)
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button("🗑️ Clear All", key="clear_all"):
                    self._clear_all_models()
                    st.success("✅ All models cleared")
            
            with col2:
                if st.button("🔄 Unload Ollama", key="unload_ollama"):
                    self._unload_ollama_model()
                    st.success("✅ Ollama unloaded")
            
            st.caption("💡 Models will reload automatically when needed")
        
        # System monitoring
        if self.metrics_collector:
            with st.sidebar.expander("📊 System Monitoring"):
                try:
                    metrics = self.metrics_collector._collect_system_metrics()
                    st.metric("💻 CPU", f"{metrics.cpu_percent:.1f}%")
                    st.metric("🧠 Memory", f"{metrics.memory_percent:.1f}%")
                    st.metric("💾 Disk", f"{metrics.disk_usage_percent:.1f}%")
                except Exception as e:
                    st.warning("Metrics unavailable")
        
        # Model information
        with st.sidebar.expander("ℹ️ Model Information"):
            st.markdown("""
            **Ollama (Local LLM):**
            - ⚡ Very fast (~3-6s)
            - 🆓 100% Free
            - 🏠 Offline & Private
            - 🌍 Multi-language
            - 🎨 Abstractive summaries
            - 💾 Low RAM (2GB)
            
            **OpenAI GPT:**
            - ✅ Very fast (~5-15s)
            - ✅ Multi-language
            - 💰 Cost per usage
            - 🌐 Requires internet
            - 🎨 Abstractive summaries
            """)
        
        return {
            'model': model_option,
            'length': length_option,
            'language': language_option
        }
    
    def render_video_input(self):
        """Display video input options"""
        st.header("📹 Video Source")
        
        # Tabs for different sources
        tab1, tab2, tab3 = st.tabs(["🔗 YouTube", "📁 Local File", "📝 Direct Text"])
        
        video_data = None
        
        with tab1:
            st.subheader("YouTube")
            youtube_url = st.text_input(
                "YouTube URL:",
                placeholder="https://www.youtube.com/watch?v=...",
                help="Paste a YouTube video URL"
            )
            
            language_pref = st.selectbox(
                "Preferred subtitle language:",
                ["Auto", "English", "French", "Spanish", "German"],
                help="Language of subtitles to extract"
            )
            
            if st.button("📥 Extract Transcript", key="youtube"):
                if youtube_url.strip():
                    try:
                        with st.spinner("🔄 Extracting transcript..."):
                            video_data = self.ingestion.process_youtube_url(youtube_url)
                            st.success(f"✅ Transcript extracted: {video_data.title}")
                            return video_data
                    except Exception as e:
                        st.error(f"❌ Error during extraction: {str(e)}")
                else:
                    st.warning("⚠️ Please enter a YouTube URL")
        
        with tab2:
            st.subheader("Local File")
            uploaded_file = st.file_uploader(
                "Choose an audio/video file:",
                type=['mp4', 'avi', 'mov', 'mp3', 'wav', 'm4a'],
                help="Supported formats: MP4, AVI, MOV, MP3, WAV, M4A"
            )
            
            if uploaded_file and st.button("� Transcribe Audio", key="local"):
                st.warning("🚧 Feature in development (requires Whisper)")
                # TODO: Implement local transcription with Whisper
        
        with tab3:
            st.subheader("Direct Text")
            direct_text = st.text_area(
                "Paste your text here:",
                height=200,
                placeholder="Paste the transcript or text you want to summarize...",
                help="Raw text to summarize directly"
            )
            
            custom_title = st.text_input(
                "Title (optional):",
                placeholder="Title for your text"
            )
            
            if st.button("📝 Use This Text", key="direct"):
                if direct_text.strip():
                    video_data = self.ingestion.process_text_input(
                        direct_text, 
                        custom_title or "Custom Text"
                    )
                    st.success("✅ Text ready for summary")
                else:
                    st.warning("⚠️ Please enter some text")
        
        return video_data
    
    def render_video_info(self, video_data: VideoData):
        """Display video information"""
        st.header("📊 Content Information")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("📝 Title", value="", delta=video_data.title)
        
        with col2:
            word_count = len(video_data.transcript.split())
            st.metric("📊 Words", word_count)
        
        with col3:
            st.metric("🌍 Language", video_data.language.upper())
        
        with col4:
            if video_data.duration:
                duration_min = video_data.duration // 60
                st.metric("⏱️ Duration", f"{duration_min}min")
            else:
                st.metric("📄 Source", video_data.source)
        
        # Transcript preview
        with st.expander("👁️ Preview Transcript"):
            preview_length = min(500, len(video_data.transcript))
            st.text_area(
                "Transcript (first 500 characters):",
                video_data.transcript[:preview_length] + ("..." if len(video_data.transcript) > preview_length else ""),
                height=150,
                disabled=True
            )
    
    def render_summary_generation(self, video_data: VideoData, params: Dict[str, str]):
        """Display summary generation section"""
        st.header("🎯 Summary Generation")
        
        if not self.initialize_models():
            return
        
        # Configure parameters
        model_type = "auto"
        if "Ollama" in params['model'] and "Unavailable" not in params['model']:
            model_type = "ollama"
        elif "OpenAI" in params['model'] and "Unavailable" not in params['model']:
            model_type = "openai"
        # If unavailable model selected, use auto
        
        summary_length = "short" if "Short" in params['length'] else "long"
        
        language = None
        if params['language'] != "Auto-detect":
            if params['language'] == "English":
                language = "english"
            elif params['language'] == "French":
                language = "french"
            elif params['language'] == "Spanish":
                language = "spanish"
            elif params['language'] == "German":
                language = "german"
        
        # Generation button
        if st.button("🚀 Generate Summary", type="primary", use_container_width=True):
            try:
                start_time = time.time()
                
                with st.spinner("🔄 Generating summary..."):
                    # Cleanup unused models BEFORE generation
                    self.cleanup_unused_models(model_type)
                    
                    # Preprocessing
                    processed_data = self.preprocessor.preprocess(video_data.transcript)
                    
                    # Summary generation
                    summary = self.model_manager.summarize_simple(
                        text=processed_data.text,
                        model_type=model_type,
                        summary_length=summary_length,
                        language=language
                    )
                
                processing_time = time.time() - start_time
                
                # Display summary
                st.success(f"✅ Summary generated in {processing_time:.1f}s")
                
                # Summary metrics
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("📊 Words", len(summary.split()))
                with col2:
                    compression_ratio = len(summary.split()) / len(video_data.transcript.split()) * 100
                    st.metric("📉 Compression", f"{compression_ratio:.1f}%")
                with col3:
                    st.metric("⏱️ Time", f"{processing_time:.1f}s")
                
                # Summary
                st.subheader("📋 Summary")
                st.markdown(f"**{video_data.title}**")
                st.write(summary)
                
                # Save to history
                summary_data = {
                    'title': video_data.title,
                    'summary': summary,
                    'model_type': model_type,
                    'length': summary_length,
                    'processing_time': processing_time,
                    'timestamp': time.strftime("%Y-%m-%d %H:%M:%S")
                }
                st.session_state.summary_history.append(summary_data)
                
                # Export options
                self.render_export_options(summary_data)
                
            except Exception as e:
                st.error(f"❌ Error during generation: {e}")
                logger.error(f"Summary error: {e}")
    
    def render_export_options(self, summary_data: Dict[str, Any]):
        """Display export options"""
        st.subheader("💾 Export")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # Export TXT
            txt_content = f"""Title: {summary_data['title']}
Date: {summary_data['timestamp']}
Model: {summary_data['model_type']}
Length: {summary_data['length']}

Summary:
{summary_data['summary']}"""
            
            st.download_button(
                "📄 Download TXT",
                txt_content,
                file_name=f"summary_{summary_data['timestamp'].replace(':', '-')}.txt",
                mime="text/plain"
            )
        
        with col2:
            # Export JSON
            import json
            json_content = json.dumps(summary_data, indent=2, ensure_ascii=False)
            
            st.download_button(
                "📊 Download JSON",
                json_content,
                file_name=f"summary_{summary_data['timestamp'].replace(':', '-')}.json",
                mime="application/json"
            )
        
        with col3:
            # Copy to clipboard (with JavaScript)
            if st.button("📋 Copy"):
                st.write("Select the text above and copy it (Ctrl+C)")
    
    def render_history(self):
        """Display summary history"""
        if st.session_state.summary_history:
            st.header("📚 Summary History")
            
            for i, item in enumerate(reversed(st.session_state.summary_history)):
                with st.expander(f"📄 {item['title']} - {item['timestamp']}"):
                    col1, col2 = st.columns([3, 1])
                    
                    with col1:
                        st.write(item['summary'])
                    
                    with col2:
                        st.metric("Model", item['model_type'])
                        st.metric("Length", item['length'])
                        st.metric("Time", f"{item['processing_time']:.1f}s")
            
            # Button to clear history
            if st.button("🗑️ Clear History"):
                st.session_state.summary_history = []
                st.rerun()
    
    def render_stats(self):
        """Display global statistics"""
        if self.model_manager:
            st.header("📈 Statistics")
            
            try:
                stats = self.model_manager.get_stats()
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("📊 Total Requests", stats.get('total_requests', 0))
                
                with col2:
                    st.metric("🆕 Ollama", stats.get('ollama_requests', 0))
                
                with col3:
                    st.metric("⚡ OpenAI", stats.get('openai_requests', 0))
                
                with col4:
                    avg_time = stats.get('average_processing_time', 0)
                    st.metric("⏱️ Average Time", f"{avg_time:.1f}s")
                
                # Graphique simple des requêtes
                if stats.get('total_requests', 0) > 0:
                    import matplotlib.pyplot as plt
                    
                    fig, ax = plt.subplots(figsize=(10, 4))
                    models = ['Ollama', 'OpenAI']
                    requests = [
                        stats.get('ollama_requests', 0),
                        stats.get('openai_requests', 0)
                    ]
                    
                    ax.bar(models, requests, color=['#2ecc71', '#ff7f0e'])
                    ax.set_ylabel('Number of requests')
                    ax.set_title('Model usage')
                    
                    st.pyplot(fig)
                    
            except Exception as e:
                st.error(f"Error loading statistics: {e}")
    
    def run(self):
        """Launch the Streamlit application"""
        self.render_header()
        
        # Sidebar
        params = self.render_sidebar()
        
        # Main content
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Video input
            video_data = self.render_video_input()
            
            # If we have video data
            if video_data:
                st.session_state.current_video_data = video_data
            
            # Display info and generate summary
            if st.session_state.current_video_data:
                self.render_video_info(st.session_state.current_video_data)
                self.render_summary_generation(st.session_state.current_video_data, params)
        
        with col2:
            # History and statistics
            self.render_history()
            
            # Statistics (if models are loaded)
            if self.model_manager:
                self.render_stats()


def main():
    """Main entry point"""
    try:
        app = VideoSummarizerApp()
        app.run()
    except Exception as e:
        st.error(f"Critical error: {e}")
        logger.error(f"Critical error: {e}")


if __name__ == "__main__":
    main()