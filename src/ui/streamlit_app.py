"""
Streamlit Interface for Video Summarizer
Modern web application for video summarization
"""

import streamlit as st
import os
import sys
import logging
import time
from pathlib import Path
from typing import Optional, Dict, Any

# Add src directory to PYTHONPATH
sys.path.append(str(Path(__file__).parent.parent))

try:
    from data.ingestion import DataIngestion, VideoData
    from data.preprocessing import TextPreprocessor
    from models.model_manager import ModelManager, ModelType, SummaryLength
    from monitoring.metrics import MetricsCollector
    from evaluation.evaluator import SummaryEvaluator
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
        self.model_manager = None
        
        # Initialize monitoring and evaluation
        try:
            self.metrics_collector = MetricsCollector()
            self.evaluator = SummaryEvaluator(load_models=False)  # Lazy loading
        except Exception as e:
            st.warning(f"Monitoring/Evaluation unavailable: {e}")
            self.metrics_collector = None
            self.evaluator = None
        
        # Application state
        if 'summary_history' not in st.session_state:
            st.session_state.summary_history = []
        
        if 'current_video_data' not in st.session_state:
            st.session_state.current_video_data = None
    
    def initialize_models(self):
        """Initialize model manager (lazy loading)"""
        if self.model_manager is None:
            with st.spinner("🔄 Initializing models..."):
                try:
                    config_path = Path(__file__).parent.parent.parent / "config" / "model_config.yaml"
                    self.model_manager = ModelManager(str(config_path) if config_path.exists() else None)
                    st.success("✅ Models initialized successfully!")
                except Exception as e:
                    st.error(f"❌ Error initializing models: {e}")
                    return False
        return True
    
    def render_header(self):
        """Display main header"""
        st.title("🎥 Video Summarizer")
        st.markdown("""
        **Transform your videos into intelligent summaries** with two model options:
        - 🎯 **LED** : Free & offline extractive summaries with precise content selection
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
            led_available, led_msg = self.model_manager.is_model_available(ModelType.LED)
            openai_available, openai_msg = self.model_manager.is_model_available(ModelType.OPENAI)
            
            if led_available:
                model_options.append("LED (Offline - Free)")
            else:
                model_options.append("LED (Unavailable)")
                
            if openai_available:
                model_options.append("OpenAI (Speed)")
            else:
                model_options.append("OpenAI (Unavailable)")
        else:
            model_options.extend(["LED (Quality - Free)", "OpenAI (Speed)"])
        
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
            **LED Fine-tuned:**
            - ✅ Long texts specialist
            - 🆓 Free & Offline
            - ⏱️ Slower (~30-200s)
            - 🇺🇸 Best for English
            - 📊 Extractive summaries
            
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
        
        # Quality warning if necessary
        if video_data.metadata and 'quality_warning' in video_data.metadata:
            st.warning(f"⚠️ {video_data.metadata['quality_warning']}")
            st.info("💡 **Tip**: Try using the OpenAI model for better summaries with this type of content.")
        elif video_data.metadata and 'quality_score' in video_data.metadata:
            quality_score = video_data.metadata['quality_score']
            if quality_score >= 0.7:
                st.success(f"✅ High quality transcript (score: {quality_score:.2f})")
            elif quality_score >= 0.5:
                st.info(f"ℹ️ Medium quality transcript (score: {quality_score:.2f})")
        
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
        if "LED" in params['model'] and "Unavailable" not in params['model']:
            model_type = "led"
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
                
                # Automatic summary evaluation
                evaluation_data = None
                if self.evaluator:
                    try:
                        with st.spinner("🎯 Evaluating quality..."):
                            # Load evaluation models if necessary
                            if not hasattr(self.evaluator, 'sentence_model') or self.evaluator.sentence_model is None:
                                self.evaluator._load_models()
                            
                            evaluation = self.evaluator.evaluate_summary(
                                original_text=processed_data.text,
                                generated_summary=summary,
                                model_name=model_type
                            )
                            
                            if evaluation and hasattr(evaluation, 'metrics'):
                                evaluation_data = evaluation.metrics
                                
                                # Display evaluation metrics
                                st.subheader("🎯 Quality Evaluation")
                                
                                col1, col2, col3 = st.columns(3)
                                with col1:
                                    st.metric("🏆 Overall Score", f"{evaluation_data.overall_score:.3f}")
                                with col2:
                                    st.metric("🧠 BERTScore", f"{evaluation_data.bert_score:.3f}")
                                with col3:
                                    st.metric("📏 Compression", f"{evaluation_data.compression_quality:.3f}")
                                
                                # Secondary metrics
                                col1 = st.columns(1)[0]
                                with col1:
                                    st.metric("🎯 Word Overlap (NER+Keywords)", f"{evaluation_data.word_overlap_ratio:.3f}")
                    except Exception as e:
                        st.warning(f"Evaluation unavailable: {e}")
                
                # Save to history
                summary_data = {
                    'title': video_data.title,
                    'summary': summary,
                    'model_type': model_type,
                    'length': summary_length,
                    'processing_time': processing_time,
                    'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
                    'evaluation': evaluation_data.__dict__ if evaluation_data else None
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
                # Icône selon la qualité (si évaluation disponible)
                quality_icon = "📄"
                if item.get('evaluation'):
                    score = item['evaluation'].get('overall_score', 0)
                    if score >= 0.8:
                        quality_icon = "🏆"
                    elif score >= 0.6:
                        quality_icon = "✅"
                    elif score >= 0.4:
                        quality_icon = "🟡"
                    else:
                        quality_icon = "🔴"
                
                with st.expander(f"{quality_icon} {item['title']} - {item['timestamp']}"):
                    col1, col2 = st.columns([3, 1])
                    
                    with col1:
                        st.write(item['summary'])
                        
                        # Display evaluation if available
                        if item.get('evaluation'):
                            eval_data = item['evaluation']
                            st.markdown("**📊 Quality:**")
                            sub_col1, sub_col2, sub_col3 = st.columns(3)
                            with sub_col1:
                                st.write(f"Score: {eval_data.get('overall_score', 0):.3f}")
                            with sub_col2:
                                st.write(f"BERT: {eval_data.get('bert_score', eval_data.get('semantic_similarity', 0)):.3f}")
                            with sub_col3:
                                st.write(f"WordOverlap: {eval_data.get('word_overlap_ratio', 0):.3f}")
                    
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
                    st.metric("🎯 LED", stats.get('led_requests', 0))
                
                with col3:
                    st.metric("⚡ OpenAI", stats.get('openai_requests', 0))
                
                with col4:
                    avg_time = stats.get('average_processing_time', 0)
                    st.metric("⏱️ Average Time", f"{avg_time:.1f}s")
                
                # Graphique simple des requêtes
                if stats.get('total_requests', 0) > 0:
                    import matplotlib.pyplot as plt
                    
                    fig, ax = plt.subplots(figsize=(8, 4))
                    models = ['LED', 'OpenAI']
                    requests = [stats.get('led_requests', 0), stats.get('openai_requests', 0)]
                    
                    ax.bar(models, requests, color=['#1f77b4', '#ff7f0e'])
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