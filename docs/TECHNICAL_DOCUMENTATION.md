# Video Summarizer - Technical Overview

**Core Technologies:**
- **Backend:** Python 3.8+, PyTorch, Transformers (Hugging Face), FastAPI
- **AI Models:** LED (Longformer Encoder-Decoder), OpenAI GPT-4/3.5-turbo
- **UI/UX:** Streamlit, HTML/CSS, Plotly (dashboards)
- **Data:** SQLite, YAML, JSON, pandas
- **NLP:** spaCy, sentence-transformers, semantic similarity
- **Infrastructure:** MPS (Apple Silicon), CUDA, Docker-ready
- **Monitoring:** Real-time metrics, alerting, performance tracking

**Key Concepts:** #NLP #MachineLearning #DeepLearning #REST-API #Microservices #PerformanceOptimization #CacheManagement #Summarization #TransformerModels

## 1. Project Overview

### 1.1 Purpose and Scope

Video Summarizer is an AI system that automatically creates summaries from video content. The system extracts transcripts from various sources and processes them using natural language models to generate clear, concise summaries.

This project tackles the problem of information overload by helping users understand video content without watching entire recordings. It's useful for education, work, and research where time is limited but content understanding is essential.

### 1.2 Core Functionality

The system processes multiple input types through a unified pipeline:

**Input Sources:**
- **YouTube Videos**: Direct URL processing with automatic transcript extraction
- **Local Media Files**: MP4, AVI, MOV, MP3, WAV, M4A support
- **Direct Text**: Raw text input for immediate processing

**Dual AI Model Strategy:**
- **LED Model** (`allenai/led-base-16384`): 
  - ✅ 100% offline, no costs
  - ✅ Handles up to 16K tokens natively
  - ✅ Optimized for Apple Silicon (MPS)
  - ⚠️ English-focused, slower processing

- **OpenAI GPT Models** (GPT-4/3.5-turbo):
  - ✅ Fast processing (2-3 seconds)
  - ✅ Excellent multilingual support
  - ✅ High-quality abstractive summaries
  - ⚠️ API costs, internet required

```python
class ModelManager:
    def select_optimal_model(self, request: SummaryRequest) -> ModelType
    def is_model_available(self, model_type: ModelType) -> bool
    def get_model_stats(self) -> Dict[str, Any]
```

### 1.3 Technical Architecture

The project uses a modular design where each part handles a specific job. The main parts are data ingestion, text processing, model management, quality checking, and system monitoring.

You can use it through a web interface built with Streamlit or connect to it programmatically using REST API endpoints. This makes it work both for direct users and for integration into other applications.

```python
# Main application structure
src/
├── data/           # Handle input data
├── models/         # AI model management  
├── evaluation/     # Quality assessment
├── monitoring/     # System tracking
├── api/           # REST endpoints
└── ui/            # Web interfaces
```

### 1.4 Key Features

The system checks transcript quality before processing and gives confidence scores for the summaries it creates. If something goes wrong with the main model, backup systems kick in automatically.

Real-time monitoring keeps track of how the system is performing and how people are using it. The quality checking uses reliable metrics including BERTScore for semantic similarity, sentence coherence for flow assessment, and compression quality for optimal summarization ratios.

```python
@dataclass
class EvaluationMetrics:
    bert_score: float              # BERTScore - Contextual semantic similarity
    compression_quality: float     # Compression with quality control
    word_overlap_ratio: float      # Important words overlap
    sentence_coherence: float      # True coherence between sentences
    overall_score: float          # Simplified global score
    processing_time: float        # Generation time in seconds
    model_used: str               # Name of the model used
    timestamp: str                # Evaluation timestamp
```

The system is optimized for Apple Silicon chips (M1, M2, M3) using Metal Performance Shaders for faster processing.

## 2. Data Ingestion and Processing Pipeline

### 2.1 Multi-Source Data Ingestion

The system can handle different types of input sources. For YouTube videos, it uses the YouTube Transcript API to get existing transcripts in various languages. If the first language doesn't work, it tries others automatically.

```python
class YouTubeTranscriptExtractor:
    def get_transcript(self, video_url, language=None)
    def extract_video_id(self, url)
```

For local files, it supports common video formats (MP4, AVI, MOV, MKV, WebM) and audio formats (MP3, WAV, M4A). 

**Note:** Currently, the system focuses on text-based processing. Audio-to-text transcription using Whisper models is planned for future implementation. The current version works best with pre-existing transcripts or text content.

The system handles errors well - network problems, API limits, and corrupted files don't crash it. It also extracts useful information like video length, language, and quality indicators.

### 2.2 Text Preprocessing Pipeline

Raw transcripts need cleaning before they can be summarized effectively. The cleaning process removes YouTube-specific stuff like timestamps `[00:15]`, speaker labels, music tags `[Music]`, and applause markers that don't help with understanding the content.

```python
class TextCleaner:
    def clean_text(self, text)
    def normalize_text(self, text)
```

The system also removes URLs, email addresses, and repeated characters while keeping the actual content intact. It fixes punctuation and spacing to make everything consistent.

Quality checking looks at how coherent the transcript is - checking if words make sense, if sentences are properly formed, and if the vocabulary is diverse enough. Poor quality transcripts get flagged with warnings.

### 2.3 Content Segmentation and Validation

The preprocessing splits text intelligently, keeping sentences together while staying within model limits. The LED model can handle up to 16,384 tokens, while OpenAI models have different limits depending on which version you're using.

The system checks content quality by looking at language patterns and vocabulary consistency. If content quality is too low, it rejects it and suggests alternatives.

Language detection happens automatically so the system can pick the right model and use appropriate prompts. It works mainly with French and English but can be extended to other languages.

### 2.4 Caching and Performance Optimization

The system caches processed transcripts using content hashes to avoid doing the same work twice. The cache has configurable retention rules and cleans itself up automatically to manage disk space.

```python
class DataIngestion:
    def get_cached_transcript(self, content_hash)
    def cache_transcript(self, content_hash, transcript_data)
```

Performance improvements include streaming for large files, parallel processing for multiple items, and efficient memory usage. The system tracks processing times and resource usage to spot where it can do better.

## 3. AI Models and Implementation

### 3.1 LED Model Architecture

The LED (Longformer Encoder-Decoder) model is the core of the offline summarization system. It's built on transformer architecture but modified to handle very long documents - up to 16,384 tokens compared to typical models that max out around 512 tokens.

```python
class LEDSummarizer:
    def __init__(self, model_name, device)
    def _get_device(self, device)
    def summarize(self, text, summary_type)
```

The model uses attention patterns that scale efficiently with document length. Instead of computing attention between every token pair (which would be expensive), it uses a sliding window approach combined with global attention on key tokens.

For Apple Silicon chips (M1, M2, M3), the system uses Metal Performance Shaders (MPS) which can make processing 1.5x faster than CPU while using less memory and generating less heat.

### 3.2 OpenAI Integration

The OpenAI integration provides fast, high-quality summaries through GPT-4 and GPT-3.5-turbo models. The system uses carefully crafted prompts that adapt based on the desired summary length and language.

```python
class OpenAISummarizer:
    def __init__(self, api_key, model_name)
    def summarize(self, text, summary_type, language)
    def _track_usage(self, tokens, cost)
```

The system tracks API usage including token consumption and costs. It also implements automatic fallback to GPT-3.5-turbo if GPT-4 is unavailable or rate-limited.

### 3.3 Model Selection Logic

The model manager automatically chooses the best model based on several factors. For long English texts, LED often works better. For multilingual content or when speed is critical, OpenAI models are preferred.

```python
class ModelManager:
    def select_optimal_model(self, request)
    def _get_fallback_model(self)
    def generate_summary(self, request)
```

The system also checks model availability in real-time. If the LED model fails to load (insufficient memory, missing dependencies), it automatically switches to OpenAI. If OpenAI is unavailable (no API key, network issues), it falls back to LED.

### 3.4 Generation Parameters and Quality Control

Both models use carefully tuned generation parameters to ensure quality output. The LED model uses beam search with specific penalties to avoid repetition and encourage conciseness.

```python
# LED generation configuration
generation_config = {
    'num_beams': 4, 'length_penalty': 2.0, 'repetition_penalty': 1.3,
    'no_repeat_ngram_size': 4, 'early_stopping': True
}
```

The system includes quality validation for generated summaries. It checks for coherence, proper sentence structure, and meaningful content. Low-quality outputs trigger regeneration with different parameters or fallback to the alternative model.


## 4. System Architecture and Components

### 4.1 Web Interface (Streamlit)

The main user interface is built with Streamlit, providing an intuitive web application for video summarization. The interface handles multiple input types and provides real-time feedback during processing.

```python
class VideoSummarizerApp:
    def __init__(self)
    def render_main_interface(self)
    def process_youtube_video(self, url)
    def process_local_file(self, file)
```

The interface includes progress bars, quality indicators, and error handling with user-friendly messages. It also provides model comparison features and summary history tracking.

### 4.2 REST API (FastAPI)

The system exposes a comprehensive REST API for programmatic access. The API supports text summarization, YouTube processing, batch operations, and system monitoring.

```python
app = FastAPI(
    title="Video Summarizer API",
    description="Professional API for automatic video summarization",
    version="2.0.0"
)

@app.post("/api/v1/summarize/text")
async def summarize_text(request: TextSummaryRequest)

@app.post("/api/v1/summarize/youtube") 
async def summarize_youtube(request: YouTubeSummaryRequest)
```

The API includes authentication, rate limiting, and comprehensive error handling. It also provides endpoints for health checks, model status, and system metrics.

### 4.3 Monitoring Dashboard

A separate monitoring dashboard tracks system performance, user activity, and model effectiveness. It's built as a standalone Streamlit application with real-time charts and alerts.

```python
class MonitoringDashboard:
    def render_system_metrics(self)
    def render_performance_charts(self)
    def render_model_usage_stats(self)
```

The dashboard shows processing times, error rates, model performance comparisons, and user behavior patterns. It also includes alert management for system issues.

### 4.4 Database and Storage

The system uses SQLite for metrics storage and file-based caching for processed content. The database schema tracks performance metrics, user interactions, and system events.

The database schema includes tables for system metrics, performance tracking, and user interactions.

```python
class MetricsCollector:
    def record_performance_metric(self, operation, model_name, processing_time)
    def get_system_metrics(self)
    def get_business_metrics(self)
```

The storage system includes automatic cleanup, backup capabilities, and data export functions for analysis.

### 4.5 Configuration Management

The system uses YAML configuration files for flexible parameter management. Configuration is split between application settings and model parameters.

Configuration uses YAML files for app settings, UI parameters, API configuration, and monitoring options.

Configuration changes are detected automatically and applied without requiring system restart for most parameters.



## 5. Evaluation and Quality Assurance

### 5.1 Automatic Quality Assessment

The system includes a simplified and reliable evaluation framework that automatically assesses summary quality using three main metrics: BERTScore for contextual semantic similarity, sentence coherence for text flow assessment, and compression quality for optimal summarization ratios.

```python
@dataclass
class EvaluationMetrics:
    bert_score: float              # BERTScore - Contextual semantic similarity (0.6+ = good, 0.8+ = excellent)
    compression_quality: float     # Compression with quality control (0.7-0.9 = optimal)
    word_overlap_ratio: float      # Important words overlap
    sentence_coherence: float      # Coherence between sentences
    overall_score: float          # Weighted average of all metrics
```

The evaluation happens automatically when users request it, providing immediate feedback on summary quality with clear, interpretable scores.

### 5.2 BERTScore and Sentence Coherence

BERTScore measures contextual semantic similarity by comparing sentence embeddings between the original text and summary. Unlike traditional word-overlap metrics, BERTScore captures semantic meaning even when different words are used.

```python
class SummaryEvaluator:
    def _calculate_bert_score(self, original: str, summary: str) -> float
    def _calculate_sentence_coherence(self, text: str) -> float
    def _calculate_compression_quality(self, original: str, summary: str) -> float
```

Sentence coherence measures the thematic continuity between consecutive sentences in the summary, ensuring the generated text flows naturally and maintains logical progression throughout the content.

### 5.3 Compression Quality Assessment

The system evaluates compression quality by analyzing the ratio between original and summary text, ensuring optimal summarization that's neither too brief (information loss) nor too verbose (insufficient synthesis).

Compression quality combines ratio optimization with information density measurement, rewarding summaries that maintain high informative content while achieving appropriate length reduction.

### 5.4 Quality Scoring and Interpretation

The system uses clear quality thresholds to categorize summary performance across all metrics.

The overall score provides a single reliable metric combining all evaluation dimensions, with scores above 0.7 indicating high-quality summaries and scores above 0.5 indicating acceptable quality. The enhanced weight on sentence coherence (35%) makes the system more suitable for evaluating abstractive summaries that use different words but maintain semantic meaning and logical flow.


### 5.5 Model Performance Evaluation

The evaluation system automatically tracks model performance across different metrics, enabling data-driven model selection:

```python
class SummaryEvaluator:
    def evaluate_summary(self, original_text: str, generated_summary: str, 
                        model_name: str, processing_time: float) -> EvaluationReport
```

**Performance Tracking:**
- BERTScore effectiveness by model type
- Sentence coherence assessment for abstractive vs extractive summaries
- Processing time and resource usage
- Compression quality optimization



## 6. Monitoring and Performance

### 6.1 Real-Time System Monitoring

The system includes comprehensive monitoring that tracks both technical performance and business metrics. It monitors CPU, memory, disk usage, and GPU utilization in real-time to ensure optimal performance.

```python
@dataclass
class SystemMetrics:
    timestamp: str
    cpu_percent: float            # CPU usage percentage
    memory_percent: float         # RAM usage percentage
    memory_used_mb: float        # Used memory in MB
    disk_usage_percent: float    # Disk usage percentage
```

The collector runs in a background thread and stores metrics in SQLite for historical analysis. It can detect resource bottlenecks and performance degradation automatically.

### 6.2 Performance Tracking

Every summarization request is tracked with detailed performance metrics including processing time, input/output lengths, model used, and success status.

```python
@dataclass 
class PerformanceMetrics:
    timestamp: str
    model_name: str             # Model used for processing
    processing_time: float      # Time taken in seconds
    input_length: int           # Input text length in tokens
    output_length: int          # Output summary length in tokens
    success: bool               # Operation success status
```

This tracking helps identify slow operations, memory leaks, and performance patterns across different content types and models.



### 6.3 Performance Optimization

The monitoring data is used to automatically optimize system performance. The system can adjust model parameters, manage memory usage, and balance load based on observed patterns.

```python
class PerformanceOptimizer:
    def optimize_model_selection(self)
    def manage_memory_usage(self)
    def _clear_model_caches(self)
```

The optimizer runs periodically and makes automatic adjustments to maintain optimal performance without manual intervention.

### 6.4 Dashboard and Reporting

All monitoring data is visualized through interactive dashboards that provide both real-time views and historical analysis.

```python
def render_performance_dashboard(self):
    # Real-time metrics with Streamlit
    # Performance charts with Plotly
```

The dashboard includes charts for processing times, error rates, resource usage, model performance comparisons, and user activity patterns.


## 7. Current Implementation and Deployment

### 7.1 Local Development Setup

The system is currently implemented for local development and testing. It includes automated installation scripts that handle environment creation and dependency installation:

- `scripts/install.py`: Automated installer that creates virtual environment and installs dependencies
- `scripts/launch.py`: Cross-platform launcher with cache clearing functionality
- `scripts/setup_led.py`: LED model-specific configuration
- `requirements.txt`: Complete dependency specification

The installation process creates a Python virtual environment, installs all required packages (PyTorch, Transformers, Streamlit, FastAPI), and configures the system for the target platform.

### 7.2 Platform Optimization

The system includes implemented optimizations for different hardware platforms:

- **Apple Silicon Support**: Automatic detection and use of Metal Performance Shaders (MPS) for M1/M2/M3 chips
- **CUDA Support**: Automatic GPU detection for NVIDIA hardware when available
- **CPU Fallback**: Efficient CPU processing when GPU acceleration is unavailable

Device selection is handled automatically through the `LEDSummarizer._get_device()` method.

### 7.3 Configuration Management

The system uses YAML-based configuration files for flexible parameter management:

- `config/app_config.yaml`: Application settings, UI parameters, API configuration
- `config/model_config.yaml`: Model parameters, generation settings, training configuration

Configuration is managed through the `ConfigManager` class with support for nested parameter access and default values.

### 7.4 Data Storage

Current implementation uses:

- **SQLite database**: For metrics storage and system monitoring data
- **File-based caching**: Content-based hashing for transcript caching
- **Local file storage**: For temporary files and processed content

The database schema includes tables for system metrics, performance tracking, and business metrics.

### 7.5 Monitoring Implementation

The system includes basic monitoring capabilities:

- Real-time system metrics collection (CPU, memory, disk usage)
- Performance tracking for summarization requests
- Basic alerting system for resource thresholds
- Monitoring dashboard built with Streamlit

Monitoring is handled by the `MetricsCollector` class with SQLite storage and configurable retention policies.

## 7.6 Next Steps and Future Enhancements

### Production Deployment
- **Container Deployment**: Docker containerization with multi-stage builds
- **Cloud Deployment**: Support for AWS, Azure, GCP with container orchestration
- **Database Migration**: PostgreSQL/MySQL support for production environments

### Advanced Monitoring
- **External Integration**: Prometheus/Grafana integration
- **Performance Analytics**: Advanced performance profiling and optimization
- **Business Intelligence**: Enhanced usage analytics and reporting

### Scalability Improvements
- **Model Serving**: Distributed model serving across multiple instances
- **Caching Strategy**: Redis/Memcached integration for improved performance
- **Queue System**: Asynchronous processing with task queues

## 8. Testing Guide

### 8.1 Repository Access

The Video Summarizer project is available on GitHub at:
```
https://github.com/faridgnank02/video-summarizer
```

Clone the repository to get started with testing:
```bash
git clone https://github.com/faridgnank02/video-summarizer.git
cd video-summarizer
```

### 8.2 Quick Installation and Setup

The project includes automated installation scripts for easy setup:

```bash
# Run the automated installer
python scripts/install.py

# Activate the virtual environment
source video-summarizer-env/bin/activate    # Linux/Mac
# video-summarizer-env\Scripts\activate     # Windows

# Optional: Configure OpenAI API key
echo "OPENAI_API_KEY=your-api-key-here" >> .env

# Launch the application
python scripts/launch.py
```

The installer automatically creates a virtual environment, installs all dependencies, and configures the system for your platform.

### 8.3 Testing the Web Interface

Once launched, the Streamlit web interface is available at `http://localhost:8501`. Test the main features:

**YouTube Video Processing:**
1. Go to the "YouTube" tab
2. Paste a YouTube URL (try: https://www.youtube.com/watch?v=dQw4w9WgXcQ)
3. Select transcript language (French/English)
4. Choose model (LED for free offline, OpenAI for speed)
5. Select summary length (Short/Long)
6. Click "Generate Summary"

**Local File Upload:**
1. Switch to "Local Files" tab
2. Upload a video or audio file (MP4, MP3, etc.)
3. Wait for transcript extraction
4. Generate summary with preferred settings

**Direct Text Input:**
1. Use "Direct Text" tab
2. Paste or type text content
3. Configure summarization options
4. Generate and evaluate results

### 8.4 API Testing

Test the REST API endpoints using curl or API testing tools:

```bash
# Health check
curl http://localhost:8000/health

# Text summarization
curl -X POST http://localhost:8000/api/v1/summarize/text \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Your long text here...",
    "model_type": "auto",
    "summary_length": "short",
    "evaluate": true
  }'

# YouTube summarization  
curl -X POST http://localhost:8000/api/v1/summarize/youtube \
  -H "Content-Type: application/json" \
  -d '{
    "url": "https://www.youtube.com/watch?v=example",
    "model_type": "led",
    "summary_length": "long"
  }'
```

API documentation is available at `http://localhost:8000/docs` when the API server is running.

### 8.5 Running Automated Tests

The project includes comprehensive test suites for validation:

```bash
# Run all tests
python -m pytest tests/

# Specific test categories
python tests/test_functionality.py    # Core functionality
python tests/test_api.py             # API endpoints
python tests/test_integration.py     # Integration tests
python tests/test_architecture.py    # Architecture validation
```

The test suite covers model functionality, API endpoints, data processing, and system integration.

### 8.6 Monitoring Dashboard

Access the monitoring dashboard to view system performance:

```bash
# Launch monitoring dashboard
python -c "from src.ui.monitoring_dashboard import MonitoringDashboard; MonitoringDashboard().render_dashboard()"
```

The dashboard shows real-time metrics, processing statistics, model performance, and system health indicators.

### 8.7 Performance Testing

Test system performance with different content types and models:

**LED Model Testing:**
- Test with long documents (>5000 words)
- Check memory usage during processing
- Measure processing times for different lengths

**OpenAI Model Testing:**
- Test multilingual content processing
- Verify API key configuration
- Check cost tracking functionality  
- Test fallback mechanisms

**Quality Evaluation:**
- Enable evaluation mode in interface
- Test semantic similarity measurements
- Verify coherence assessments

### 8.8 Troubleshooting Common Issues

**Model Loading Issues:**
- **RAM Requirements**: 8GB minimum, 16GB recommended for LED model
- **Disk Space**: 4GB free space for model cache and temporary files
- **GPU Compatibility**: 
  - Apple Silicon: M1/M2/M3 with MPS support
  - NVIDIA: CUDA 11.8+ compatible cards
  - Intel: XPU support (experimental)
- **Network**: Required for initial model download (~2GB for LED base model)
- **Python Version**: 3.8+ required, 3.10+ recommended

**API Configuration:**
- Confirm OpenAI API key is valid
- Check API rate limits and quotas
- Verify network connectivity

**Performance Issues:**
- Monitor system resources during processing
- Check for memory leaks in long sessions
- Verify optimal device selection (GPU vs CPU)

The project includes detailed logging to help diagnose issues. Check log files in the application directory for troubleshooting information.

---

## 9. Project Status and Roadmap

### 9.1 Current Implementation Status

**✅ Fully Implemented:**
- LED and OpenAI model integration
- Streamlit web interface
- FastAPI REST endpoints
- Basic monitoring and metrics
- YouTube transcript extraction
- Text preprocessing pipeline
- ROUGE-based evaluation

**🔄 Partially Implemented:**
- Dashboard monitoring (basic version)
- Caching system (file-based only)
- Error handling (basic coverage)
- Configuration management (YAML-based)

**⏳ Planned Features:**
- Audio-to-text transcription (Whisper integration)
- Advanced A/B testing framework
- Distributed model serving


### 9.2 Known Issues and Workarounds

1. **Memory Leaks**: LED model may accumulate memory over time
   - *Workaround*: Restart application after processing large batches

2. **Unicode Handling**: Some special characters may cause issues
   - *Workaround*: Text cleaning pipeline handles most cases

3. **Large File Processing**: Files >500MB may cause timeouts
   - *Workaround*: Split large content into smaller chunks



### 9.3 Contributing Guidelines

**Development Setup:**
```bash
# Fork the repository
git clone https://github.com/yourusername/video-summarizer.git
cd video-summarizer

# Set up development environment
python scripts/install.py
source video-summarizer-env/bin/activate
```
---
