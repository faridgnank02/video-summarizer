# 🚀 Quick Start - Video Summarizer

Express guide to install and use Video Summarizer in 5 minutes.

## ⚡ Express Installation

```bash
# 1. Automated installation
python scripts/install.py

# 2. Activate environment
source video-summarizer-env/bin/activate

# 3. Optional OpenAI configuration
echo "OPENAI_API_KEY=sk-your-key" >> .env

# 4. Launch
python scripts/launch.py
```

## 🎯 Immediate Usage

### Web Interface (Recommended)
1. Open your browser to `http://localhost:8501`
2. Paste a YouTube URL in the "YouTube" tab
3. Choose LED model (free) or OpenAI (fast)
4. Select length: Short or Long
5. Click "Generate Summary" and wait

### Quick Command Line Test
```python
# Test with simple text
python -c "
from src.models.led_model import LEDSummarizer
led = LEDSummarizer()
print(led.summarize('Your long text here...'))
"
```

## 🚀 M1 GPU Optimization (MacBook Pro/Air)

**✅ M1 GPU automatically detected and used!**

### M1 GPU Performance
- 🔥 **1.5x faster** than CPU
- ⚡ **Metal Performance Shaders (MPS) acceleration**
- 🧠 **Lower memory consumption**
- 🌡️ **Less heat generation**

### GPU Verification
```bash
# Test that M1 GPU is being used
python3 -c "
import torch
from src.models.led_model import LEDSummarizer
led = LEDSummarizer()
print(f'Device: {led.device}')  # Should display 'mps'
"
```

## 🔧 Quick Troubleshooting

| Problem | Solution |
|----------|----------|
| ModuleNotFoundError | `pip install -r requirements.txt` |  
| Insufficient memory | Use OpenAI instead of LED |
| M1 GPU not used | Check `config/model_config.yaml` → `device: auto` |
| YouTube error | `pip install --upgrade youtube-transcript-api` |
| No OpenAI key | Use LED model only |

## 📊 Model Comparison Quick Reference

### 🆓 LED Model - **FREE & OFFLINE**
- ✅ No cost, no API key needed
- ✅ Works offline
- ✅ Best for English content
- ✅ GPU accelerated on M1 Macs
- ⏱️ ~5-10 seconds processing
- 📝 Extractive style (preserves original phrasing)

### ⚡ OpenAI Model
- 💰 Costs per usage (API key required)
- 🌐 Requires internet
- ✅ Excellent for all languages
- ⚡ ~2-3 seconds processing
- 🎨 Abstractive style (natural rephrasing)

## 🎯 Quick Quality Tips

- **High-quality transcripts** → LED works excellently
- **Poor-quality transcripts** → OpenAI handles artifacts better
- **Non-English content** → OpenAI recommended
- **Long documents** → LED specialized for 16K+ tokens
- **Fast summaries** → OpenAI for speed

## 📱 Interface Overview

1. **Settings Sidebar**: Model selection, length, language
2. **Video Source**: YouTube URL, local file, or direct text
3. **Generate**: One-click summary generation
4. **Quality Metrics**: Automatic evaluation scores
5. **History**: Previous summaries with quality indicators

---

**Ready to summarize? Launch the app and start in seconds!** 🚀