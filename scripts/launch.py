#!/usr/bin/env python3
"""
Python launch script for Video Summarizer
Cross-platform launcher with cache clearing for Windows/Mac/Linux
"""

import os
import sys
import subprocess
from pathlib import Path

def main():
    print("🎥 Video Summarizer - Launch")
    print("=" * 40)
    
    # Check directory
    if not Path("requirements.txt").exists():
        print("❌ Error: Run this script from the project directory")
        sys.exit(1)
    
    # Check virtual environment
    if "video-summarizer-env" not in sys.executable:
        print("⚠️  Virtual environment not activated")
        print("💡 Activate it with:")
        print("   source video-summarizer-env/bin/activate  # Linux/Mac")
        print("   video-summarizer-env\\Scripts\\activate     # Windows")
        
        # Try to activate automatically (Unix only)
        if os.name != 'nt':
            venv_path = Path("video-summarizer-env/bin/python")
            if venv_path.exists():
                print("🔄 Attempting automatic activation...")
                os.execv(str(venv_path), [str(venv_path), __file__])
    
    # Check Streamlit
    try:
        import streamlit
        print("✅ Streamlit available")
    except ImportError:
        print("❌ Streamlit not installed")
        print("💡 Install with: pip install streamlit")
        sys.exit(1)
    
    # Clear Streamlit cache for fresh start
    print("🧹 Clearing Streamlit cache...")
    try:
        subprocess.run([sys.executable, "-m", "streamlit", "cache", "clear"], 
                      check=False, capture_output=True)
        print("✅ Cache cleared successfully")
    except Exception:
        print("⚠️ Cache clear failed, continuing anyway...")
    
    # Load environment variables
    env_file = Path(".env")
    if env_file.exists():
        print("📝 Loading .env file...")
        with open(env_file) as f:
            for line in f:
                if line.strip() and not line.startswith('#'):
                    key, value = line.strip().split('=', 1)
                    os.environ[key] = value
    
    # Check OpenAI API key
    if os.getenv('OPENAI_API_KEY'):
        print("✅ OpenAI API key configured")
    else:
        print("⚠️  OPENAI_API_KEY not configured")
        print("💡 Set it in .env to use OpenAI GPT models")
    
    # Launch Streamlit
    print("\n🚀 Starting web interface...")
    print("📱 Access at: http://localhost:8501")
    print("🛑 Stop with Ctrl+C")
    print()
    
    try:
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", 
            "src/ui/streamlit_app.py"
        ], check=True)
    except KeyboardInterrupt:
        print("\n👋 Application stopped by user")
    except subprocess.CalledProcessError as e:
        print(f"❌ Error during launch: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()