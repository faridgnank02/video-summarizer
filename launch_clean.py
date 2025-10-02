#!/usr/bin/env python3
"""
Clean launch script for Video Summarizer
Clears cache and starts the application fresh
"""

import subprocess
import sys
import os
from pathlib import Path

def main():
    """Launch the Video Summarizer with clean cache"""
    print("🚀 Starting Video Summarizer...")
    
    # Get the project directory
    project_dir = Path(__file__).parent
    os.chdir(project_dir)
    
    # Activate virtual environment and clear cache
    print("🧹 Clearing Streamlit cache...")
    try:
        subprocess.run([
            "bash", "-c", 
            "source video-summarizer-env/bin/activate && streamlit cache clear"
        ], check=True, capture_output=True)
        print("✅ Cache cleared successfully")
    except subprocess.CalledProcessError as e:
        print("⚠️ Cache clear failed, continuing anyway...")
    
    # Launch the application
    print("🎬 Launching application...")
    print("📱 App will be available at: http://localhost:8501")
    print("🔄 Press Ctrl+C to stop")
    
    try:
        subprocess.run([
            "bash", "-c", 
            "source video-summarizer-env/bin/activate && streamlit run src/ui/streamlit_app.py"
        ], check=True)
    except KeyboardInterrupt:
        print("\n👋 Application stopped by user")
    except subprocess.CalledProcessError as e:
        print(f"❌ Error launching app: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())