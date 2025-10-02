#!/usr/bin/env python3
"""
Test script to verify the DataIngestion functionality
"""

import sys
from pathlib import Path

# Add src directory to PYTHONPATH
sys.path.append(str(Path(__file__).parent / "src"))

def test_data_ingestion():
    """Test DataIngestion class methods"""
    print("🧪 Testing DataIngestion class...")
    
    try:
        from data.ingestion import DataIngestion
        
        # Initialize
        ingestion = DataIngestion()
        print("✅ DataIngestion initialized successfully")
        
        # Check available methods
        methods = [m for m in dir(ingestion) if not m.startswith('_')]
        print(f"📋 Available methods: {methods}")
        
        # Check specific methods exist
        required_methods = ['process_youtube_url', 'process_local_video', 'process_text_input']
        for method in required_methods:
            if hasattr(ingestion, method):
                print(f"✅ Method '{method}' exists")
            else:
                print(f"❌ Method '{method}' missing")
        
        # Test text processing
        print("\n🧪 Testing text input processing...")
        test_text = "This is a test text for summarization."
        video_data = ingestion.process_text_input(test_text, "Test Title")
        print(f"✅ Text processing successful: {video_data.title}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_streamlit_app():
    """Test Streamlit app initialization"""
    print("\n🧪 Testing Streamlit app...")
    
    try:
        from ui.streamlit_app import VideoSummarizerApp
        
        app = VideoSummarizerApp()
        print("✅ Streamlit app initialized successfully")
        
        # Check ingestion methods
        if hasattr(app.ingestion, 'process_youtube_url'):
            print("✅ app.ingestion.process_youtube_url exists")
        else:
            print("❌ app.ingestion.process_youtube_url missing")
            
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🚀 Running Video Summarizer Tests\n")
    
    success = True
    success &= test_data_ingestion()
    success &= test_streamlit_app()
    
    if success:
        print("\n🎉 All tests passed! The application should work correctly.")
        print("\n💡 To run the app: streamlit run src/ui/streamlit_app.py")
    else:
        print("\n❌ Some tests failed. Check the errors above.")