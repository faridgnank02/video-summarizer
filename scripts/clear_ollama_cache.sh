#!/bin/bash

# Script to clear Ollama model cache and free memory
# Usage: ./clear_ollama_cache.sh

echo "🧹 Clearing Ollama Cache..."
echo "================================"

# Stop all running Ollama models
echo "1. Unloading all Ollama models..."
curl -X POST http://localhost:11434/api/generate \
  -H "Content-Type: application/json" \
  -d '{"model": "gemma3:1b", "keep_alive": 0}' \
  2>/dev/null

curl -X POST http://localhost:11434/api/generate \
  -H "Content-Type: application/json" \
  -d '{"model": "qwen3:4b", "keep_alive": 0}' \
  2>/dev/null

echo "✅ Models unload request sent"

# Check Ollama processes
echo ""
echo "2. Current Ollama processes:"
ps aux | grep ollama | grep -v grep

# Memory usage
echo ""
echo "3. Memory usage:"
vm_stat | perl -ne '/page size of (\d+)/ and $size=$1; /Pages\s+([^:]+)[^\d]+(\d+)/ and printf("%-16s % 16.2f MB\n", "$1:", $2 * $size / 1048576);'

echo ""
echo "✅ Cache cleanup complete!"
echo ""
echo "💡 Tips to reduce memory usage:"
echo "   - Use smaller models (gemma3:1b instead of qwen3:4b)"
echo "   - Clear models after each use in Streamlit"
echo "   - Restart Ollama server: killall ollama && ollama serve"
