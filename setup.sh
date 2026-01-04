#!/bin/bash
set -e

echo "🚀 AuRAG Complete Setup (Llama-3.1-8B via Ollama)"
echo "=================================================="

# ==========================================
# 1. Python Environment Setup
# ==========================================
echo ""
echo "📦 Python Setup..."
echo "Checking Python..."
python3 --version || { echo "❌ Python 3 required"; exit 1; }

echo "Creating venv..."
python3 -m venv .venv

echo "Activating venv..."
source .venv/bin/activate

echo "Upgrading pip..."
pip install --upgrade pip > /dev/null 2>&1

echo "Installing Python dependencies..."
pip install -r requirements.txt > /dev/null 2>&1

echo "Creating data folders..."
mkdir -p data/raw
mkdir -p data/processed
mkdir -p data/vector_db

echo "✅ Python environment ready"

# ==========================================
# 2. Ollama Setup (macOS only)
# ==========================================
echo ""
echo "🤖 Ollama Setup..."

# Check if on macOS
if [[ "$OSTYPE" == "darwin"* ]]; then
    # Check if Ollama installed
    if ! command -v ollama &> /dev/null; then
        echo "Installing Ollama via Homebrew..."
        brew install ollama
    else
        echo "✅ Ollama already installed"
    fi
    
    # Start Ollama service
    echo "Starting Ollama service..."
    brew services start ollama > /dev/null 2>&1 || true
    sleep 2  # Wait for service to start
    
    # Pull model
    echo "Pulling Llama-3.1-8B model (~5GB, this may take 5-10 minutes)..."
    ollama pull llama3.1
    
    echo "✅ Ollama ready"
else
    echo "⚠️  Ollama setup for non-macOS detected."
    echo "   Please install Ollama manually: https://ollama.ai/download"
    echo "   Then run: ollama pull llama3.1"
fi

# ==========================================
# Summary
# ==========================================
echo ""
echo "=================================================="
echo "✅ AuRAG Setup Complete!"
echo "=================================================="
echo ""
echo "Next steps:"
echo "1. Add contracts to data/raw/ folder"
echo "2. Run: source .venv/bin/activate && python src/AuRAG.py"
echo ""

