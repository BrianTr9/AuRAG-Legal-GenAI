#!/bin/bash
set -e

echo "🚀 Local RAG Baseline Setup"
echo "==========================="

echo "Checking Python..."
python3 --version || { echo "❌ Python 3 required"; exit 1; }

echo "Creating venv..."
python3 -m venv .venv

echo "Activating..."
source .venv/bin/activate

echo "Upgrading pip..."
pip install --upgrade pip

echo "Installing dependencies..."
pip install -r requirements.txt

echo "Creating docs folder..."
mkdir -p docs

echo ""
echo "✅ Done!"
echo ""
echo "Next:"
echo "1. source .venv/bin/activate"
echo "2. Add PDFs to docs/"
echo "3. python RAG.py"
echo ""
