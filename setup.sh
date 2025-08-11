#!/bin/bash

# RAG Document Handler - Setup Script
# This script sets up the development environment and installs dependencies

echo "🚀 Setting up RAG Document Handler..."

# Check if Python 3.8+ is available
python3 --version >/dev/null 2>&1
if [ $? -ne 0 ]; then
    echo "❌ Python 3 is required but not installed."
    exit 1
fi

# Create virtual environment if it doesn't exist
if [ ! -d ".venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv .venv
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source .venv/bin/activate

# Upgrade pip
echo "⬆️ Upgrading pip..."
pip install --upgrade pip

# Install dependencies
echo "📚 Installing dependencies..."
pip install flask pymilvus sentence-transformers werkzeug python-dotenv pypdf python-docx chardet

# Install development dependencies (optional)
echo "🛠️ Installing development dependencies..."
pip install pytest pytest-cov black flake8 mypy isort

# Create staging and uploaded directories
echo "📁 Creating staging and uploaded directories..."
mkdir -p staging
mkdir -p uploaded

# Check if .env file exists
if [ ! -f ".env" ]; then
    echo "⚠️ .env file not found. Using default configuration."
    echo "📝 You may want to copy and customize .env for your environment."
fi

echo "✅ Setup complete!"
echo ""
echo "To start the application:"
echo "1. Activate the virtual environment: source .venv/bin/activate"
echo "2. Start Milvus in Docker: docker run -d --name milvus -p 19530:19530 milvusdb/milvus:latest"
echo "3. Run the application: python app.py"
echo ""
echo "The application will be available at: http://localhost:3000"
