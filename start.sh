#!/bin/bash

# RAG Document Handler - Start Script
# This script starts the application with proper environment setup

echo "🚀 Starting RAG Document Handler..."

# Check if virtual environment exists
if [ ! -d ".venv" ]; then
    echo "❌ Virtual environment not found. Please run ./setup.sh first."
    exit 1
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source .venv/bin/activate

# Check if Milvus is running
echo "🔍 Checking Milvus connection..."
if ! nc -z localhost 19530 2>/dev/null; then
    echo "⚠️ Milvus is not running on localhost:19530"
    echo "Please start Milvus with: docker run -d --name milvus -p 19530:19530 milvusdb/milvus:latest"
    echo "Continuing anyway (you can start Milvus later)..."
fi

# Start the application
echo "🌐 Starting Flask application..."
echo "Application will be available at: http://localhost:3000"
echo "Press Ctrl+C to stop the server"
echo ""

python app.py
