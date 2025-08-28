#!/bin/bash

# RAG Knowledge Base Manager - Start Script
# This script starts the application with proper environment setup

echo "🚀 Starting RAG Knowledge Base Manager"
echo "The first application start can take some time as we configure the environment..."
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
    echo "Please start with: docker compose up milvus -d"
    echo "Continuing anyway (you can start Milvus later)..."
else
    echo "✅ Milvus is running on localhost:19530"
fi

# Check if PostgreSQL is running
echo "🐘 Checking PostgreSQL connection..."
if ! nc -z localhost 5432 2>/dev/null; then
    echo "⚠️ PostgreSQL is not running on localhost:5432"
    echo "Please start with: docker compose up postgres -d"
    echo "Continuing anyway (you can start PostgreSQL later)..."
else
    echo "✅ PostgreSQL is running on localhost:5432"
fi

# Check if all containers are running
RUNNING_CONTAINERS=$(docker compose ps --services --filter "status=running" 2>/dev/null | wc -l | xargs)
if [ "$RUNNING_CONTAINERS" -gt 0 ]; then
    echo "🐳 Docker containers running: $RUNNING_CONTAINERS"
    echo "💡 Start all services: docker compose up -d"
else
    echo "🐳 No Docker containers running"
    echo "💡 Start all services: docker compose up -d"
fi

# Start the application
echo ""
echo "🌐 Starting Flask application..."
echo "Application will be available at: http://localhost:3000"
echo ""
echo "📊 Services Status:"
echo "   🌐 Web App: http://localhost:3000 (starting...)"
echo "   🐘 PostgreSQL: localhost:5432"
echo "   🔍 Milvus: localhost:19530"
echo ""
echo "📁 Data Directories:"
echo "   📤 Uploads: ./uploaded/"
echo "   🗂️  Staging: ./staging/"
echo "   🗑️  Deleted: ./deleted/"
echo "   📝 Logs: ./logs/"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

python main.py
