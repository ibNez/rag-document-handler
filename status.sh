#!/bin/bash

# RAG Document Handler - Status Check Script
# Shows the current state of the project installation

echo "📊 RAG Document Handler - Project Status"
echo "========================================"
echo ""

# Check if we're in the right directory
if [ ! -f "app.py" ]; then
    echo "❌ Not in the RAG Document Handler directory"
    echo "   Run this script from the project root"
    exit 1
fi

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to check if directory exists and show info
check_directory() {
    if [ -d "$1" ]; then
        echo "✅ $1/ ($(du -sh "$1" 2>/dev/null | cut -f1))"
    else
        echo "❌ $1/ (missing)"
    fi
}

# Function to check if file exists
check_file() {
    if [ -f "$1" ]; then
        echo "✅ $1"
    else
        echo "❌ $1 (missing)"
    fi
}

# 1. Check Python environment
echo "🐍 Python Environment:"
if [ -d ".venv" ]; then
    echo "✅ Virtual environment (.venv)"
    if [ -f ".venv/bin/activate" ]; then
        # Check if we can activate and get Python version
        if source .venv/bin/activate 2>/dev/null && python --version >/dev/null 2>&1; then
            version=$(source .venv/bin/activate && python --version 2>&1)
            echo "   Version: $version"
            
            # Check key packages
            if source .venv/bin/activate && python -c "import psycopg2" 2>/dev/null; then
                echo "   ✅ PostgreSQL driver (psycopg2)"
            else
                echo "   ❌ PostgreSQL driver missing"
            fi
            
            if source .venv/bin/activate && python -c "import pymilvus" 2>/dev/null; then
                echo "   ✅ Milvus driver (pymilvus)"
            else
                echo "   ❌ Milvus driver missing"
            fi
        else
            echo "   ⚠️  Virtual environment may be corrupted"
        fi
    fi
else
    echo "❌ Virtual environment (.venv) missing"
fi

echo ""

# 2. Check Docker status
echo "🐳 Docker Services:"
if command_exists docker; then
    if docker info >/dev/null 2>&1; then
        echo "✅ Docker daemon running"
        
        # Check for project containers
        if docker compose ps >/dev/null 2>&1; then
            echo ""
            echo "   Container Status:"
            docker compose ps --format "table {{.Service}}\t{{.State}}\t{{.Ports}}"
        else
            echo "❌ No docker-compose services found"
        fi
    else
        echo "❌ Docker daemon not running"
    fi
else
    echo "❌ Docker not installed"
fi

echo ""

# 3. Check project directories
echo "📁 Project Directories:"
check_directory "databases"
check_directory "databases/postgres"
check_directory "databases/milvus"
check_directory "logs"
check_directory "staging"
check_directory "uploaded"
check_directory "ingestion"

echo ""

# 4. Check configuration files
echo "⚙️  Configuration Files:"
check_file "docker-compose.yml"
check_file ".env.example"
if [ -f ".env" ]; then
    echo "✅ .env (configured)"
else
    echo "⚠️  .env (using defaults)"
fi

echo ""

# 5. Check scripts
echo "🔧 Management Scripts:"
if [ -x "setup.sh" ]; then
    echo "✅ setup.sh (executable)"
else
    echo "❌ setup.sh (missing or not executable)"
fi

if [ -x "uninstall.sh" ]; then
    echo "✅ uninstall.sh (executable)"
else
    echo "❌ uninstall.sh (missing or not executable)"
fi

check_file "test_postgres.py"

echo ""

# 6. Check database connectivity (if environment is set up)
if [ -d ".venv" ] && [ -f ".venv/bin/activate" ]; then
    echo "🔌 Database Connectivity:"
    
    # Test PostgreSQL
    if source .venv/bin/activate && python -c "
import os
os.environ.setdefault('POSTGRES_HOST', 'localhost')
os.environ.setdefault('POSTGRES_PORT', '5432')
os.environ.setdefault('POSTGRES_DB', 'rag_metadata')
os.environ.setdefault('POSTGRES_USER', 'rag_user')
os.environ.setdefault('POSTGRES_PASSWORD', 'secure_password')
from ingestion.core.postgres_manager import PostgreSQLConfig, PostgreSQLManager
try:
    config = PostgreSQLConfig()
    manager = PostgreSQLManager(config)
    manager.close()
    print('✅ PostgreSQL connection successful')
except Exception as e:
    print(f'❌ PostgreSQL connection failed: {e}')
" 2>/dev/null; then
        :
    else
        echo "❌ Cannot test PostgreSQL (environment issues)"
    fi
    
    # Test Milvus
    if source .venv/bin/activate && python -c "
from pymilvus import connections
import os
host = os.getenv('MILVUS_HOST', 'localhost')
port = int(os.getenv('MILVUS_PORT', '19530'))
try:
    connections.connect(host=host, port=port)
    print('✅ Milvus connection successful')
except Exception as e:
    print(f'❌ Milvus connection failed: {e}')
" 2>/dev/null; then
        :
    else
        echo "❌ Cannot test Milvus (environment issues)"
    fi
else
    echo "⏭️  Skipping connectivity tests (environment not ready)"
fi

echo ""

# 7. Show next steps based on status
echo "💡 Recommended Actions:"

if [ ! -d ".venv" ]; then
    echo "   1. Run ./setup.sh to install the project"
elif ! docker info >/dev/null 2>&1; then
    echo "   1. Start Docker daemon"
    echo "   2. Run docker compose up -d"
    echo "   3. Start the application with: python app.py"
elif [ -z "$(docker compose ps -q)" ]; then
    echo "   1. Start services: docker compose up -d"
    echo "   2. Test connectivity: python test_postgres.py"
    echo "   3. Start the application: python app.py"
else
    echo "   ✅ System appears ready!"
    echo "   → Start the application: python app.py"
    echo "   → Visit: http://localhost:3000"
fi

echo ""
echo "📚 For more help:"
echo "   ./setup.sh --help     # Installation help"
echo "   ./uninstall.sh        # Complete removal"
echo "   python test_postgres.py  # Test databases"
