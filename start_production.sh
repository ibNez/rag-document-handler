#!/bin/bash

# RAG Knowledge Base Manager - Production Start Script
# This script starts the application in production mode using Gunicorn WSGI server

set -e  # Exit on any error

# Detect if running in Docker container
if [ -f /.dockerenv ] || [ -n "$KUBERNETES_SERVICE_HOST" ]; then
    IN_DOCKER=true
    echo "🐳 Running in Docker container"
else
    IN_DOCKER=false
    echo "🖥️ Running on host system"
fi

# Configuration
WORKERS=${WORKERS:-4}
WORKER_CLASS=${WORKER_CLASS:-sync}
MAX_REQUESTS=${MAX_REQUESTS:-1000}
MAX_REQUESTS_JITTER=${MAX_REQUESTS_JITTER:-100}
TIMEOUT=${TIMEOUT:-120}
PRELOAD=${PRELOAD:-true}

# Set bind address based on environment
if [ "$IN_DOCKER" = true ]; then
    BIND_ADDRESS=${BIND_ADDRESS:-0.0.0.0:443}
    DEFAULT_LOG_DIR="/app/logs"
else
    BIND_ADDRESS=${BIND_ADDRESS:-0.0.0.0:3000}
    DEFAULT_LOG_DIR="logs"
fi

# Load LOG_DIR from .env file if it exists
LOG_DIR="$DEFAULT_LOG_DIR"
if [ -f ".env" ]; then
    ENV_LOG_DIR=$(grep "^LOG_DIR=" .env 2>/dev/null | cut -d'=' -f2 | sed 's/"//g' | sed "s/'//g")
    if [ -n "$ENV_LOG_DIR" ]; then
        LOG_DIR="$ENV_LOG_DIR"
    fi
fi

# Ensure log directory exists
mkdir -p "$LOG_DIR"

echo "🚀 Starting RAG Knowledge Base Manager (Production Mode)"
echo "=================================================="

# Handle Docker vs host environment setup
if [ "$IN_DOCKER" = true ]; then
    echo "🐳 Docker container environment detected"
    
    # Generate SSL certificates if they don't exist
    if [ ! -f "/app/ssl/certs/server.crt" ] || [ ! -f "/app/ssl/private/server.key" ]; then
        echo "🔐 Generating SSL certificates..."
        mkdir -p /app/ssl/certs /app/ssl/private
        python /app/tools/generate_ssl_cert.py
        echo "✅ SSL certificates generated"
    else
        echo "✅ SSL certificates found"
    fi
    
    # Check service connectivity in Docker
    echo "🔍 Checking service connectivity..."
    
    # Check Milvus
    if nc -z "$MILVUS_HOST" "$MILVUS_PORT" 2>/dev/null; then
        echo "✅ Milvus is accessible at $MILVUS_HOST:$MILVUS_PORT"
    else
        echo "⚠️ Milvus not accessible at $MILVUS_HOST:$MILVUS_PORT"
    fi
    
    # Check PostgreSQL
    if nc -z "$POSTGRES_HOST" "$POSTGRES_PORT" 2>/dev/null; then
        echo "✅ PostgreSQL is accessible at $POSTGRES_HOST:$POSTGRES_PORT"
    else
        echo "⚠️ PostgreSQL not accessible at $POSTGRES_HOST:$POSTGRES_PORT"
    fi
    
    # Check Ollama
    if nc -z "$OLLAMA_HOST" "$OLLAMA_PORT" 2>/dev/null; then
        echo "✅ Ollama is accessible at $OLLAMA_HOST:$OLLAMA_PORT"
    else
        echo "⚠️ Ollama not accessible at $OLLAMA_HOST:$OLLAMA_PORT"
    fi
    
else
    # Host environment setup
    if [ ! -d ".venv" ]; then
        echo "❌ Virtual environment not found. Please run ./setup.sh first."
        exit 1
    fi
    
    echo "🔧 Activating virtual environment..."
    source .venv/bin/activate
    
    # Install gunicorn if not present
    echo "📦 Checking Gunicorn installation..."
    if ! python -c "import gunicorn" 2>/dev/null; then
        echo "📥 Installing Gunicorn..."
        pip install gunicorn[gevent] --quiet
        echo "✅ Gunicorn installed successfully"
    else
        echo "✅ Gunicorn is already installed"
    fi
    
    # Check local services
    echo "🔍 Checking local service connections..."
    if ! nc -z localhost 19530 2>/dev/null; then
        echo "⚠️ Milvus is not running on localhost:19530"
        echo "Please start with: docker compose up milvus -d"
    else
        echo "✅ Milvus is running on localhost:19530"
    fi
    
    if ! nc -z localhost 5432 2>/dev/null; then
        echo "⚠️ PostgreSQL is not running on localhost:5432"
        echo "Please start with: docker compose up postgres -d"
    else
        echo "✅ PostgreSQL is running on localhost:5432"
    fi
fi

# Set production environment variables
export FLASK_DEBUG=False
export FLASK_ENV=production

echo ""
echo "🌐 Starting Gunicorn WSGI server..."
echo "=================================================="
echo "Configuration:"
echo "   👥 Workers: $WORKERS"
echo "   🔄 Worker Class: $WORKER_CLASS"
echo "   📊 Max Requests: $MAX_REQUESTS (±$MAX_REQUESTS_JITTER)"
echo "   ⏱️  Timeout: ${TIMEOUT}s"
echo "   🌐 Bind: $BIND_ADDRESS"
echo "   📁 Logs: $LOG_DIR/"
echo "   🔧 Preload: $PRELOAD"

if [ "$IN_DOCKER" = true ]; then
    echo "   🔒 SSL: Enabled (Docker container)"
    echo ""
    echo "📊 Services Status:"
    echo "   🌐 Web App: https://localhost:443 (starting...)"
    echo "   🌐 HTTP Redirect: http://localhost:80 → https://localhost:443"
else
    echo "   🔒 SSL: Development mode"
    echo ""
    echo "📊 Services Status:"
    echo "   🌐 Web App: http://localhost:3000 (starting...)"
fi

echo "   🐘 PostgreSQL: ${POSTGRES_HOST:-localhost}:${POSTGRES_PORT:-5432}"
echo "   🔍 Milvus: ${MILVUS_HOST:-localhost}:${MILVUS_PORT:-19530}"
echo "   🤖 Ollama: ${OLLAMA_HOST:-localhost}:${OLLAMA_PORT:-11434}"
echo ""
echo "📁 Data Directories:"
if [ "$IN_DOCKER" = true ]; then
    echo "   📤 Uploads: /app/uploaded/"
    echo "   🗂️  Staging: /app/staging/"
    echo "   🗑️  Deleted: /app/deleted/"
    echo "   📝 Logs: /app/logs/"
else
    echo "   📤 Uploads: ./uploaded/"
    echo "   🗂️  Staging: ./staging/"
    echo "   🗑️  Deleted: ./deleted/"
    echo "   📝 Logs: ./$LOG_DIR/"
fi
echo ""
echo "Press Ctrl+C to stop the server"
echo "=================================================="
echo ""

# Create SSL-enabled Gunicorn configuration
if [ "$IN_DOCKER" = true ] && [ -f "/app/ssl/certs/server.crt" ] && [ -f "/app/ssl/private/server.key" ]; then
    SSL_CONFIG="
# SSL Configuration
certfile = '/app/ssl/certs/server.crt'
keyfile = '/app/ssl/private/server.key'
ssl_version = 3  # SSLv23
ciphers = 'TLSv1.2:TLSv1.3'
"
else
    SSL_CONFIG=""
fi

# Create Gunicorn configuration
GUNICORN_CONFIG=$(cat << EOF
# Gunicorn Production Configuration
import multiprocessing
import os

# Server socket
bind = "$BIND_ADDRESS"
backlog = 2048
$SSL_CONFIG
# Worker processes
workers = $WORKERS
worker_class = "$WORKER_CLASS"
worker_connections = 1000
max_requests = $MAX_REQUESTS
max_requests_jitter = $MAX_REQUESTS_JITTER
timeout = $TIMEOUT
keepalive = 2

# Restart workers gracefully
preload_app = $PRELOAD
reload = False

# Process naming
proc_name = 'rag-knowledgebase-manager'

# Logging
accesslog = "$LOG_DIR/gunicorn_access.log"
errorlog = "$LOG_DIR/gunicorn_error.log"
loglevel = "info"
access_log_format = '%(h)s %(l)s %(u)s %(t)s "%(r)s" %(s)s %(b)s "%(f)s" "%(a)s" %(D)s'

# Security
limit_request_line = 8192
limit_request_fields = 200
limit_request_field_size = 8190

# Performance
if os.path.exists("/dev/shm"):
    worker_tmp_dir = "/dev/shm"
EOF
)

# Write Gunicorn configuration to file
echo "$GUNICORN_CONFIG" > gunicorn.conf.py

# Start Gunicorn with the WSGI application
exec gunicorn \
    --config gunicorn.conf.py \
    --pid "$LOG_DIR/gunicorn.pid" \
    wsgi:application
