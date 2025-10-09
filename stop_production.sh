#!/bin/bash

# RAG Knowledge Base Manager - Production Stop Script
# This script gracefully stops the production Gunicorn server

# Load LOG_DIR from .env file if it exists, default to "logs"
LOG_DIR="logs"
if [ -f ".env" ]; then
    ENV_LOG_DIR=$(grep "^LOG_DIR=" .env 2>/dev/null | cut -d'=' -f2 | sed 's/"//g' | sed "s/'//g")
    if [ -n "$ENV_LOG_DIR" ]; then
        LOG_DIR="$ENV_LOG_DIR"
    fi
fi

PID_FILE="$LOG_DIR/gunicorn.pid"

echo "🛑 Stopping RAG Knowledge Base Manager (Production Mode)"
echo "=================================================="

if [ -f "$PID_FILE" ]; then
    PID=$(cat "$PID_FILE")
    echo "📋 Found PID file: $PID_FILE"
    echo "🔍 Process ID: $PID"
    
    if kill -0 "$PID" 2>/dev/null; then
        echo "⏳ Sending SIGTERM for graceful shutdown..."
        kill -TERM "$PID"
        
        # Wait for graceful shutdown (up to 30 seconds)
        for i in {1..30}; do
            if ! kill -0 "$PID" 2>/dev/null; then
                echo "✅ Server stopped gracefully"
                rm -f "$PID_FILE"
                rm -f "gunicorn.conf.py"
                exit 0
            fi
            echo -n "."
            sleep 1
        done
        
        echo ""
        echo "⚠️ Graceful shutdown timeout, forcing termination..."
        kill -KILL "$PID" 2>/dev/null || true
        rm -f "$PID_FILE"
        rm -f "gunicorn.conf.py"
        echo "🔴 Server forcefully terminated"
    else
        echo "❌ Process $PID is not running"
        rm -f "$PID_FILE"
        rm -f "gunicorn.conf.py"
    fi
else
    echo "❌ PID file not found at $PID_FILE"
    echo "🔍 Checking for running Gunicorn processes..."
    
    # Find Gunicorn processes related to this application
    PIDS=$(pgrep -f "gunicorn.*wsgi:application" || true)
    
    if [ -n "$PIDS" ]; then
        echo "🔍 Found Gunicorn processes: $PIDS"
        echo "⏳ Stopping processes..."
        echo "$PIDS" | xargs kill -TERM 2>/dev/null || true
        sleep 3
        echo "$PIDS" | xargs kill -KILL 2>/dev/null || true
        echo "🔴 Processes terminated"
    else
        echo "ℹ️ No running Gunicorn processes found"
    fi
    
    # Clean up configuration file
    rm -f "gunicorn.conf.py"
fi

echo "=================================================="
echo "✅ Production server shutdown complete"