#!/bin/bash

# RAG Knowledge Base Manager - Production Status Script
# This script checks the status of the production Gunicorn server

# Load LOG_DIR from .env file if it exists, default to "logs"
LOG_DIR="logs"
if [ -f ".env" ]; then
    ENV_LOG_DIR=$(grep "^LOG_DIR=" .env 2>/dev/null | cut -d'=' -f2 | sed 's/"//g' | sed "s/'//g")
    if [ -n "$ENV_LOG_DIR" ]; then
        LOG_DIR="$ENV_LOG_DIR"
    fi
fi

PID_FILE="$LOG_DIR/gunicorn.pid"

echo "📊 RAG Knowledge Base Manager - Production Status"
echo "=================================================="

# Check PID file
if [ -f "$PID_FILE" ]; then
    PID=$(cat "$PID_FILE")
    echo "📋 PID File: $PID_FILE"
    echo "🔍 Process ID: $PID"
    
    if kill -0 "$PID" 2>/dev/null; then
        echo "✅ Status: RUNNING"
        
        # Get process details
        echo ""
        echo "🔍 Process Details:"
        ps -p "$PID" -o pid,ppid,user,pcpu,pmem,etime,command --no-headers | while read line; do
            echo "   $line"
        done
        
        # Check worker processes
        echo ""
        echo "👥 Worker Processes:"
        pgrep -P "$PID" | while read worker_pid; do
            ps -p "$worker_pid" -o pid,pcpu,pmem,etime,command --no-headers | while read line; do
                echo "   $line"
            done
        done
        
    else
        echo "❌ Status: NOT RUNNING (stale PID file)"
        echo "🔧 Cleaning up stale PID file..."
        rm -f "$PID_FILE"
    fi
else
    echo "❌ PID File: Not found"
    
    # Check for running Gunicorn processes
    PIDS=$(pgrep -f "gunicorn.*wsgi:application" || true)
    
    if [ -n "$PIDS" ]; then
        echo "⚠️ Status: RUNNING (no PID file)"
        echo "🔍 Found orphaned processes: $PIDS"
        echo ""
        echo "👥 Running Processes:"
        echo "$PIDS" | while read pid; do
            ps -p "$pid" -o pid,pcpu,pmem,etime,command --no-headers | while read line; do
                echo "   $line"
            done
        done
    else
        echo "❌ Status: NOT RUNNING"
    fi
fi

echo ""
echo "📡 Service Connectivity:"

# Check application port
if nc -z localhost 3000 2>/dev/null; then
    echo "✅ Application: http://localhost:3000 (responding)"
else
    echo "❌ Application: http://localhost:3000 (not responding)"
fi

# Check dependencies
if nc -z localhost 5432 2>/dev/null; then
    echo "✅ PostgreSQL: localhost:5432 (connected)"
else
    echo "❌ PostgreSQL: localhost:5432 (not connected)"
fi

if nc -z localhost 19530 2>/dev/null; then
    echo "✅ Milvus: localhost:19530 (connected)"
else
    echo "❌ Milvus: localhost:19530 (not connected)"
fi

# Docker containers status
echo ""
echo "🐳 Docker Services:"
RUNNING_CONTAINERS=$(docker compose ps --services --filter "status=running" 2>/dev/null | wc -l | xargs)
if [ "$RUNNING_CONTAINERS" -gt 0 ]; then
    echo "✅ Running containers: $RUNNING_CONTAINERS"
    docker compose ps --format "table {{.Service}}\t{{.Status}}\t{{.Ports}}" 2>/dev/null | head -10
else
    echo "❌ No Docker containers running"
fi

# Log files
echo ""
echo "📝 Log Files:"
if [ -d "$LOG_DIR" ]; then
    if [ -f "$LOG_DIR/gunicorn_access.log" ]; then
        ACCESS_SIZE=$(du -h "$LOG_DIR/gunicorn_access.log" | cut -f1)
        echo "📊 Access Log: $LOG_DIR/gunicorn_access.log ($ACCESS_SIZE)"
    fi
    
    if [ -f "$LOG_DIR/gunicorn_error.log" ]; then
        ERROR_SIZE=$(du -h "$LOG_DIR/gunicorn_error.log" | cut -f1)
        echo "🚨 Error Log: $LOG_DIR/gunicorn_error.log ($ERROR_SIZE)"
        
        # Check for recent errors
        RECENT_ERRORS=$(tail -n 50 "$LOG_DIR/gunicorn_error.log" 2>/dev/null | grep -c "ERROR" || echo "0")
        if [ "$RECENT_ERRORS" -gt 0 ]; then
            echo "⚠️ Recent errors found: $RECENT_ERRORS (last 50 lines)"
        fi
    fi
    
    if [ -f "$LOG_DIR/rag_knowledge_base_manager.log" ]; then
        APP_SIZE=$(du -h "$LOG_DIR/rag_knowledge_base_manager.log" | cut -f1)
        echo "📋 Application Log: $LOG_DIR/rag_knowledge_base_manager.log ($APP_SIZE)"
    fi
else
    echo "❌ Log directory not found: $LOG_DIR"
fi

echo ""
echo "=================================================="

# Exit with appropriate code
if [ -f "$PID_FILE" ]; then
    PID=$(cat "$PID_FILE")
    if kill -0 "$PID" 2>/dev/null; then
        echo "✅ Production server is running normally"
        exit 0
    fi
fi

echo "❌ Production server is not running"
exit 1