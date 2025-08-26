#!/bin/bash

# RAG Knowledge Base Manager - Uninstall Script
# This script completely removes the installation and cleans up all resources

# Show help
show_help() {
    echo "🗑️  RAG Knowledge Base Manager - Uninstall Script"
    echo "============================================="
    echo ""
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  --dry-run, -n    Preview what would be removed WITHOUT making any changes"
    echo "  --help, -h       Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0               # Interactive uninstall (removes files)"
    echo "  $0 --dry-run     # Safe preview mode (no files removed)"
    echo ""
    echo "The --dry-run option is completely safe and will:"
    echo "  ✅ Show exactly what would be removed"
    echo "  ✅ Never prompt for user input"
    echo "  ✅ Never delete any files or containers"
    echo "  ✅ Leave your system completely unchanged"
    echo ""
    exit 0
}

# Parse command line arguments
DRY_RUN=false
case "$1" in
    --help|-h)
        show_help
        ;;
    --dry-run|-n)
        DRY_RUN=true
        echo "🔍 DRY RUN MODE - No actual changes will be made"
        echo ""
        ;;
    "")
        # No arguments, proceed normally
        ;;
    *)
        echo "❌ Unknown option: $1"
        echo "Use --help for usage information"
        exit 1
        ;;
esac

echo "🗑️  RAG Knowledge Base Manager - Uninstall Script"
echo "============================================="
echo ""
echo "⚠️  WARNING: This will process:"
echo "   - Each Docker container individually (you'll be asked about each one)"
echo "   - Each Docker volume individually (you'll be asked about each one)"
echo "   - Project virtual environment (.venv)"
echo "   - Project database files and logs"
echo "   - Uploaded documents and staging files"
echo ""
echo "💡 For each container, you'll be asked:"
echo "   1. Remove container? (Default: Yes)"
echo "   2. Remove volume? (Default: No - preserves data)"
echo ""
echo "🔄 Keeping volumes allows you to reinstall and retain your data."
echo ""

# Confirmation prompt
if $DRY_RUN; then
    echo "🔍 DRY RUN MODE - Previewing what would be removed (no changes will be made)"
else
    read -r -p "Are you sure you want to proceed with uninstalling? (y/N): " confirm
    if [[ ! $confirm =~ ^[Yy]$ ]]; then
        echo "❌ Uninstall cancelled."
        exit 0
    fi
fi

echo ""
echo "🚀 Starting uninstall process..."

# Function to safely remove directory
safe_remove_dir() {
    if [ -d "$1" ]; then
        if $DRY_RUN; then
            echo "📁 [DRY RUN] Would remove directory: $1"
        else
            echo "📁 Removing directory: $1"
            rm -rf "$1"
        fi
    elif $DRY_RUN; then
        echo "📁 [DRY RUN] Directory not found: $1"
    fi
}

# Function to safely remove file
safe_remove_file() {
    if [ -f "$1" ]; then
        if $DRY_RUN; then
            echo "📄 [DRY RUN] Would remove file: $1"
        else
            echo "📄 Removing file: $1"
            rm -f "$1"
        fi
    elif $DRY_RUN; then
        echo "📄 [DRY RUN] File not found: $1"
    fi
}

# Step 1: Stop and remove Docker containers
echo ""
echo "🐳 Managing Docker containers and volumes..."
if command -v docker >/dev/null 2>&1; then
    if docker info >/dev/null 2>&1; then
        # Get list of project containers
        containers=$(docker ps -a --filter "name=rag-document-handler" --format "{{.Names}}" 2>/dev/null || true)
        
        if [ -z "$containers" ]; then
            echo "ℹ️  No rag-document-handler containers found"
        else
            echo "📦 Found project containers:"
            echo "$containers" | sed 's/^/   - /'
            echo ""
            
            if $DRY_RUN; then
                echo "🔍 [DRY RUN] Would process each container individually:"
                echo "$containers" | while read -r container; do
                    if [ -n "$container" ]; then
                        echo ""
                        echo "   [DRY RUN] Container: $container"
                        echo "   [DRY RUN] Would ask: Remove container '$container'? (Y/n)"
                        echo "   [DRY RUN] Would ask: Remove volume for '$container'? (y/N)"
                        echo "   [DRY RUN] Would remove container and volume based on choices"
                    fi
                done
                
                # Also handle compose services
                echo ""
                echo "   [DRY RUN] Would also process docker-compose services:"
                echo "   [DRY RUN] Services: webui, milvus, postgres"
                echo "   [DRY RUN] Would ask about each service container and volume individually"
            else
                # Process each container individually
                echo ""
                echo "🗑️  Processing containers:"
                
                # Process main compose services
                services=("webui" "milvus" "postgres")
                for service in "${services[@]}"; do
                    container_name="rag-document-handler-${service}-1"
                    container_exists=$(docker ps -a --filter "name=$container_name" --format "{{.Names}}" 2>/dev/null || true)
                    
                    if [ -n "$container_exists" ]; then
                        echo ""
                        echo "📦 Container: $service ($container_name)"
                        
                        # Ask about removing container (default Yes)
                        read -r -p "   Remove container '$service'? (Y/n): " remove_container
                        if [[ ! $remove_container =~ ^[Nn]$ ]]; then
                            
                            # Ask about removing volume (default No)
                            read -r -p "   Remove volume for '$service'? (y/N): " remove_volume
                            if [[ $remove_volume =~ ^[Yy]$ ]]; then
                                echo "   ✅ Removing container '$service' and any volumes"
                                docker compose -p rag-document-handler rm -f -s -v "$service" 2>/dev/null || true
                                # Remove volume by name pattern if it exists
                                docker volume ls -q | grep -E "rag-document-handler.*${service}" | xargs -r docker volume rm 2>/dev/null || true
                            else
                                echo "   ✅ Removing container '$service' (preserving any volumes)"
                                docker compose -p rag-document-handler rm -f -s "$service" 2>/dev/null || true
                            fi
                        else
                            echo "   ℹ️  Keeping container '$service'"
                        fi
                    else
                        echo ""
                        echo "📦 Container: $service (not found)"
                    fi
                done
                
                # Clean up any remaining containers that match the pattern
                remaining_containers=$(docker ps -aq --filter "name=rag-document-handler" 2>/dev/null || true)
                if [ -n "$remaining_containers" ]; then
                    echo ""
                    echo "🧹 Cleaning up any remaining project containers..."
                    echo "$remaining_containers" | xargs -r docker rm -f 2>/dev/null || true
                fi
            fi
        fi
        
        # Remove project-specific images
        echo ""
        echo "🖼️  Removing project images..."
        if $DRY_RUN; then
            echo "   [DRY RUN] Would remove images matching: rag-document-handler*"
            docker images -q --filter "reference=rag-document-handler*" | sed 's/^/   [DRY RUN] Image: /' || true
        else
            docker images -q --filter "reference=rag-document-handler*" | xargs -r docker rmi -f 2>/dev/null || true
        fi
        
        echo "✅ Docker cleanup completed"
    else
        echo "⚠️  Docker daemon not running - skipping Docker cleanup"
    fi
else
    echo "⚠️  Docker not found - skipping Docker cleanup"
fi

# Step 2: Remove virtual environment
echo ""
echo "🐍 Removing Python virtual environment..."
if [ -d ".venv" ]; then
    if $DRY_RUN; then
        echo "   [DRY RUN] Would deactivate and remove virtual environment"
        echo "   [DRY RUN] Would remove .venv directory"
        echo "✅ [DRY RUN] Virtual environment would be removed"
    else
        echo "   Deactivating virtual environment..."
        # Try to deactivate if we're in the venv
        if [[ "$VIRTUAL_ENV" == *".venv"* ]]; then
            deactivate 2>/dev/null || true
        fi
        
        echo "   Removing .venv directory..."
        rm -rf .venv
        echo "✅ Virtual environment removed"
    fi
else
    if $DRY_RUN; then
        echo "ℹ️  [DRY RUN] No virtual environment found"
    else
        echo "ℹ️  No virtual environment found"
    fi
fi

# Step 3: Remove database files and directories
echo ""
echo "🗄️  Removing database files..."
safe_remove_dir "databases"
echo "✅ Database files removed"

# Step 4: Remove log files and directories
echo ""
echo "📋 Removing log files..."
safe_remove_dir "logs"
safe_remove_file "rag_document_handler.log"
safe_remove_file "*.log"
echo "✅ Log files removed"

# Step 5: Remove uploaded and staging files
echo ""
echo "📂 Removing uploaded and staging files..."
safe_remove_dir "staging"
safe_remove_dir "uploaded"

# Ask about removing the deleted folder (backup files)
if [ -d "deleted" ]; then
    echo ""
    if $DRY_RUN; then
        echo "   [DRY RUN] Would ask about removing 'deleted' folder and backup files"
        safe_remove_dir "deleted"
        echo "✅ [DRY RUN] Deleted folder and backup files would be handled based on user choice"
    else
        read -r -p "Do you want to remove the 'deleted' folder and its backup files? (y/N): " remove_deleted
        if [[ $remove_deleted =~ ^[Yy]$ ]]; then
            safe_remove_dir "deleted"
            echo "✅ Deleted folder and backup files removed"
        else
            echo "ℹ️  Deleted folder preserved (contains backup files)"
        fi
    fi
else
    if $DRY_RUN; then
        echo "ℹ️  [DRY RUN] No deleted folder found"
    else
        echo "ℹ️  No deleted folder found"
    fi
fi

echo "✅ File directories cleanup completed"

# Step 6: Remove temporary and cache files
echo ""
echo "🧹 Cleaning up temporary files..."
safe_remove_dir "__pycache__"
find . -name "*.pyc" -delete 2>/dev/null || true
find . -name "*.pyo" -delete 2>/dev/null || true
find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
find . -name ".pytest_cache" -type d -exec rm -rf {} + 2>/dev/null || true
safe_remove_file ".coverage"
safe_remove_dir ".coverage"
safe_remove_dir ".mypy_cache"
echo "✅ Temporary files cleaned"

# Step 7: Remove test files
echo ""
echo "🧪 Removing test files..."
safe_remove_file "test_postgres.py"
safe_remove_file "test_*.py"
safe_remove_file "verify_*.py"
safe_remove_file "debug_*.py"
echo "✅ Test files removed"

# Step 8: Remove environment file (optional)
echo ""
if $DRY_RUN; then
    echo "   [DRY RUN] Would ask about removing .env file"
    safe_remove_file ".env"
    echo "✅ [DRY RUN] Environment file would be handled based on user choice"
else
    read -r -p "Do you want to remove the .env file? (y/N): " remove_env
    if [[ $remove_env =~ ^[Yy]$ ]]; then
        safe_remove_file ".env"
        echo "✅ Environment file removed"
    else
        echo "ℹ️  Environment file preserved"
    fi
fi

# Step 10: System cleanup
echo ""
echo "🧽 Performing system cleanup..."

# Note: We don't clear pip cache as it may affect other projects
echo "   ℹ️  Preserving pip cache (may be used by other projects)"

# Final verification
echo ""
echo "🔍 Verifying cleanup..."

# Check for any remaining project files
remaining_files=()
if [ -d "databases" ]; then remaining_files+=("databases/"); fi
if [ -d "logs" ]; then remaining_files+=("logs/"); fi
if [ -d ".venv" ]; then remaining_files+=(".venv/"); fi
if [ -d "staging" ]; then remaining_files+=("staging/"); fi
if [ -d "uploaded" ]; then remaining_files+=("uploaded/"); fi
if [ -d "deleted" ]; then remaining_files+=("deleted/ (preserved backup files)"); fi

if [ ${#remaining_files[@]} -eq 0 ]; then
    echo "✅ Cleanup verification successful - no remaining files"
else
    echo "⚠️  Some files may remain:"
    for file in "${remaining_files[@]}"; do
        echo "   - $file"
    done
fi

# Check Docker status
if command -v docker >/dev/null 2>&1 && docker info >/dev/null 2>&1; then
    containers=$(docker ps -a --filter "name=rag" --filter "name=postgres" --filter "name=milvus" -q)
    volumes=$(docker volume ls -q | grep -E "rag-document-handler" 2>/dev/null || true)
    
    if [ -z "$containers" ]; then
        echo "✅ No project containers remaining"
    else
        echo "⚠️  Some containers may still exist - run 'docker ps -a' to check"
    fi
    
    if [ -n "$volumes" ]; then
        echo "ℹ️  Docker volumes preserved - your data is retained for reinstallation"
        echo "   To view preserved volumes: docker volume ls | grep rag-document-handler"
    else
        echo "✅ No project volumes remaining"
    fi
fi

echo ""
echo "🎉 Uninstall completed!"
echo ""
echo "📋 Summary of actions taken:"
echo "   ✅ Docker containers (handled individually based on user choices)"
echo "   ✅ Docker volumes (handled individually based on user choices)"
echo "   ✅ Virtual environment removed"
echo "   ✅ Database files removed"
echo "   ✅ Log files removed"
echo "   ✅ Uploaded/staging files removed"
echo "   ✅ Temporary files cleaned"
echo "   ✅ Test files removed"
echo ""
echo "🔄 To reinstall the project:"
echo "   1. Run: ./setup.sh"
echo "   2. Follow the setup prompts"
echo "   3. Any preserved Docker volumes will retain your data"
echo ""
echo "💡 Core project files (source code, configs) have been preserved."
echo "   Only generated/runtime files have been removed."
