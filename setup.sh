#!/bin/bash

# RAG Knowledge Base Manager - Setup Script
# This script sets up the development environment and installs dependencies

# Script configuration
SCRIPT_VERSION="2.0"
AUTO_YES=false
DEV_MODE=false
SHOW_HELP=false
VERBOSE=false

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Log / directory configuration
LOG_DIR="logs/setup"

# Spinner / progress indicator
SPINNER_PID=""
start_spinner() {
    local msg="$1"
    local frames='|/-\\'
    local i=0
    echo -ne "${BLUE}$msg...${NC}" >&2
    (
        tput civis 2>/dev/null || true
        while true; do
            printf "\r${BLUE}%s %s${NC}" "$msg" "${frames:i++%${#frames}:1}" >&2
            sleep 0.15
        done
    ) &
    SPINNER_PID=$!
}

stop_spinner() {
    local status=$1
    if [ -n "$SPINNER_PID" ]; then
        kill "$SPINNER_PID" 2>/dev/null || true
        wait "$SPINNER_PID" 2>/dev/null || true
        SPINNER_PID=""
        tput cnorm 2>/dev/null || true
    fi
    if [ "$status" = 0 ]; then
        echo -e "\r${GREEN}✅ $2${NC}"
    else
        echo -e "\r${YELLOW}⚠️  $2${NC}"
    fi
}

# Help function
show_help() {
    echo "🚀 RAG Knowledge Base Manager - Setup Script v${SCRIPT_VERSION}"
    echo "============================================="
    echo ""
    echo "USAGE:"
    echo "  ./setup.sh [OPTIONS]"
    echo ""
    echo "OPTIONS:"
    echo "  --all        Install everything without prompts (full automated setup)"
    echo "               • Creates virtual environment"
    echo "               • Installs all dependencies"
    echo "               • Starts all Docker containers (webui, postgres, milvus, ollama)"
    echo "               • Sets up all directories and configuration"
    echo ""
    echo "  --dev        Development mode setup (containers only, no webui)"
    echo "               • Creates virtual environment"
    echo "               • Installs all dependencies"
    echo "               • Starts only postgres and milvus containers"
    echo "               • Skips webui container for local development"
    echo "               • Sets up all directories and configuration"
    echo ""
    echo "  --help       Show this help message and exit"
    echo "  --verbose    Show full pip output and retain logs (pip_install_*.log)"
    echo ""
    echo "EXAMPLES:"
    echo "  ./setup.sh              # Interactive setup with prompts"
    echo "  ./setup.sh --all        # Fully automated setup"
    echo "  ./setup.sh --dev        # Setup for local development"
    echo "  ./setup.sh --help       # Show this help"
    echo ""
    echo "WHAT GETS INSTALLED:"
    echo "  📦 Python virtual environment (.venv)"
    echo "  📚 Python dependencies (Flask, Milvus, PostgreSQL, etc.)"
    echo "  🗂️  Required directories (staging, uploaded, logs, databases)"
    echo "  🐳 Docker containers:"
    echo "      • postgres:15      (metadata database)"
    echo "      • milvus:v2.4.13   (vector database)"
    echo "      • ollama:latest    (local LLM server)"
    echo "      • webui            (web interface) - only with --all"
    echo ""
    echo "AFTER SETUP:"
    echo "  🌐 Web interface: http://localhost:3000 (if webui container running)"
    echo "  🔧 Local development: source .venv/bin/activate && python app.py"
    echo "  📊 Status check: ./status.sh"
    echo "  🤖 Robots.txt monitoring: python tools/robots_monitor.py --test-system"
    echo "  📈 Performance testing: python tools/robots_performance_test.py"
    echo "  🧪 Run tests: pytest tests/ -v"
    echo "  🗑️  Cleanup: ./uninstall.sh"
    echo ""
    exit 0
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --all)
            AUTO_YES=true
            shift
            ;;
        --dev)
            DEV_MODE=true
            shift
            ;;
        --help)
            SHOW_HELP=true
            shift
            ;;
        --verbose)
            VERBOSE=true
            shift
            ;;
        *)
            echo "❌ Unknown option: $1"
            echo "   Run './setup.sh --help' for usage information"
            exit 1
            ;;
    esac
done

# Show help if requested
if [ "$SHOW_HELP" = true ]; then
    show_help
fi

# Function to ask yes/no questions (respects --all flag)
ask_yes_no() {
    local question="$1"
    local default="$2"
    
    if [ "$AUTO_YES" = true ]; then
        echo -e "${BLUE}$question${NC} (auto-yes)"
        return 0
    fi
    
    while true; do
        if [ "$default" = "y" ]; then
            read -r -p "$(echo -e ${BLUE}$question${NC}) (Y/n): " answer
            answer=${answer:-y}
        else
            read -r -p "$(echo -e ${BLUE}$question${NC}) (y/N): " answer
            answer=${answer:-n}
        fi
        
        case $answer in
            [Yy]* ) return 0;;
            [Nn]* ) return 1;;
            * ) echo "Please answer yes or no.";;
        esac
    done
}

# Print setup mode
echo "🚀 RAG Knowledge Base Manager - Setup Script v${SCRIPT_VERSION}"
echo "======================================================="

if [ "$AUTO_YES" = true ]; then
    echo -e "${GREEN}🤖 Mode: Fully Automated Setup${NC}"
    echo "   Installing all components without prompts..."
elif [ "$DEV_MODE" = true ]; then
    echo -e "${YELLOW}👨‍💻 Mode: Development Setup${NC}"
    echo "   Installing for local development (no webui container)..."
else
    echo -e "${BLUE}🔧 Mode: Interactive Setup${NC}"
    echo "   You will be prompted for each component..."
fi
echo ""

# Check if Python 3.8+ is available
echo "🐍 Checking Python installation..."
python3 --version >/dev/null 2>&1
if [ $? -ne 0 ]; then
    echo -e "${RED}❌ Python 3 is required but not installed.${NC}"
    echo "   Please install Python 3.8+ and try again."
    exit 1
fi

PYTHON_VERSION=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
echo -e "${GREEN}✅ Python $PYTHON_VERSION found${NC}"

# Create virtual environment if it doesn't exist
if [ ! -d ".venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv .venv
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✅ Virtual environment created${NC}"
    else
        echo -e "${RED}❌ Failed to create virtual environment${NC}"
        exit 1
    fi
else
    echo -e "${GREEN}✅ Virtual environment already exists${NC}"
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source .venv/bin/activate

# Ensure directory structure (moved earlier so logs can be captured inside logs/)
echo "📁 Setting up directory structure..."
directories=("staging" "uploaded" "deleted" "logs" "databases/milvus/db" "databases/milvus/conf" "databases/postgres" "containers/ollama" "logs/milvus" "logs/postgres" "logs/ollama" "$LOG_DIR")
for dir in "${directories[@]}"; do
    if [ ! -d "$dir" ]; then
        mkdir -p "$dir"
        echo "   ✅ Created: $dir"
    else
        echo "   ✅ Exists: $dir"
    fi
done

# Upgrade pip
echo "⬆️ Upgrading pip..."
if [ "$VERBOSE" = true ]; then
    pip install --upgrade pip || echo -e "${YELLOW}⚠️  Pip upgrade reported warnings/errors (continuing)${NC}"
else
    if ! pip install --upgrade pip > "$LOG_DIR/pip_upgrade.log" 2>&1; then
        echo -e "${YELLOW}⚠️  Pip upgrade had issues (see $LOG_DIR/pip_upgrade.log)${NC}"
    fi
fi

# Install dependencies (with spinner in non-verbose mode)
echo "📚 Installing Python dependencies (editable mode)..."
PRIMARY_LOG="$LOG_DIR/pip_install_primary.log"
FALLBACK_LOG="$LOG_DIR/pip_install_fallback.log"

if [ "$VERBOSE" = true ]; then
    echo "(verbose) pip install -e ."
    if pip install -e . 2>&1 | tee "$PRIMARY_LOG"; then
        echo -e "${GREEN}✅ Dependencies installed successfully${NC}"
    else
        echo -e "${YELLOW}⚠️  Primary installation failed. Attempting fallback explicit package set...${NC}"
        echo "Fallback: core explicit packages" | tee "$FALLBACK_LOG"
        if pip install flask pymilvus sentence-transformers werkzeug python-dotenv pypdf python-docx chardet requests beautifulsoup4 google-api-python-client google-auth psycopg2-binary playwright 2>&1 | tee -a "$FALLBACK_LOG"; then
            echo -e "${GREEN}✅ Fallback dependencies installed${NC}"
        else
            echo -e "${RED}❌ Fallback installation failed. Review $FALLBACK_LOG${NC}"
            exit 1
        fi
    fi
else
    start_spinner "Resolving & installing Python packages"
    if pip install -e . > "$PRIMARY_LOG" 2>&1; then
        stop_spinner 0 "Dependencies installed"
        echo -e "${BLUE}ℹ️  Install log: $PRIMARY_LOG${NC}"
    else
        stop_spinner 1 "Primary install failed – attempting fallback"
        echo -e "${BLUE}🔍 Last 15 lines from primary log:${NC}"
        tail -n 15 "$PRIMARY_LOG"
        start_spinner "Installing fallback explicit packages"
        if pip install flask pymilvus sentence-transformers werkzeug python-dotenv pypdf python-docx chardet requests beautifulsoup4 google-api-python-client google-auth psycopg2-binary playwright > "$FALLBACK_LOG" 2>&1; then
            stop_spinner 0 "Fallback dependencies installed"
            echo -e "${BLUE}ℹ️  Logs: primary=$PRIMARY_LOG fallback=$FALLBACK_LOG${NC}"
        else
            stop_spinner 1 "Fallback installation failed"
            echo -e "${BLUE}🔍 Last 25 lines from fallback log:${NC}"
            tail -n 25 "$FALLBACK_LOG"
            echo -e "${YELLOW}💡 Re-run ./setup.sh --verbose for full pip output${NC}"
            exit 1
        fi
    fi
fi

echo -e "${GREEN}📦 Python dependency phase complete${NC}"


# Install development dependencies (optional)
if ask_yes_no "Install development dependencies (pytest, black, flake8, mypy)?" "n"; then
    echo "🛠️ Installing development dependencies..."
    DEV_LOG="$LOG_DIR/dev_deps_install.log"
    if [ "$VERBOSE" = true ]; then
        if pip install pytest pytest-cov black flake8 mypy isort 2>&1 | tee "$DEV_LOG"; then
            echo -e "${GREEN}✅ Development dependencies installed${NC}"
        else
            echo -e "${YELLOW}⚠️  Some dev dependencies failed (see $DEV_LOG)${NC}"
        fi
    else
        start_spinner "Installing dev dependencies"
        if pip install pytest pytest-cov black flake8 mypy isort > "$DEV_LOG" 2>&1; then
            stop_spinner 0 "Dev dependencies installed"
            echo -e "${BLUE}ℹ️  Log: $DEV_LOG${NC}"
        else
            stop_spinner 1 "Some dev dependencies failed"
            echo -e "${BLUE}🔍 Last 20 lines:${NC}"
            tail -n 20 "$DEV_LOG"
        fi
    fi
fi

# (Directory setup moved earlier; block retained intentionally blank to preserve script flow)

# Install Playwright browsers for URL snapshots
echo ""
echo "🎭 Installing Playwright browsers for URL snapshots..."
python -m playwright install chromium
if [ $? -eq 0 ]; then
    echo -e "${GREEN}✅ Playwright browsers installed successfully${NC}"
else
    echo -e "${YELLOW}⚠️  Playwright installation had issues, but continuing...${NC}"
    echo -e "${BLUE}ℹ️  You can install manually later with: python -m playwright install chromium${NC}"
fi

# SSL Certificate Setup - Always run for enterprise security (before environment setup)
echo ""
echo "🔐 SSL Certificate Setup..."
echo "SSL certificates are required for secure enterprise operation and OAuth integration."

# Generate SSL certificate if it doesn't exist
if [ ! -f "ssl/certs/server.crt" ] || [ ! -f "ssl/private/server.key" ]; then
    echo "🔑 Generating SSL certificate for localhost..."
    if command -v python3 >/dev/null 2>&1; then
        PYTHON_CMD="python3"
    elif command -v python >/dev/null 2>&1; then
        PYTHON_CMD="python"
    else
        echo -e "${RED}❌ Python not found. SSL certificate generation requires Python.${NC}"
        echo -e "${YELLOW}⚠️  SSL setup skipped. You can generate certificates later with:${NC}"
        echo -e "${BLUE}   python tools/generate_ssl_cert.py${NC}"
    fi
    
    if [ -n "$PYTHON_CMD" ]; then
        if $PYTHON_CMD tools/generate_ssl_cert.py --hostname localhost >/dev/null 2>&1; then
            echo -e "${GREEN}✅ SSL certificate generated successfully${NC}"
        else
            echo -e "${YELLOW}⚠️  SSL certificate generation failed. You can generate manually later.${NC}"
            echo -e "${BLUE}   $PYTHON_CMD tools/generate_ssl_cert.py${NC}"
        fi
    fi
else
    echo -e "${GREEN}✅ SSL certificate already exists${NC}"
fi

# Check environment file
echo ""
echo "📋 Environment configuration..."
if [ ! -f ".env" ]; then
    if [ -f ".env.example" ]; then
        if ask_yes_no "Copy .env.example to .env for configuration?" "y"; then
            cp .env.example .env
            echo -e "${GREEN}✅ .env file created from example${NC}"
            echo -e "${YELLOW}💡 You may want to customize .env for your environment${NC}"
            
            # Add SSL configuration since we just created .env from .env.example
            echo "⚙️  Configuring SSL certificate paths in .env file..."
            echo "" >> .env
            echo "# SSL Configuration (Enterprise Security - Auto-configured by setup.sh)" >> .env
            echo "# The application always runs with HTTPS for secure enterprise operation." >> .env
            echo "# Replace with your own certificates if needed by updating the paths below." >> .env
            echo "SSL_CERT_PATH=ssl/certs/server.crt" >> .env
            echo "SSL_KEY_PATH=ssl/private/server.key" >> .env
            echo "GMAIL_REDIRECT_URI=https://localhost:3000/oauth/gmail/callback" >> .env
            echo -e "${GREEN}✅ SSL configuration added to .env${NC}"
            echo -e "${BLUE}ℹ️ Application will run on https://localhost:3000${NC}"
            echo -e "${YELLOW}💡 Update Google OAuth redirect URI to: https://localhost:3000/oauth/gmail/callback${NC}"
        else
            echo -e "${YELLOW}⚠️  .env file not found. Using default configuration.${NC}"
            echo -e "${BLUE}📋 You can copy .env.example to .env later if needed${NC}"
        fi
    else
        echo -e "${YELLOW}⚠️  No .env or .env.example file found. Using default configuration.${NC}"
    fi
else
    echo -e "${GREEN}✅ .env file already exists${NC}"
fi

echo ""
echo -e "${GREEN}✅ Environment setup complete!${NC}"
echo ""
echo "🐳 Docker Container Setup"
echo "========================="

# Check if Docker is available and running
check_docker() {
    if ! command -v docker >/dev/null 2>&1; then
        echo -e "${RED}❌ Docker is not installed.${NC}"
        echo "   Please install Docker and try again."
        return 1
    fi
    
    if ! docker info >/dev/null 2>&1; then
        echo -e "${RED}❌ Docker is not running.${NC}"
        echo "   Please start Docker and try again."
        echo -e "${BLUE}💡 You can start containers later with: docker compose up -d${NC}"
        return 1
    fi
    
    echo -e "${GREEN}✅ Docker is available and running${NC}"
    return 0
}

# Function to start specific services
start_services() {
    local services="$1"
    local description="$2"
    
    echo "🚀 Starting $description..."
    docker compose up $services -d
    
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✅ $description started successfully!${NC}"
        return 0
    else
        echo -e "${RED}❌ Failed to start $description${NC}"
        return 1
    fi
}

# Main Docker setup logic
if check_docker; then
    if [ "$AUTO_YES" = true ]; then
        # --all flag: Install everything
        echo -e "${BLUE}🤖 Auto-installing all containers...${NC}"
        start_services "postgres milvus ollama webui" "all services (PostgreSQL, Milvus, Ollama, WebUI)"
        
        if [ $? -eq 0 ]; then
            echo ""
            echo -e "${GREEN}🎉 All services are running!${NC}"
            echo -e "${BLUE}🌐 RAG Knowledge Base Web interface: http://localhost:3000${NC}"
            echo -e "${BLUE}🐘 PostgreSQL: localhost:5432${NC}"
            echo -e "${BLUE}🔍 Milvus: localhost:19530${NC}"
            echo -e "${BLUE}🤖 Ollama: localhost:11434${NC}"
        fi
        
    elif [ "$DEV_MODE" = true ]; then
        # --dev flag: Install only databases, skip webui
        echo -e "${YELLOW}👨‍💻 Development mode: Installing database containers only...${NC}"
        start_services "postgres milvus" "database services (PostgreSQL, Milvus)"
        
        if [ $? -eq 0 ]; then
            echo ""
            echo -e "${GREEN}🎉 Database services are running!${NC}"
            echo -e "${YELLOW}💡 WebUI container skipped for local development${NC}"
            echo -e "${BLUE}🐘 PostgreSQL: localhost:5432${NC}"
            echo -e "${BLUE}🔍 Milvus: localhost:19530${NC}"
            echo ""
            echo -e "${BLUE}To start local development:${NC}"
            echo "   source .venv/bin/activate"
            echo "   ./start.sh"
        fi
        
        echo ""
        
        # Ollama
        if ask_yes_no "Install and start Ollama local LLM server?" "n"; then
            start_services "ollama" "Ollama container"
            if [ $? -eq 0 ]; then
                echo -e "${BLUE}🔗 Ollama: localhost:11434${NC}"
                echo -e "${BLUE}🤖 Local LLM server ready for model downloads${NC}"
            fi
        else
            echo -e "${YELLOW}⏭️  Ollama skipped${NC}"
        fi

    else
        # Interactive mode: Ask for each service
        echo -e "${BLUE}🔧 Interactive container setup...${NC}"
        echo "   You'll be asked about each service individually."
        echo "   Each service is optional and independent."
        echo ""
        
        # PostgreSQL
        if ask_yes_no "Install and start PostgreSQL metadata database?" "y"; then
            start_services "postgres" "PostgreSQL container"
            if [ $? -eq 0 ]; then
                echo -e "${BLUE}🔗 PostgreSQL: localhost:5432${NC}"
                echo -e "${BLUE}📊 Database: rag_metadata, User: rag_user${NC}"
            fi
        else
            echo -e "${YELLOW}⏭️  PostgreSQL skipped${NC}"
        fi
        
        echo ""
        
        # Milvus
        if ask_yes_no "Install and start Milvus vector database?" "y"; then
            start_services "milvus" "Milvus container"
            if [ $? -eq 0 ]; then
                echo -e "${BLUE}🔗 Milvus: localhost:19530${NC}"
            fi
        else
            echo -e "${YELLOW}⏭️  Milvus skipped${NC}"
        fi
        
        echo ""
        
        # Ollama
        if ask_yes_no "Install and start Ollama local LLM server?" "n"; then
            start_services "ollama" "Ollama container"
            if [ $? -eq 0 ]; then
                echo -e "${BLUE}🔗 Ollama: localhost:11434${NC}"
                echo -e "${BLUE}🤖 Local LLM server ready for model downloads${NC}"
            fi
        else
            echo -e "${YELLOW}⏭️  Ollama skipped${NC}"
        fi
        
        echo ""
        
        # WebUI
        if ask_yes_no "Install and start RAG Knowledge Base WebUI container?" "n"; then
            start_services "webui" "RAG Knowledge Base WebUI container"
            if [ $? -eq 0 ]; then
                echo -e "${BLUE}🌐 RAG Knowledge Base Web interface: http://localhost:3000${NC}"
            fi
        else
            echo -e "${YELLOW}⏭️  WebUI skipped (good for local development)${NC}"
        fi
    fi
else
    echo -e "${YELLOW}⚠️  Docker setup skipped due to issues above${NC}"
    echo -e "${BLUE}💡 You can start containers manually later with:${NC}"
    echo "   docker compose up -d                              # All services"
    echo "   docker compose up postgres milvus -d              # Database services only"
    echo "   docker compose up postgres milvus ollama -d       # Databases + Ollama"
    echo "   docker compose up ollama -d                       # Ollama only"
fi

echo ""
echo "🎉 Setup Complete!"
echo "=================="

# Summary based on setup mode
if [ "$AUTO_YES" = true ]; then
    echo -e "${GREEN}✅ Fully automated setup completed${NC}"
    echo -e "${BLUE}🌐 Ready to use: http://localhost:3000${NC}"
elif [ "$DEV_MODE" = true ]; then
    echo -e "${GREEN}✅ Development environment ready${NC}"
    echo -e "${YELLOW}💻 Start local development with: python app.py${NC}"
else
    echo -e "${GREEN}✅ Interactive setup completed${NC}"
fi

echo ""
echo "📊 What was installed:"
echo "   ✅ Python virtual environment (.venv)"
echo "   ✅ All Python dependencies"
echo "   ✅ Required directories and structure"
if [ "$DEV_MODE" != true ]; then
    echo "   ✅ Docker containers (as selected)"
fi

echo ""
echo "🚀 Next Steps:"

if [ "$DEV_MODE" = true ]; then
    echo -e "${YELLOW}Development Mode - Local Execution:${NC}"
    echo -e "   1. Activate environment: ${BLUE}source .venv/bin/activate${NC}"
    echo -e "   2. Start RAG Knowledge Base application: ${BLUE}./start.sh${NC}"
    echo -e "   3. Open browser: ${BLUE}http://localhost:3000${NC}"
elif [ "$AUTO_YES" = true ]; then
    echo -e "${GREEN}Production Mode - All containers running:${NC}"
    echo "   🌐 RAG Knowledge Base Web interface: ${BLUE}http://localhost:3000${NC}"
    echo "   📊 Application ready to use immediately!"
else
    echo -e "${BLUE}Manual Options:${NC}"
    echo "   🐳 Start all containers: ${BLUE}docker compose up -d${NC}"
    echo "   💻 Or run locally: ${BLUE}source .venv/bin/activate && python app.py${NC}"
fi

echo ""
echo "🔧 Useful Commands:"
echo -e "   ${BLUE}./status.sh${NC}        # Check project status"
echo -e "   ${BLUE}./start.sh${NC}        # Start application"
echo -e "   ${BLUE}./setup.sh --help${NC}  # Show setup options"
echo -e "   ${BLUE}./uninstall.sh${NC}     # Complete removal"
echo ""
echo "🏗️  Container Management:"
echo -e "   ${BLUE}docker compose up -d${NC}                        # Start all services"
echo -e "   ${BLUE}docker compose up postgres milvus -d${NC}        # Database services only"
echo -e "   ${BLUE}docker compose up postgres milvus ollama -d${NC} # Databases + Ollama"
echo -e "   ${BLUE}docker compose down${NC}                         # Stop all services"
echo -e "   ${BLUE}docker compose logs -f webui${NC}                # View webui logs"

echo ""
echo "📚 Key Components:"
echo "   🐘 PostgreSQL (metadata): localhost:5432"
echo "   🔍 Milvus (vectors): localhost:19530"
echo "   🤖 Ollama (local LLM): localhost:11434"
echo "   🌐 Web Interface: https://localhost:3000"
echo "   📁 Document uploads: ./uploaded/"
echo "   📁 Processing staging: ./staging/"
echo "   📊 Logs: ./logs/"

echo ""
echo -e "${GREEN}🎯 RAG Knowledgebase Manager is ready!${NC}"

# Test connection if containers are running
if command -v docker >/dev/null 2>&1 && docker info >/dev/null 2>&1; then
    if docker compose ps --services --filter "status=running" | grep -q "postgres\|milvus"; then
        echo ""
        echo "🧪 Quick system test..."
        if [ -f "test_postgres.py" ]; then
            echo "   Testing PostgreSQL connection..."
            python test_postgres.py >/dev/null 2>&1 && echo -e "   ${GREEN}✅ PostgreSQL connection OK${NC}" || echo -e "   ${YELLOW}⚠️  PostgreSQL test failed${NC}"
        fi
    fi
fi

echo ""
echo -e "${GREEN}🎉 Setup complete! Your RAG Knowledge Base Manager is ready to use.${NC}"
echo -e "${BLUE}📖 Check the README.md for usage instructions and next steps.${NC}"
