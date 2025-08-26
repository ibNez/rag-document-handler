#!/bin/bash

# RAG Knowledge Base Manager - Setup Script
# This script sets up the development environment and installs dependencies

# Script configuration
SCRIPT_VERSION="2.0"
AUTO_YES=false
DEV_MODE=false
SHOW_HELP=false

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

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
    echo "               • Starts all Docker containers (webui, postgres, milvus)"
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
    echo "      • webui            (web interface) - only with --all"
    echo ""
    echo "AFTER SETUP:"
    echo "  🌐 Web interface: http://localhost:3000 (if webui container running)"
    echo "  🔧 Local development: source .venv/bin/activate && python app.py"
    echo "  📊 Status check: ./status.sh"
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

# Upgrade pip
echo "⬆️ Upgrading pip..."
pip install --upgrade pip >/dev/null 2>&1

# Install dependencies
echo "📚 Installing Python dependencies..."
pip install -e . >/dev/null 2>&1
if [ $? -eq 0 ]; then
    echo -e "${GREEN}✅ Dependencies installed successfully${NC}"
else
    echo -e "${YELLOW}⚠️  Package installation had issues, trying alternative method...${NC}"
    pip install flask pymilvus sentence-transformers werkzeug python-dotenv pypdf python-docx chardet requests beautifulsoup4 google-api-python-client google-auth psycopg2-binary >/dev/null 2>&1
fi

# Install development dependencies (optional)
if ask_yes_no "Install development dependencies (pytest, black, flake8, mypy)?" "n"; then
    echo "🛠️ Installing development dependencies..."
    pip install pytest pytest-cov black flake8 mypy isort >/dev/null 2>&1
    echo -e "${GREEN}✅ Development dependencies installed${NC}"
fi

# Create required directories
echo ""
echo "📁 Setting up directory structure..."
directories=("staging" "uploaded" "deleted" "logs" "databases/milvus/db" "databases/milvus/conf" "databases/postgres" "logs/milvus" "logs/postgres")

for dir in "${directories[@]}"; do
    if [ ! -d "$dir" ]; then
        mkdir -p "$dir"
        echo "   ✅ Created: $dir"
    else
        echo "   ✅ Exists: $dir"
    fi
done

# Check environment file
echo ""
echo "� Environment configuration..."
if [ ! -f ".env" ]; then
    if [ -f ".env.example" ]; then
        if ask_yes_no "Copy .env.example to .env for configuration?" "y"; then
            cp .env.example .env
            echo -e "${GREEN}✅ .env file created from example${NC}"
            echo -e "${YELLOW}💡 You may want to customize .env for your environment${NC}"
        else
            echo -e "${YELLOW}⚠️  .env file not found. Using default configuration.${NC}"
            echo -e "${BLUE}� You can copy .env.example to .env later if needed${NC}"
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
        start_services "postgres milvus webui" "all services (PostgreSQL, Milvus, WebUI)"
        
        if [ $? -eq 0 ]; then
            echo ""
            echo -e "${GREEN}🎉 All services are running!${NC}"
            echo -e "${BLUE}🌐 RAG Knowledge Base Web interface: http://localhost:3000${NC}"
            echo -e "${BLUE}🐘 PostgreSQL: localhost:5432${NC}"
            echo -e "${BLUE}🔍 Milvus: localhost:19530${NC}"
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
            echo "   python app.py"
        fi
        
    else
        # Interactive mode: Ask for each service
        echo -e "${BLUE}� Interactive container setup...${NC}"
        
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
    echo "   docker compose up -d                    # All services"
    echo "   docker compose up postgres milvus -d    # Database services only"
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
echo -e "   ${BLUE}docker compose up -d${NC}              # Start all services"
echo -e "   ${BLUE}docker compose up postgres milvus -d${NC} # Database services only"
echo -e "   ${BLUE}docker compose down${NC}                # Stop all services"
echo -e "   ${BLUE}docker compose logs -f webui${NC}       # View webui logs"

echo ""
echo "📚 Key Components:"
echo "   🐘 PostgreSQL (metadata): localhost:5432"
echo "   🔍 Milvus (vectors): localhost:19530"
echo "   🌐 Web Interface: localhost:3000"
echo "   📁 Document uploads: ./uploaded/"
echo "   � Processing staging: ./staging/"
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
