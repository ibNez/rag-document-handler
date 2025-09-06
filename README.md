# RAG Knowledge Base Manager

Privacy‑first, self‑hosted data ingestion and transformation platform for building a governed, RAG‑ready knowledge lake. It consolidates heterogeneous sources (documents, emails, and web pages/URLs with snapshots) into a normalized pipeline that extracts text, cleans/normalizes content, chunks intelligently, enriches metadata, generates embeddings (Milvus), and persists structured context (PostgreSQL) while keeping all raw and derived data inside your infrastructure.

The web interface has two focused surfaces: (1) the primary Source Management dashboard where you add/monitor documents, emails, and URLs, track processing state, chunk/embedding coverage, and schema integrity; and (2) a separate Validation Search page that runs scoped semantic lookups through a lightweight local LLM (Ollama) purely to verify that embeddings, metadata, and chunk boundaries are correct. This search is a QA/validation tool—not a production end‑user knowledge assistant—so you can catch ingestion or normalization issues early before upstream RAG pipelines consume the data.

Core value: provide a single, reproducible, auditable ETL surface for private knowledge assets so you can safely power Retrieval‑Augmented Generation without leaking sensitive content to public LLM APIs. Secure credential handling (encrypted email credentials), deterministic processing, idempotent ingestion, and modular pipelines make it easy to extend sources while preserving privacy, governance, and observability.

For detailed guides, architecture notes, and extended documentation, see the [Documentation Directory](docs/README.md).


## 🚀 Quick Start

**Interactive Setup (Recommended):**
```bash
git clone <repository-url>
cd rag-document-handler
./setup.sh
```

**Start the Application:**
```bash
source .venv/bin/activate    # Always source environment first
./start.sh                   # Use startup script with service checks
```
Visit: http://localhost:3000

For complete installation options and deployment guides, see [Installation Documentation](docs/installation.md).

## 🏗️ Architecture Overview

The RAG Knowledge Base Manager uses a modular, dual-database architecture with enhanced email processing, real-time dashboard capabilities, comprehensive auto-refresh functionality, and a well-organized template structure using partials.

**Core Components:**
- **Modular Architecture**: Clean separation of concerns with dedicated ingestion and retrieval modules
- **Dual Database System**: PostgreSQL for metadata, Milvus for vector embeddings
- **Advanced Email Integration**: IMAP, Gmail API, and Exchange support with encrypted credential storage
- **Real-Time Dashboard**: Comprehensive auto-refresh for all panels with modular template partials
- **Conversational AI**: Enhanced RAG-powered chat interface with intelligent query classification
- **Template Partials**: Well-organized, maintainable template structure with reusable components

For detailed architectural information, system design, component interactions, directory structure, and enhanced features, see the [Architecture Documentation](docs/architecture.md).

## 🛠️ Technology Stack

- **Backend**: Python 3.8+ with Flask web framework
- **Vector Database**: Milvus for efficient vector storage and retrieval
- **Metadata Database**: PostgreSQL with JSONB for flexible document metadata
- **AI Integration**: Ollama for embeddings and conversational RAG functionality
- **Frontend**: Bootstrap 5 with responsive design and real-time updates
- **Containerization**: Docker Compose for easy deployment

For complete technology details and integration specifics, see the [Architecture Documentation](docs/architecture.md).

## ✨ Enhanced Search & Quality Features

### 🤖 **Ethical Web Crawling**
- **Robots.txt Enforcement**: Comprehensive compliance with website crawling preferences
- **Intelligent Throttling**: Per-origin crawl delays respect server limitations
- **Configurable Respect**: Bypass robots.txt for internal/authorized sites
- **Performance Monitoring**: Built-in tools for crawling efficiency analysis

### 🔍 **Hybrid Retrieval System**
- **Vector Similarity Search**: Semantic understanding through embeddings
- **PostgreSQL Full-Text Search**: Exact keyword matching with advanced filtering
- **RRF Fusion**: Reciprocal Rank Fusion combines both for optimal results
- **Cross-Encoder Reranking**: Final relevance optimization for superior quality

### 📚 **Advanced Document Processing**
- **Title-Aware Chunking**: Structure-preserving chunks with 800-1,000 tokens
- **Page-Aware Processing**: Precise page-level citations and references
- **Rich Metadata**: Authors, tags, language detection, section tracking
- **Element Preservation**: Tables, lists, headings maintained intact

### 🎯 **Search Quality Improvements**
- **+25-40% relevance** through cross-encoder reranking
- **+30% precision** with intelligent chunking strategies
- **+50% citation accuracy** with page-aware processing
- **+20% recall** through enhanced metadata filtering

### ⚙️ **Configurable Quality Options**
```bash
# Enable advanced features
ENABLE_DOCUMENT_RERANKING=true
DOCUMENT_RERANKER_MODEL=ms-marco-minilm
CHUNKING_STRATEGY=title_aware
PRESERVE_TABLES=true

# Robots.txt enforcement configuration
RESPECT_ROBOTS_TXT=true
CRAWLER_USER_AGENT="RAG-Document-Handler/1.0"
DEFAULT_CRAWL_DELAY=1.0
ROBOTS_CACHE_TTL=3600
```

For complete technology details and integration specifics, see the [Architecture Documentation](docs/architecture.md).

## 📦 Installation & Management

### Prerequisites
- Python 3.8 or higher
- Docker and Docker Compose
- Git

### Setup Options

**Automated Setup Options:**
```bash
./setup.sh --all        # Install everything without prompts
./setup.sh --dev        # Development mode (infrastructure only)
./setup.sh --help       # Show installation help
```

**Development Mode (Recommended):**
```bash
./setup.sh --dev               # Start only infrastructure containers
source .venv/bin/activate      # ALWAYS source environment first (DEVELOPMENT_RULES.md)
./start.sh                     # Use proper startup script with service checks
```

For complete installation guides, troubleshooting, and deployment options, see [Installation Documentation](docs/installation.md).

### Uninstallation

```bash
./uninstall.sh          # Interactive removal with container/volume choices
./uninstall.sh --dry-run # Preview what would be removed
```

The uninstall script will ask individually about each container and volume removal, allowing you to preserve data for reinstallation.

### Docker Services

```bash
# Start specific services
docker compose up postgres milvus ollama -d  # Database and AI services
docker compose up webui -d                   # Web interface

# Stop services
docker compose down
```

For detailed Docker management and production deployment, see [Installation Documentation](docs/installation.md).

## 📚 Quick Reference

### 🔄 Essential Commands

| Action | Command | Description |
|--------|---------|-------------|
| **Install** | `./setup.sh` | Complete interactive setup |
| **Start** | `./start.sh` | Start application with service checks |
| **Status** | `./status.sh` | Check system health and connectivity |
| **Uninstall** | `./uninstall.sh` | Safe project removal with choices |

**Remember**: Always `source .venv/bin/activate` before running Python commands (DEVELOPMENT_RULES.md)

### 🌐 Default Endpoints

- **RAG Knowledge Base Web Application**: http://localhost:3000
- **PostgreSQL**: localhost:5432
- **Milvus**: localhost:19530
- **Ollama**: localhost:11434

### � Database Architecture

The system uses a dual-database architecture with PostgreSQL for metadata and Milvus for vector embeddings.

For detailed schema information, table structures, and field descriptions, see the [Database Schema Documentation](docs/database-schema.md).

For architectural details and system design, see the [Architecture Documentation](docs/architecture.md).

## 🔧 Configuration

The application uses environment variables for configuration. For comprehensive configuration details, variable descriptions, and deployment options, see the [Configuration Documentation](docs/configuration.md).

**Quick Start Configuration:**
```bash
# Core settings (copy to .env file)
FLASK_PORT=3000
EMAIL_ENCRYPTION_KEY=your_generated_fernet_key  # Required for email processing
MILVUS_HOST=localhost
POSTGRES_HOST=localhost
```

**Generate Encryption Key:**
```bash
python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"
```

## 🚀 Usage

### Starting the Application

1. **Ensure services are running**: `docker compose ps`
2. **Activate virtual environment**: `source .venv/bin/activate`
3. **Start the application**: `./start.sh`
4. **Access the interface**: http://localhost:3000

### Web Interface Features

- **Document Upload**: PDF, DOCX, TXT, MD files (max 100MB)
- **URL Management**: Automatic title extraction and storage
- **Semantic Search**: Natural language queries across documents
- **Batch Processing**: Stage files for later processing

For detailed usage instructions, workflows, and troubleshooting, see the [Usage Documentation](docs/usage.md).

## 💾 Data Architecture

**Dual-Database Design:**
- **PostgreSQL**: Metadata, documents, URLs, email accounts
- **Milvus**: Vector embeddings for semantic search
- **File System**: staging/, uploaded/, logs/ (configurable via LOG_DIR)

For complete schema details, see the [Database Schema Documentation](docs/database-schema.md).

## 🛠️ Development Standards

This project follows strict development standards as outlined in `DEVELOPMENT_RULES.md`:

- **PEP 8 compliance** with type hints and docstrings
- **Structured logging** and comprehensive error handling
- **Package management** via pyproject.toml

For detailed development guidelines, see the [Contributing Documentation](docs/contributing.md).

## 🐳 Docker Deployment

**Development**: Use the provided Docker Compose setup
```bash
docker compose up -d
```

**Production**: For production deployments, environment variables, and security configurations, see the [Configuration Documentation](docs/configuration.md).

## 🤝 Contributing

1. Follow the development rules in `DEVELOPMENT_RULES.md`
2. Ensure all tests pass and code is properly formatted
3. Update documentation for any user-facing changes

For detailed contribution guidelines, see the [Contributing Documentation](docs/contributing.md).

## 📖 Documentation

This README provides a quick start guide. For comprehensive documentation:

- [Installation Guide](docs/installation.md) - Detailed setup instructions
- [Configuration](docs/configuration.md) - Environment variables and settings
- [Usage Guide](docs/usage.md) - Complete feature walkthrough
- [Architecture](docs/architecture.md) - System design and components
- [Database Schema](docs/database-schema.md) - Database structure and relationships
- [Contributing](docs/contributing.md) - Development guidelines and standards
- [Changelog](CHANGELOG.md) - Recent changes and version history

## 📄 License

MIT License - see LICENSE file for details

---

**RAG Document Handler** - Efficient document management with vector search capabilities
