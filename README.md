# RAG Knowledge Base Manager

A comprehensive document management system with refactored modular architecture for storing and retrieving document embeddings in a Milvus vector database and metadata in PostgreSQL for use with RAG (Retrieval-Augmented Generation) applications.

For extended guides and architecture notes, see the [documentation directory](docs/README.md).

## 🏗️ Architecture

The RAG Knowledge Base Manager uses a modular, dual-database architecture with enhanced email processing, real-time dashboard capabilities, and comprehensive auto-refresh functionality.

For detailed architectural information, system design, component interactions, and enhanced features, see the [Architecture Documentation](docs/architecture.md).

## 🚀 Features

- **Modular Architecture**: Clean separation of concerns with dedicated modules for email, URL, and document processing
- **Enhanced Email Processing**: Robust IMAP connector with corrupted email detection, offset-based error logging, and proper validation
- **Document Upload & Management**: Enhanced document processing with improved chunking and metadata extraction
- **Smart URL Management**: Refactored URL orchestrator with PostgreSQL-based scheduling and content processing
- **Vector Embeddings**: Automatic text extraction and embedding generation using Ollama embeddings
- **Dual Database Architecture**: 
  - **Milvus**: Vector embeddings and similarity search with comprehensive logging
  - **PostgreSQL**: Document metadata, analytics managed through dedicated managers
- **Advanced Email Integration**: IMAP, Gmail API, and Exchange support with encrypted credential storage
- **Real-Time Dashboard**: Comprehensive auto-refresh for all panels (10-second intervals)
- **Conversational AI**: Enhanced RAG-powered chat interface with improved error handling and debugging
- **Semantic Search**: Find relevant documents using natural language queries with detailed logging
- **Web Interface**: Clean, responsive Flask web application with Bootstrap UI and comprehensive auto-refresh
- **Background Processing**: Coordinated processing through orchestrator classes with fail-fast error handling

## 📋 How It Works

1. **Content Processing**: Documents (PDFs, text files, etc.) are uploaded and processed for text extraction
2. **Metadata Storage**: Document metadata, URLs, and email data stored in PostgreSQL with JSONB for flexible attributes
3. **Vector Embedding**: The embedding model transforms text content into numerical vectors representing semantic meaning
4. **Vector Storage**: Generated embeddings are stored in a Milvus vector database optimized for high-dimensional vectors
5. **Semantic Retrieval**: Applications can query both databases for comprehensive search and content recommendation
6. **Conversational AI**: RAG system combines retrieved documents with LLM to provide intelligent responses

## 🛠️ Technology Stack

- **Backend**: Python 3.8+ with Flask web framework and Werkzeug
- **Vector Database**: Milvus for efficient vector storage and retrieval
- **Metadata Database**: PostgreSQL with JSONB for flexible document metadata
- **Web Framework**: Flask with Jinja2 templating and session management
- **Email Processing**: 
  - IMAP with email-validator for robust email handling
  - Encrypted credential storage with Fernet encryption
  - Offset-based processing with corruption detection
- **Web Scraping**: BeautifulSoup4 and Requests for automatic URL title extraction
- **AI Integration**: Ollama for embeddings and conversational RAG functionality
- **ML Framework**: Ollama embeddings (mxbai-embed-large)
- **Frontend**: 
  - Bootstrap 5 with responsive design
  - JavaScript with auto-refresh functionality
  - Real-time progress tracking and status updates
  - Local timezone formatting
- **File Processing**: pypdf, python-docx for document parsing
- **Background Processing**: Threading for UI responsiveness with coordinated orchestrators
- **Database Connectivity**: psycopg2 for PostgreSQL, pymilvus for vector operations
- **Containerization**: Docker Compose for easy deployment
- **Development**: Following strict development rules with fail-fast error handling

## 📦 Installation & Management

### Prerequisites

- Python 3.8 or higher
- Docker and Docker Compose
- Git

### 🚀 Complete Installation

**Interactive Setup (Recommended):**
```bash
git clone <repository-url>
cd rag-document-handler
./setup.sh
```

**Automated Setup Options:**
```bash
./setup.sh --all        # Install everything without prompts
./setup.sh --dev        # Development mode (infrastructure only)
./setup.sh --help       # Show installation help
```

The setup script will:
- Create Python virtual environment
- Install all dependencies (including PostgreSQL drivers)
- Create directory structure
- Start Docker containers (Milvus + PostgreSQL)
- Test database connections
- Configure environment files

**Development Mode (Recommended):**
```bash
./setup.sh --dev               # Start only infrastructure containers
source .venv/bin/activate      # ALWAYS source environment first (DEVELOPMENT_RULES.md)
./start.sh                     # Use proper startup script with service checks
```

**Start the Application:**
```bash
source .venv/bin/activate    # Always source environment first
./start.sh                   # Use startup script with service checks
```
Visit: http://localhost:3000

### 🗑️ Complete Uninstallation

**Safe Removal:**
```bash
./uninstall.sh          # Interactive removal
./uninstall.sh --dry-run # Preview what would be removed
./uninstall.sh --help   # Show uninstall help
```

The uninstall script will:
- Stop and remove RAG Knowledgebase Manager containers and volumes
- Remove Python virtual environment (.venv)
- Clean up all database files and logs
- Remove uploaded/staging files
- Clean temporary and cache files
- Preserve other Docker containers and system files

**Reinstall:**
```bash
./setup.sh  # Start fresh installation
```

### 🐳 Docker Services

**Individual Service Management:**
```bash
# Start specific services
docker compose up postgres -d  # PostgreSQL metadata database
docker compose up milvus -d    # Milvus vector database

# Start all services
docker compose up -d

# Stop services
docker compose down

# Remove with volumes
docker compose down --volumes
```

### 🧪 Testing & Validation

**Test Database Connections:**
```bash
python test_postgres.py  # Test PostgreSQL connectivity
```

**Check Service Status:**
```bash
docker compose ps         # View running containers
docker compose logs       # View container logs
```

## 📚 Quick Reference

### 🔄 Project Management Commands

| Action | Command | Description |
|--------|---------|-------------|
| **Install** | `./setup.sh` | Complete interactive setup |
| **Install All** | `./setup.sh --all` | Automated setup without prompts |
| **Dev Setup** | `./setup.sh --dev` | Development mode (infrastructure only) |
| **Start** | `./start.sh` | Start application with service checks |
| **Status** | `./status.sh` | Check system health and connectivity |
| **Preview** | `./uninstall.sh --dry-run` | Preview removal without changes |
| **Uninstall** | `./uninstall.sh` | Safe project removal |
| **Services** | `docker compose up -d` | Start all database services |
| **Stop** | `docker compose down` | Stop all services |
| **Clean** | `docker compose down --volumes` | Stop and remove data |

**Remember**: Always `source .venv/bin/activate` before running Python commands (DEVELOPMENT_RULES.md)

### 🗃️ Database Architecture

The system uses a dual-database architecture with PostgreSQL for metadata and Milvus for vector embeddings. For detailed schema information, table structures, and field descriptions, see the [Database Schema Documentation](docs/database-schema.md).

For architectural details and system design, see the [Architecture Documentation](docs/architecture.md).

### 🌐 Default Development Endpoints

- **RAG Knowledge Base Web Application**: http://localhost:3000
- **PostgreSQL**: localhost:5432
- **Milvus**: localhost:19530

### 📁 Directory Structure

```
rag-document-handler/
├── app.py                 # Main application
├── setup.sh              # Installation script  
├── uninstall.sh          # Cleanup script
├── test_postgres.py      # Database test
├── docker-compose.yml    # Container configuration
├── databases/            # Database files
│   ├── postgres/         # PostgreSQL data
│   └── milvus/          # Milvus data
├── logs/                 # Application logs
├── staging/              # File upload staging
├── uploaded/             # Processed files
└── ingestion/            # Database modules
```

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

1. **Ensure services are running**:
   ```bash
   docker compose ps  # Check Milvus and PostgreSQL
   ```

2. **Activate virtual environment**:
   ```bash
   source .venv/bin/activate  # Always source first (DEVELOPMENT_RULES.md)
   ```

3. **Start the web server**:
   ```bash
   ./start.sh  # Use startup script with service checks
   ```

4. **Access the application**:
   Open your browser to `http://localhost:3000`

### Using the Web Interface

1. **Upload Documents**: 
   - **Upload to Staging**: Upload files to the staging area for later batch processing
   - **Add to Database**: Immediately process and add documents to the vector database
   - Supported formats: PDF, DOCX, DOC, TXT, MD
   - Maximum file size: 100MB

2. **Process Staged Documents**:
   - Click "Process" on files in the staging area
   - Documents are parsed, chunked, embedded
   - Embeddings are stored in Milvus database
   - Document moved to the uploaded folder

3. **Search Documents**:
   - Use the search page to find relevant content
   - Enter natural language queries
   - Return results that show as relevant document associated with the result

4. **Manage URLs**:
   - **Add URLs**: Simply paste any URL and the system automatically extracts the page title
   - **View Collection**: Browse all stored URLs with their extracted titles
   - **Open Links**: Click any URL to open it in a new tab
   - **Delete URLs**: Remove URLs you no longer need
   - **Automatic Title Extraction**: No manual entry needed - titles are scraped from web pages

5. **Manage Documents**:
   - **Staging Files**: Files waiting to be processed
   - **Uploaded Files**: Files that have been processed and are in the database
   - View database statistics and file information
   - Delete files from staging, or uploaded folder and database

## 💾 Data Stores

The application uses a dual-database architecture for optimal performance and scalability:

- **PostgreSQL Database**: Stores all metadata including documents, URLs, email accounts, and email messages with JSONB support for flexible attributes
- **Milvus Vector Database**: Stores high-dimensional vector embeddings for semantic search and retrieval
- **File System Storage**: 
  - `staging/` - Temporary storage for uploaded files awaiting processing
  - `uploaded/` - Processed documents and URL snapshots
  - `logs/` - Application and service logs

For detailed database schemas, table structures, field descriptions, and relationships, see the [Database Schema Documentation](docs/database-schema.md).

## ️ Development Standards

This project follows strict development standards as outlined in `DEVELOPMENT_RULES.md`:

- **PEP 8 compliance** with proper type hints and docstrings
- **Structured logging** for all operations
- **Error handling** with specific exception types
- **Package management** via pyproject.toml (no requirements.txt)
- **Code organization** following Python best practices

## 🐳 Docker Deployment

The application uses Docker Compose for container orchestration. For production deployment, ensure all services are properly configured:

### Development Deployment
```bash
# Use the provided Docker Compose setup
docker compose up -d
```

### Production Deployment

**PostgreSQL Database:**
```bash
docker run -d \
  --name rag-postgres \
  -p 5432:5432 \
  -e POSTGRES_DB=rag_metadata \
  -e POSTGRES_USER=rag_user \
  -e POSTGRES_PASSWORD=secure_production_password \
  -v postgres_data:/var/lib/postgresql/data \
  postgres:15
```

**Milvus Vector Database:**
```bash
docker run -d \
  --name rag-milvus \
  -p 19530:19530 \
  -v milvus_data:/var/lib/milvus \
  milvusdb/milvus:latest
```

**RAG Application:**
```bash
docker run -d \
  --name rag-app \
  -p 3000:3000 \
  --link rag-postgres:postgres \
  --link rag-milvus:milvus \
  -e POSTGRES_HOST=postgres \
  -e MILVUS_HOST=milvus \
  -e EMAIL_ENCRYPTION_KEY=your_production_key \
  -v uploaded_data:/app/uploaded \
  -v staging_data:/app/staging \
  your-registry/rag-document-handler:latest
```

For detailed deployment configurations and environment variables, see the [Configuration Documentation](docs/configuration.md).

## 🤝 Contributing

1. Follow the development rules in `DEVELOPMENT_RULES.md`
2. Ensure all tests pass and code is properly formatted
3. Update documentation for any user-facing changes
4. Use type hints and comprehensive docstrings

## 📄 License

MIT License - see LICENSE file for details

## 🔗 References

### Core Technologies
- [PostgreSQL Documentation](https://www.postgresql.org/docs/) - Metadata database system
- [Milvus Documentation](https://milvus.io/docs) - Vector database and similarity search
- [Flask Documentation](https://flask.palletsprojects.com/) - Web framework
- [Docker Documentation](https://docs.docker.com/) - Containerization platform

### AI/ML Frameworks
- [Ollama Documentation](https://ollama.com/docs) - Local LLM and embeddings service
- [LangChain Milvus Integration](https://python.langchain.com/docs/integrations/vectorstores/milvus/) - Vector store integration

### Python Libraries
- [psycopg2 Documentation](https://www.psycopg.org/docs/) - PostgreSQL adapter for Python
- [pymilvus Documentation](https://milvus.io/docs/install_pymilvus.md) - Python SDK for Milvus
- [BeautifulSoup4 Documentation](https://www.crummy.com/software/BeautifulSoup/bs4/doc/) - Web scraping library
- [Bootstrap 5 Documentation](https://getbootstrap.com/docs/5.0/) - Frontend framework

### Email Processing
- [IMAP Protocol RFC](https://tools.ietf.org/html/rfc3501) - Internet Message Access Protocol
- [Cryptography Documentation](https://cryptography.io/) - Password encryption library

---

**RAG Knowledge Base Manager** - Efficient document management with vector search capabilities
