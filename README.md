# RAG Document Handler

A comprehensive document management system with refactored modular architecture for storing and retrieving document embeddings in a Milvus vector database and metadata in PostgreSQL for use with RAG (Retrieval-Augmented Generation) applications.

For extended guides and architecture notes, see the [documentation directory](docs/README.md).

## 🏗️ Refactored Architecture

The application has been restructured into clean, modular components:

```
ingestion/              # Modular data ingestion
├── core/              # Database abstractions
├── email/             # Email processing pipeline
├── url/               # URL crawling and management  
├── document/          # Document extraction
└── utils/             # Shared utilities

rag_manager/           # RAG functionality
├── managers/          # Vector operations
└── web/              # Web interface
```

## 🚀 Features

- **Modular Architecture**: Clean separation of concerns with dedicated modules for email, URL, and document processing
- **Document Upload & Management**: Enhanced document processing with improved chunking and metadata extraction
- **Smart URL Management**: Refactored URL orchestrator with PostgreSQL-based scheduling and content processing
- **Vector Embeddings**: Automatic text extraction and embedding generation using Ollama embeddings
- **Dual Database Architecture**: 
  - **Milvus**: Vector embeddings and similarity search with comprehensive logging
  - **PostgreSQL**: Document metadata, analytics managed through dedicated managers
- **Email Integration**: Modular email system supporting IMAP, Gmail API, and Exchange with encrypted credential storage
- **Conversational AI**: Enhanced RAG-powered chat interface with improved error handling and debugging
- **Semantic Search**: Find relevant documents using natural language queries with detailed logging
- **Web Interface**: Clean, responsive Flask web application with Bootstrap UI and proper error handling
- **Background Processing**: Coordinated processing through orchestrator classes

## 📋 How It Works

1. **Content Processing**: Documents (PDFs, text files, etc.) are uploaded and processed for text extraction
2. **Metadata Storage**: Document metadata, URLs, and email data stored in PostgreSQL with JSONB for flexible attributes
3. **Vector Embedding**: The embedding model transforms text content into numerical vectors representing semantic meaning
4. **Vector Storage**: Generated embeddings are stored in a Milvus vector database optimized for high-dimensional vectors
5. **Semantic Retrieval**: Applications can query both databases for comprehensive search and content recommendation
6. **Conversational AI**: RAG system combines retrieved documents with LLM to provide intelligent responses

## 🛠️ Technology Stack

- **Backend**: Python 3.8+ with Flask web framework
- **Vector Database**: Milvus for efficient vector storage and retrieval
- **Metadata Database**: PostgreSQL with JSONB for flexible document metadata
- **Web Scraping**: BeautifulSoup4 and Requests for automatic URL title extraction
- **AI Integration**: Ollama for embeddings and conversational RAG functionality
- **ML Framework**: Ollama embeddings (mxbai-embed-large)
- **Frontend**: Bootstrap 5 with responsive design
- **File Processing**: pypdf, python-docx for document parsing
- **Background Processing**: Threading for UI responsiveness
- **Containerization**: Docker Compose for easy deployment

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
- Stop and remove RAG Document Handler containers and volumes
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

```
┌─────────────────┬─────────────────┐
│   PostgreSQL    │     Milvus      │
│   (Metadata)    │   (Vectors)     │
├─────────────────┼─────────────────┤
│ • Documents     │ • Embeddings    │
│ • URLs          │ • Similarity    │ 
│ • Emails        │   Search        │
│ • Accounts      │ • Collections   │
│ • Analytics     │                 │
└─────────────────┴─────────────────┘
```

### 🌐 Default Endpoints

- **Web Application**: http://localhost:3000
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

The application uses environment variables for configuration. Copy the `.env` file and modify as needed:

```bash
# Milvus connection settings
MILVUS_HOST=localhost
MILVUS_PORT=19530

# Flask settings
FLASK_HOST=0.0.0.0
FLASK_PORT=5000
FLASK_DEBUG=True

# File upload settings
MAX_CONTENT_LENGTH=16777216  # 16MB
UPLOAD_FOLDER=staging

# Vector database settings
COLLECTION_NAME=documents
VECTOR_DIM=384
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

The application automatically creates and manages the following data stores:

- **`knowledgebase.db`**: SQLite database storing URL metadata and, in the future, email content
- **Milvus Collections**: Vector embeddings stored in Milvus database
- **Staging/Uploaded Folders**: Document file organization

### URL Database Schema

```sql
CREATE TABLE urls (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    url TEXT UNIQUE NOT NULL,
    title TEXT,                          -- Automatically extracted from web page
    description TEXT,
    added_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_checked TIMESTAMP,
    status TEXT DEFAULT 'active'
);
```

## 🐳 Docker Deployment


## 🛡️ Development Standards

This project follows strict development standards as outlined in `DEVELOPMENT_RULES.md`:

- **PEP 8 compliance** with proper type hints and docstrings
- **Structured logging** for all operations
- **Error handling** with specific exception types
- **Package management** via pyproject.toml (no requirements.txt)
- **Code organization** following Python best practices

## 🐳 Docker Deployment

For production deployment, ensure Milvus is properly configured:

```bash
# Production Milvus setup
docker run -d \
  --name milvus \
  -p 19530:19530 \
  -v milvus_data:/var/lib/milvus \
  milvusdb/milvus:latest
```

## 🤝 Contributing

1. Follow the development rules in `DEVELOPMENT_RULES.md`
2. Ensure all tests pass and code is properly formatted
3. Update documentation for any user-facing changes
4. Use type hints and comprehensive docstrings

## 📄 License

MIT License - see LICENSE file for details

## 🔗 References

- [LangChain Milvus Integration](https://python.langchain.com/docs/integrations/vectorstores/milvus/)
- [Milvus Documentation](https://milvus.io/docs)
- [Flask Documentation](https://flask.palletsprojects.com/)

---

**RAG Document Handler** - Efficient document management with vector search capabilities
