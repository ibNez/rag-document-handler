# RAG Document Handler

A comprehensive document management system for storing and retrieving document embeddings in a Milvus vector database for use with RAG (Retrieval-Augmented Generation) applications.

## 🚀 Features

- **Document Upload & Management**: Upload documents to a staging area with support for multiple file formats: PDF, DOCX, DOC, TXT, and Markdown files
- **Vector Embeddings**: Automatic text extraction and embedding generation using SentenceTransformers all-MiniLM-L6-v2 model
- **Milvus Integration**: Store and retrieve document embeddings in Milvus vector database
- **Semantic Search**: Find relevant documents using natural language queries
- **Web Interface**: Clean, responsive Flask web application with Bootstrap UI
- **Single Interface**: No backend API - all functionality through web interface
- **Threading**: Background processing prevents UI freezing during long operations

## 📋 How It Works

1. **Content Processing**: Documents (PDFs, text files, etc.) are uploaded and processed for text extraction
2. **Vector Embedding**: The embedding model transforms text content into numerical vectors representing semantic meaning
3. **Vector Storage**: Generated embeddings are stored in a Milvus vector database optimized for high-dimensional vectors
4. **Semantic Retrieval**: Applications can query the database for semantic search and content recommendation
5. **Create Milvus Database**: ```
milvus_client.create_collection(
    collection_name=collection_name,
    dimension=embedding_dim,
    metric_type="IP",  # Inner product distance
    consistency_level="Bounded",  # Supported values are (`"Strong"`, `"Session"`, `"Bounded"`, `"Eventually"`). See https://milvus.io/docs/consistency.md#Consistency-Level for more details.
)```

## 🛠️ Technology Stack

- **Backend**: Python 3.8+ with Flask web framework (Single interface)
- **Vector Database**: Milvus for efficient vector storage and retrieval
- **ML Framework**: SentenceTransformers for document embeddings
- **Embedding Model**: SentenceTransformers all-MiniLM-L6-v2 (configurable)
- **Frontend**: Bootstrap 5 with responsive design
- **File Processing**: pypdf, python-docx for document parsing
- **Threading**: Background processing for UI responsiveness

## 📦 Installation

### Prerequisites

- Python 3.8 or higher
- Docker (for running Milvus)
- Git

### Quick Setup

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd "RAG Document Handler"
   ```

2. **Run the setup script**:
   ```bash
   ./setup.sh
   ```

3. **Start the application**:
   ```bash
   ./start.sh
   ```

   This will:
   - Start Milvus database (if not running)
   - Activate the virtual environment
   - Launch the web application

### Manual Installation

1. **Create virtual environment**:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

2. **Install dependencies**:
   ```bash
   pip install flask pymilvus sentence-transformers werkzeug python-dotenv pypdf python-docx chardet
   ```

3. **Start Milvus database**:
   ```bash
   docker run -d --name milvus -p 19530:19530 milvusdb/milvus:latest
   ```

4. **Run the application**:
   ```bash
   python app.py
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

1. **Ensure Milvus is running**:
   ```bash
   docker ps | grep milvus
   ```

2. **Activate virtual environment**:
   ```bash
   source .venv/bin/activate
   ```

3. **Start the web server**:
   ```bash
   python app.py
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
   - Return results that show as relivent document associated with the result

4. **Manage Documents**:
   - **Staging Files**: Files waiting to be processed
   - **Uploaded Files**: Files that have been processed and are in the database
   - View database statistics and file information
   - Delete files from staging, or uploaded folder and database


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
3. Update documentation for any API changes
4. Use type hints and comprehensive docstrings

## 📄 License

MIT License - see LICENSE file for details

## 🔗 References

- [LangChain Milvus Integration](https://python.langchain.com/docs/integrations/vectorstores/milvus/)
- [Milvus Documentation](https://milvus.io/docs)
- [Flask Documentation](https://flask.palletsprojects.com/)

---

**RAG Document Handler** - Efficient document management with vector search capabilities
