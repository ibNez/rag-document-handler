# System Architecture

The RAG Document Handler uses a dual database architecture with Flask web application, PostgreSQL for metadata, and Milvus for vector embeddings.

## Components

| Component | Description |
|-----------|-------------|
| `app.py` | Main Flask application with `RAGDocumentHandler` class implementing upload, URL management, embedding generation, and search |
| `ingestion/` | Database management modules for PostgreSQL and Milvus integration |
| `templates/` | Jinja2 templates for the responsive web UI |
| `static/` | Static assets (CSS, JS, images) used by the templates |
| `setup.sh` | Enhanced setup script with `--all`, `--dev`, and `--help` flags |
| `uninstall.sh` | Safe removal script with `--dry-run` and project-specific cleanup |
| `docker-compose.yml` | Container orchestration for PostgreSQL and Milvus |

## Database Architecture

### PostgreSQL (Metadata Storage)
- **Documents**: File metadata, processing statistics, ingestion timestamps
- **URLs**: Web page management, title extraction, refresh scheduling  
- **Emails**: Account management, message processing, duplicate detection
- **JSONB Fields**: Flexible attribute storage for extensibility
- **Full-text Search**: Built-in PostgreSQL search capabilities

### Milvus (Vector Storage)
- **Collections**: Organized by content type (documents, urls, emails)
- **Embeddings**: High-dimensional vectors from Ollama embeddings
- **Similarity Search**: Efficient nearest neighbor retrieval
- **Metadata**: Associated document IDs and chunk information

## Data Flow

1. **Content Ingestion**: Users upload documents or submit URLs through web interface
2. **Text Processing**: Content extracted and chunked for optimal embedding
3. **Metadata Storage**: Document information stored in PostgreSQL with JSONB attributes
4. **Vector Generation**: Text embedded using Ollama (mxbai-embed-large model)
5. **Vector Storage**: Embeddings stored in Milvus with metadata references
6. **Retrieval**: Search queries use both PostgreSQL metadata and Milvus similarity search
7. **RAG Generation**: Retrieved context combined with LLM for intelligent responses

## Development Architecture

### Development Mode (`./setup.sh --dev`)
```
┌─────────────────┐    ┌─────────────────┐
│   PostgreSQL    │    │     Milvus      │
│   Container     │    │   Container     │
│  (localhost:    │    │ (localhost:     │
│     5432)       │    │    19530)       │
└─────────────────┘    └─────────────────┘
         │                       │
         └───────┬───────────────┘
                 │
         ┌─────────────────┐
         │  Flask App      │
         │  (Local Dev)    │
         │ (localhost:3000)│
         └─────────────────┘
```

### Production Mode (`./setup.sh --all`)
```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   PostgreSQL    │    │     Milvus      │    │     WebUI       │
│   Container     │    │   Container     │    │   Container     │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## Management Scripts

### Setup Script (`./setup.sh`)
- **`--all`**: Automated installation without prompts
- **`--dev`**: Development mode (infrastructure only, run app locally)  
- **`--help`**: Display installation options and usage

### Uninstall Script (`./uninstall.sh`)
- **`--dry-run`**: Preview removal without making changes
- **`--help`**: Display removal options
- **Project-safe**: Only removes RAG-specific containers and files

### Status Script (`./status.sh`)
- Environment validation
- Container health checks
- Database connectivity testing
- Directory structure verification

## Security Considerations

- **Environment Variables**: Sensitive data stored in `.env` file
- **Database Isolation**: Project-specific database names and users
- **Container Security**: Non-root user execution where possible
- **Network Isolation**: Services communicate through Docker networks

## Scalability Notes

- **Horizontal Scaling**: Milvus supports clustering for large datasets
- **Database Optimization**: PostgreSQL JSONB indexes for metadata queries
- **Embedding Models**: Ollama provides local inference without API dependencies
- **Caching**: Future implementation of Redis for session and query caching

## Future Enhancements

- [ ] Split `app.py` into smaller modules for maintainability
- [ ] Add dedicated configuration module (`config.py`)
- [ ] Implement Redis caching layer
- [ ] Add Milvus clustering support
- [ ] Provide sequence diagrams describing data flow
- [ ] Document Milvus schema and indexing strategy
- [ ] Add monitoring and logging infrastructure
