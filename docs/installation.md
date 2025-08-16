# Installation

This guide covers setting up the RAG Document Handler in a development environment.

## Prerequisites

- Python 3.8 or higher
- Docker (for Milvus vector database)
- Git
- SQLite (bundled with Python) for metadata storage

## Quick Setup

```bash
git clone <repository-url>
cd rag-document-handler
./setup.sh
./start.sh
```

The setup script installs dependencies, prepares Milvus, and initializes the local SQLite database (`knowledgebase.db`). The start script launches Milvus (if needed) and runs the web application.

## Manual Installation

1. **Create virtual environment**
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```
2. **Install Python dependencies**
   ```bash
   pip install -e .
   ```
   > TODO: Publish an official package so users can run `pip install rag-document-handler`.
3. **Run Milvus**
   ```bash
   docker run -d --name milvus -p 19530:19530 milvusdb/milvus:latest
   ```
4. **Start the application**
   ```bash
   python app.py
   ```

## Next Steps

- Configure environment variables via `.env` (see root `README.md`).
- Visit [Usage Guide](usage.md) to interact with the application.
- The `knowledgebase.db` SQLite database is created on first run to store URL and future email metadata.

## TODOs

- [ ] Provide Docker Compose configuration for full-stack deployment.
- [ ] Publish package on PyPI for `pip install rag-document-handler`.
