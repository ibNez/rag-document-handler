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

## Email ingestion configuration

The application can optionally synchronise an IMAP inbox. Define the following environment variables in your `.env` file to enable and configure this behaviour:

| Variable | Default | Description |
| --- | --- | --- |
| `EMAIL_ENABLED` | `false` | Enable periodic email sync |
| `IMAP_HOST` | _(empty)_ | IMAP server hostname |
| `IMAP_PORT` | `993` | IMAP server port |
| `IMAP_USERNAME` | _(empty)_ | IMAP account username |
| `IMAP_PASSWORD` | _(empty)_ | IMAP account password |
| `IMAP_MAILBOX` | `INBOX` | Mailbox to read from |
| `IMAP_BATCH_LIMIT` | `50` | Maximum messages fetched per cycle |
| `IMAP_USE_SSL` | `true` | Use SSL/TLS for IMAP connection |
| `EMAIL_SYNC_INTERVAL_SECONDS` | `300` | Interval between sync cycles |

## TODOs

- [ ] Provide Docker Compose configuration for full-stack deployment.
- [ ] Publish package on PyPI for `pip install rag-document-handler`.
