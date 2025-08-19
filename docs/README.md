# RAG Document Handler Documentation

Welcome to the documentation hub for the **RAG Document Handler** project. This directory contains comprehensive guides that expand on the root `README.md`.

The application provides a complete RAG solution using a dual database architecture:
- **PostgreSQL**: Document metadata, URLs, emails, and analytics with JSONB flexibility
- **Milvus**: Vector embeddings and similarity search for semantic retrieval

## Quick Start

```bash
git clone <repository-url>
cd rag-document-handler
./setup.sh --dev        # Development setup
source .venv/bin/activate
python app.py           # Start local development server
```

## Table of Contents

- [Installation](installation.md) - Setup options, development mode, and deployment
- [Usage Guide](usage.md) - Web interface walkthrough and features
- [System Architecture](architecture.md) - Database design and component overview
- [Database Schema](database-schema.md) - Complete database field documentation
- [Email Processing](email-processing.md) - Email ingestion pipeline and troubleshooting
- [Contributing](contributing.md) - Development workflow and standards
- [Gmail Email Ingestion](gmail_ingestion.md) - Gmail integration setup
- [Exchange Email Ingestion](exchange_ingestion.md) - Exchange server integration
- [Roadmap](roadmap.md) - Future features and enhancements
- [Realm Tiles Prompt](realm_tiles_prompt.md) - AI prompt engineering guide

## Management Commands

| Command | Purpose |
|---------|---------|
| `./setup.sh --all` | Automated production setup |
| `./setup.sh --dev` | Development environment setup |
| `./uninstall.sh --dry-run` | Preview removal (safe testing) |
| `./status.sh` | System health and connectivity check |

Each document reflects the current project state and highlights areas for future development.
