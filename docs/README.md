# RAG Knowledgebase Manager Documentation

Welcome to the documentation hub for the **RAG Knowledgebase Manager** project. This directory contains comprehensive guides that expand on the root `README.md`.

The application provides a complete RAG solution using a dual database architecture with refactored ingestion modules, panel-specific statistics providers, and a modular template system:
- **PostgreSQL**: Document metadata, URLs, emails, and analytics with JSONB flexibility
- **Milvus**: Vector embeddings and similarity search for semantic retrieval
- **Modular Ingestion**: Separate modules for email, URL, and document processing
- **Panel-Specific Statistics**: Dedicated statistics providers for each dashboard section
- **Template Partials**: Well-organized, maintainable template structure using modular components

## Quick Start

```bash
git clone <repository-url>
cd rag-document-handler
source .venv/bin/activate   # Always source the environment first
./start.sh                  # Use the proper startup script
```

## Architecture Overview

The refactored codebase follows Python best practices with clear separation of concerns:

```
ingestion/
├── core/         # Database abstraction and PostgreSQL management
├── email/        # Email processing with multiple connector support
├── url/          # URL crawling and content processing
├── document/     # Document extraction and chunking
└── utils/        # Shared utilities (crypto, scheduling)

rag_manager/
├── managers/     # Milvus vector operations and RAG search
└── web/          # Flask routes, web interface, and panel-specific statistics
    ├── panels/   # Individual statistics providers for each dashboard panel
    └── stats.py  # Statistics coordinator

templates/
├── partials/     # Modular template components for maintainable UI
└── *.html        # Main templates using partials system
```

## Table of Contents

- [Installation](installation.md) - Setup options, development mode, and deployment
- [Usage Guide](usage.md) - Web interface walkthrough and features
- [System Architecture](architecture.md) - Database design and component overview
- [Template Architecture](template-architecture.md) - Partials system and panel-specific statistics
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
| `./start.sh` | Start application with service checks and proper environment |
| `./setup.sh --all` | Automated production setup |
| `./setup.sh --dev` | Development environment setup |
| `./status.sh` | System health and connectivity check |
| `./uninstall.sh --dry-run` | Preview removal (safe testing) |

**Important**: Always `source .venv/bin/activate` before running any Python commands as specified in DEVELOPMENT_RULES.md.

Each document reflects the current project state and highlights areas for future development.
