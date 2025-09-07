# RAG Knowledgebase Manager Documentation

Welcome to the documentation hub for the **RAG Knowledgebase Manager** project. This directory contains comprehensive guides that expand on the root `README.md`.

The application provides a complete RAG solution using a dual database architecture with modular ingestion and retrieval systems:
- **PostgreSQL**: Document metadata, URLs, emails, and analytics with JSONB flexibility
- **Milvus**: Vector embeddings and similarity search for semantic retrieval
- **Modular Ingestion**: Separate modules for email, URL, and document processing
- **Modular Retrieval**: Dedicated retrieval modules for different content types
- **Template Partials**: Well-organized, maintainable template structure using modular components

## Quick Start

```bash
git clone <repository-url>
cd rag-document-handler
source .venv/bin/activate   # Always source the environment first
./start.sh                  # Use the proper startup script
```

## Architecture Overview

The codebase follows Python best practices with clear separation of concerns:

```
ingestion/
├── core/         # Database abstraction and PostgreSQL management
├── email/        # Email processing with multiple connector support
├── url/          # URL crawling and content processing
├── document/     # Document extraction and chunking
└── utils/        # Shared utilities (crypto, scheduling)

retrieval/
├── document/     # Document search and retrieval operations
├── email/        # Email-specific search and retrieval
└── url/          # URL-specific search and retrieval

rag_manager/
├── core/         # Core configuration and models
├── data/         # Data access layer (PostgreSQL, Milvus)
├── managers/     # Business logic managers
├── web/          # Flask routes and web interface
└── utils/        # RAG-specific utilities

templates/
├── partials/     # Modular template components for maintainable UI
└── *.html        # Main templates using partials system
```

## Recent Updates

### Email Headers Collection (September 2025)
- **Complete Headers Collection**: All email headers now automatically collected during sync
- **JSONB Storage**: Headers stored in PostgreSQL for efficient querying and analysis
- **Cross-Connector Support**: IMAP, Gmail, and Exchange connectors all collect headers
- **Metadata Analysis**: Authentication, routing, threading, and priority information available
- **Query Capabilities**: Advanced JSONB queries for header-based email filtering

### PostgreSQL Architecture Consolidation
- **Unified Schema Management**: Single source of truth in `postgres_manager.py`
- **UUID-Based Architecture**: Consistent UUID primary keys across all tables
- **Improved Data Integrity**: Proper foreign key relationships and constraints
- **Enhanced Performance**: Optimized connection pooling and query efficiency

## Table of Contents

- [Installation](installation.md) - Setup options, development mode, and deployment
- [Usage Guide](usage.md) - Web interface walkthrough, email processing, and URL snapshots
- [System Architecture](architecture.md) - Database design, component overview, email processing, and retrieval
- [Configuration](configuration.md) - Environment variables, email settings, and snapshot configuration
- [Database Schema](database-schema.md) - Complete database field documentation
- [Contributing](contributing.md) - Development workflow and standards
- [Document Ingestion POC](document-ingest-poc.md) - Document processing proof of concept
- [PostgreSQL Guide](postgresql_guide.md) - Database setup and management
- [Roadmap](roadmap.md) - Future features and enhancements

## Management Commands

| Command | Purpose |
|---------|---------|
| `./start.sh` | Start application with service checks and proper environment |
| `./setup.sh --all` | Automated production setup |
| `./setup.sh --dev` | Development environment setup |
| `./status.sh` | System health and connectivity check |
| `./uninstall.sh --dry-run` | Preview removal (safe testing) |



Each document reflects the current project state and highlights areas for future development.
