# Usage Guide

The application exposes a Flask web interface for managing documents and performing RAG-powered search. Document embeddings are saved in Milvus while URLs and other metadata reside in a local SQLite database (`knowledgebase.db`).

## Starting the Application

1. Ensure Milvus is running:
   ```bash
   docker ps | grep milvus
   ```
2. Activate the virtual environment:
   ```bash
   source .venv/bin/activate
   ```
3. Launch the app:
   ```bash
   python app.py
   ```
4. Open `http://localhost:5000` in your browser.

## Features

- **Document Upload** – upload PDF, DOCX, DOC, TXT, or MD files.
- **URL Management** – store URLs with automatic title extraction in `knowledgebase.db`.
- **Metadata Storage** – persist URL and email details in `knowledgebase.db`.
- **Email Account Management** – manage multiple IMAP accounts and ingest mailboxes with duplicate detection.
- **Semantic Search** – query stored documents via vector search.
- **RAG Chat** – ask questions and receive answers synthesized from relevant documents.

## TODOs

- [ ] Command-line interface for batch operations.
- [x] Email ingestion from configured accounts with metadata stored in `knowledgebase.db` and embeddings persisted in Milvus.
- [ ] Image ingestion using TensorFlow object classification with embeddings stored in Milvus.
- [ ] Export search results to external formats.
