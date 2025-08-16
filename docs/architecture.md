# System Architecture

The project currently consists of a single Flask application (`app.py`) that ties together document processing, Milvus vector storage, a SQLite metadata store (`knowledgebase.db`), and a web interface.

## Components

| Component | Description |
|-----------|-------------|
| `app.py` | Contains the `RAGDocumentHandler` class implementing upload, URL management, embedding generation, and search. |
| `templates/` | Jinja2 templates for the web UI. |
| `static/` | Static assets (CSS, JS, images) used by the templates. |
| `examples/` | Sample data and scripts demonstrating usage. |
| `setup.sh` / `start.sh` | Helper scripts for installing dependencies and running the project. |

## Data Flow

1. Users upload documents or submit URLs through the web interface. Email ingestion is planned.
2. Text is extracted and embedded using SentenceTransformers.
3. Metadata such as URLs and emails is stored in a local SQLite database (`knowledgebase.db`).
4. Embeddings are stored in Milvus for retrieval.
5. Search and RAG chat retrieve embeddings from Milvus and combine them with metadata from SQLite to generate answers using an LLM.

## TODOs

- [ ] Split `app.py` into smaller modules for maintainability.
- [ ] Add a dedicated configuration module (`config.py`).
- [ ] Provide sequence diagrams describing data flow.
- [ ] Document Milvus schema and indexing strategy.
- [ ] Document SQLite schema for URL and email metadata.
