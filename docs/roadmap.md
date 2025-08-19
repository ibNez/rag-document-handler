# Roadmap

This stand-alone application builds a RAG-focused data lake by ingesting content into Milvus for
vector search and a single PostgreSQL database (`knowledgebase.db`) for structured metadata such as URLs and emails.
Other applications will query the database directly; no REST APIs are planned.
This document tracks planned enhancements and missing features.

## Short Term

- [ ] Add command-line interface for document ingestion and search.
- [ ] Implement ETL framework for loading diverse data sources into the RAG data lake.
- [ ] Add email ingestion module for specified account credentials with metadata stored in
      `knowledgebase` and embeddings persisted in Milvus.
- [ ] Add image ingestion using TensorFlow object classification with embeddings stored in Milvus.
- [ ] Improve test coverage and add sample tests.

## Long Term

- [ ] Publish hosted documentation site (e.g., GitHub Pages).
- [ ] Docker Compose setup for production deployment.
- [ ] Authentication and authorization for multi-user access.
- [ ] Support additional vector databases.
- [ ] Advanced scheduling and monitoring for ETL jobs.

Contributions and suggestions are welcome; open an issue to propose additional features.
