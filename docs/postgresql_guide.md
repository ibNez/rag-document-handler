# PostgreSQL Integration Guide

## Overview

The RAG Document Handler now uses PostgreSQL for metadata storage and Milvus for vector embeddings, providing a robust and scalable database architecture.

## Architecture

```
┌─────────────────┬─────────────────┐
│   PostgreSQL    │     Milvus      │
│   (Metadata)    │   (Vectors)     │
├─────────────────┼─────────────────┤
│ • Documents     │ • Embeddings    │
│ • URLs          │ • Similarity    │
│ • Emails        │   Search        │
│ • Accounts      │ • Collections   │
│ • Analytics     │                 │
└─────────────────┴─────────────────┘
```

## Quick Start

1. **Install Dependencies**
   ```bash
   ./setup.sh
   ```

2. **Start Database Services**
   ```bash
   docker compose up postgres milvus -d
   ```

3. **Test Connection**
   ```bash
   python test_postgres.py
   ```

4. **Start Application**
   ```bash
   python app.py
   ```

## Database Schema

### Documents Table
- `document_id` - Unique document identifier
- `title` - Document title
- `content_preview` - Text preview for search
- `metadata` - JSONB for flexible attributes
- `processing_status` - Current processing state
- `created_at` / `updated_at` - Timestamps

### Advanced Features
- **Full-text Search** - PostgreSQL's built-in text search
- **JSON Queries** - Flexible metadata searching
- **Analytics** - Built-in reporting capabilities
- **Indexing** - Optimized for performance

## Configuration

Environment variables in `.env`:
```bash
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_DB=rag_metadata
POSTGRES_USER=rag_user
POSTGRES_PASSWORD=secure_password
```

## Development Benefits

- **Type Safety** - Proper schema validation
- **Performance** - Optimized indexing and queries
- **Scalability** - Handle large document collections
- **Analytics** - Built-in reporting and statistics
- **Standards** - SQL compliance for complex queries
- **Backup** - Enterprise-grade backup solutions

## Troubleshooting

### Connection Issues
```bash
# Check if PostgreSQL is running
docker compose ps postgres

# View PostgreSQL logs
docker compose logs postgres

# Test connection
python test_postgres.py
```

### Performance Optimization
- Monitor query performance with `pg_stat_statements`
- Adjust connection pool size based on load
- Use JSONB indexes for metadata queries
- Consider partitioning for large datasets

### Backup and Recovery
```bash
# Backup database
docker exec postgres_container pg_dump -U rag_user rag_metadata > backup.sql

# Restore database
docker exec -i postgres_container psql -U rag_user rag_metadata < backup.sql
```
