# Database Schema Documentation

This document provides comprehensive documentation of all database schemas used in the RAG Knowledgebase Manager system.

## Overview

The system uses a dual-database architecture with refactored data access:
- **PostgreSQL**: Relationa| `category` | VARCHAR(65535) | Content element type from unstructured library | CompositeElement, TableChunk, etc. |
| `category_type` | VARCHAR(65535) | Content source type classification | "document" / "url" / "email" |
| `content_hash` | VARCHAR(65535) | Hash for deduplication | Generated based on content |metadata storage managed through `ingestion/core/postgres_manager.py`
- **Milvus**: Vector embeddings storage accessed via `rag_manager/managers/milvus_manager.py`

## Data Access Architecture

The refactored system provides clean separation of database concerns:
- **Core Layer**: `ingestion/core/database_manager.py` provides unified database abstraction
- **PostgreSQL Layer**: `ingestion/core/postgres_manager.py` handles connection pooling and operations
- **Vector Layer**: `rag_manager/managers/milvus_manager.py` manages embeddings and RAG search
- **Domain Layers**: Email, URL, and document managers use core abstractions

## PostgreSQL Schema

### Tables Overview

| Table | Purpose | Primary Key | Foreign Keys |
|-------|---------|-------------|--------------|
| `documents` | File upload metadata | `document_id` | None |
| `urls` | Web crawling metadata | `id` | None |
| `email_accounts` | Email account configuration | `id` | None |
| `email_messages` | Email message metadata | `message_id` | `account_id` → `email_accounts.id` |

### Documents Table

Stores metadata for uploaded documents.

```sql
CREATE TABLE documents (
    document_id VARCHAR PRIMARY KEY,
    filename VARCHAR NOT NULL,
    content_type VARCHAR,
    file_size INTEGER,
    upload_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    processing_status VARCHAR DEFAULT 'pending',
    metadata JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

#### Field Descriptions

| Field | Type | Description | Constraints |
|-------|------|-------------|-------------|
| `document_id` | VARCHAR | Unique identifier for the document | PRIMARY KEY, NOT NULL |
| `filename` | VARCHAR | Original filename from upload | NOT NULL |
| `content_type` | VARCHAR | MIME type (e.g., 'application/pdf') | Optional |
| `file_size` | INTEGER | File size in bytes | Optional |
| `upload_timestamp` | TIMESTAMP | When document was uploaded | DEFAULT CURRENT_TIMESTAMP |
| `processing_status` | VARCHAR | Processing state: 'pending', 'completed', 'failed' | DEFAULT 'pending' |
| `metadata` | JSONB | Additional attributes (page_count, author, etc.) | Optional |
| `created_at` | TIMESTAMP | Record creation time | DEFAULT CURRENT_TIMESTAMP |
| `updated_at` | TIMESTAMP | Last record update | DEFAULT CURRENT_TIMESTAMP |

### URLs Table

Stores metadata for web URLs and crawling configuration.

```sql
CREATE TABLE urls (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    url VARCHAR NOT NULL UNIQUE,
    title VARCHAR,
    description TEXT,
    status VARCHAR DEFAULT 'active',
    content_type VARCHAR(100),
    content_length BIGINT,
    last_crawled TIMESTAMP,
    crawl_depth INTEGER DEFAULT 0,
    refresh_interval_minutes INTEGER DEFAULT 1440,
    crawl_domain BOOLEAN DEFAULT FALSE,
    ignore_robots BOOLEAN DEFAULT FALSE,
    snapshot_enabled BOOLEAN DEFAULT FALSE,
    snapshot_retention_days INTEGER,
    snapshot_max_snapshots INTEGER,
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

#### Field Descriptions

| Field | Type | Description | Constraints |
|-------|------|-------------|-------------|
| `id` | UUID | Auto-generated unique identifier | PRIMARY KEY, DEFAULT gen_random_uuid() |
| `url` | VARCHAR | The web URL to crawl | NOT NULL, UNIQUE |
| `title` | VARCHAR | Extracted page title | Optional |
| `description` | TEXT | User-provided description | Optional |
| `status` | VARCHAR | Crawl status: 'active', 'inactive', 'failed' | DEFAULT 'active' |
| `content_type` | VARCHAR(100) | MIME type of crawled content | Optional |
| `content_length` | BIGINT | Size of crawled content in bytes | Optional |
| `last_crawled` | TIMESTAMP | Last successful crawl time | Optional |
| `crawl_depth` | INTEGER | Current crawl depth level | DEFAULT 0 |
| `refresh_interval_minutes` | INTEGER | Refresh frequency in minutes | DEFAULT 1440 (24 hours) |
| `crawl_domain` | BOOLEAN | Whether to crawl entire domain | DEFAULT FALSE |
| `ignore_robots` | BOOLEAN | Whether to ignore robots.txt | DEFAULT FALSE |
| `snapshot_enabled` | BOOLEAN | Enable point-in-time snapshots for this URL | DEFAULT FALSE |
| `snapshot_retention_days` | INTEGER | Retain snapshots for N days (NULL = unlimited) | Optional |
| `snapshot_max_snapshots` | INTEGER | Max number of snapshots to keep (NULL = unlimited) | Optional |
| `metadata` | JSONB | Additional extensible metadata | DEFAULT '{}' |
| `created_at` | TIMESTAMP | Record creation time | DEFAULT CURRENT_TIMESTAMP |
| `updated_at` | TIMESTAMP | Last record update | DEFAULT CURRENT_TIMESTAMP |

### URL Snapshots Table

Stores point-in-time captures for crawled URLs and links to stored artifacts.

```sql
CREATE TABLE url_snapshots (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    url_id UUID NOT NULL REFERENCES urls(id) ON DELETE CASCADE,
    snapshot_ts TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    pdf_document_id VARCHAR,
    mhtml_document_id VARCHAR,
    sha256 TEXT,
    notes TEXT
);
```

#### Field Descriptions

| Field | Type | Description | Constraints |
|-------|------|-------------|-------------|
| `id` | UUID | Snapshot identifier | PRIMARY KEY, DEFAULT gen_random_uuid() |
| `url_id` | UUID | Associated URL | NOT NULL, FK → urls.id |
| `snapshot_ts` | TIMESTAMP | Capture timestamp (UTC) | DEFAULT CURRENT_TIMESTAMP |
| `pdf_document_id` | VARCHAR | Reference to PDF in documents | Optional |
| `mhtml_document_id` | VARCHAR | Reference to MHTML in documents | Optional |
| `sha256` | TEXT | Hash of canonical page text used for embeddings | Optional |
| `notes` | TEXT | Additional capture details | Optional |

Notes:
- Artifact binaries (PDF/MHTML) are stored on disk; `documents` table tracks metadata and paths.
- Milvus `document_id` for URL embeddings should be set to `url_snapshots.id` to ensure point-in-time traceability.
- Snapshot capture is controlled per-URL via `urls.snapshot_enabled`; retention can be enforced via `snapshot_retention_days` and/or `snapshot_max_snapshots`.

### Email Accounts Table

Stores email account configuration and authentication.

```sql
CREATE TABLE email_accounts (
    id SERIAL PRIMARY KEY,
    account_name VARCHAR NOT NULL,
    server_type VARCHAR NOT NULL,
    server VARCHAR NOT NULL,
    port INTEGER NOT NULL,
    email_address VARCHAR NOT NULL,
    encrypted_password BYTEA NOT NULL,
    mailbox VARCHAR DEFAULT 'INBOX',
    batch_limit INTEGER DEFAULT 50,
    use_ssl BOOLEAN DEFAULT TRUE,
    refresh_interval_minutes INTEGER DEFAULT 60,
    offset_position INTEGER DEFAULT 0,
    last_synced TIMESTAMP,
    last_update_status VARCHAR,
    next_run TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

#### Field Descriptions

| Field | Type | Description | Constraints |
|-------|------|-------------|-------------|
| `id` | SERIAL | Auto-incrementing account ID | PRIMARY KEY |
| `account_name` | VARCHAR | User-friendly name for the account | NOT NULL |
| `server_type` | VARCHAR | Protocol: 'imap', 'pop3', 'exchange' | NOT NULL |
| `server` | VARCHAR | Mail server hostname | NOT NULL |
| `port` | INTEGER | Mail server port number | NOT NULL |
| `email_address` | VARCHAR | Account email address | NOT NULL |
| `encrypted_password` | BYTEA | Password encrypted with EMAIL_ENCRYPTION_KEY | NOT NULL |
| `mailbox` | VARCHAR | Target mailbox/folder | DEFAULT 'INBOX' |
| `batch_limit` | INTEGER | Maximum emails per batch | DEFAULT 50 |
| `use_ssl` | BOOLEAN | Enable SSL/TLS connection | DEFAULT TRUE |
| `refresh_interval_minutes` | INTEGER | Sync frequency in minutes | DEFAULT 60 |
| `offset_position` | INTEGER | **NEW**: Current processing offset position for resuming interrupted operations | DEFAULT 0 |
| `last_synced` | TIMESTAMP | Last successful sync time | Optional |
| `last_update_status` | VARCHAR | Status of last sync attempt | Optional |
| `next_run` | TIMESTAMP | Scheduled next sync time | Optional |
| `created_at` | TIMESTAMP | Record creation time | DEFAULT CURRENT_TIMESTAMP |
| `updated_at` | TIMESTAMP | Last record update | DEFAULT CURRENT_TIMESTAMP |

### Email Messages Table

Stores individual email message metadata and content.

```sql
CREATE TABLE email_messages (
    message_id VARCHAR PRIMARY KEY,
    account_id INTEGER NOT NULL REFERENCES email_accounts(id),
    subject VARCHAR,
    from_addr VARCHAR,
    to_addrs TEXT,
    date_utc TIMESTAMP,
    body_text TEXT,
    body_html TEXT,
    attachments_info JSONB,
    server_type VARCHAR,
    content_hash VARCHAR,
    validation_status VARCHAR DEFAULT 'valid',
    processed_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

#### Field Descriptions

| Field | Type | Description | Constraints |
|-------|------|-------------|-------------|
| `message_id` | VARCHAR | **REQUIRED**: Unique email message identifier (no fallback generation) | PRIMARY KEY |
| `account_id` | INTEGER | References email_accounts.id | NOT NULL, FOREIGN KEY |
| `subject` | VARCHAR | Email subject line | Optional |
| `from_addr` | VARCHAR | Sender email address | Optional |
| `to_addrs` | TEXT | Comma-separated recipient addresses | Optional |
| `date_utc` | TIMESTAMP | Email date in UTC | Optional |
| `body_text` | TEXT | Plain text body content | Optional |
| `body_html` | TEXT | HTML body content | Optional |
| `attachments_info` | JSONB | Attachment metadata and info | Optional |
| `server_type` | VARCHAR | Source server type | Optional |
| `content_hash` | VARCHAR | **NEW**: Hash for deduplication across accounts | Optional |
| `validation_status` | VARCHAR | **NEW**: Corruption detection results ('valid', 'corrupted', 'missing_headers') | DEFAULT 'valid' |
| `processed_timestamp` | TIMESTAMP | When email was processed | DEFAULT CURRENT_TIMESTAMP |
| `created_at` | TIMESTAMP | Record creation time | DEFAULT CURRENT_TIMESTAMP |
| `updated_at` | TIMESTAMP | Last record update | DEFAULT CURRENT_TIMESTAMP |

## Milvus Schema

### Documents Collection

The Milvus collection stores vector embeddings with metadata for all content types using a unified schema.

```python
# Collection: documents
# Dimension: 384 (mxbai-embed-large model)
fields = [
    FieldSchema(name="document_id", dtype=DataType.VARCHAR, max_length=65535, is_primary=True),
    FieldSchema(name="source", dtype=DataType.VARCHAR, max_length=65535),
    FieldSchema(name="page", dtype=DataType.INT64),
    FieldSchema(name="chunk_id", dtype=DataType.VARCHAR, max_length=65535),
    FieldSchema(name="topic", dtype=DataType.VARCHAR, max_length=65535),
    FieldSchema(name="category", dtype=DataType.VARCHAR, max_length=65535),
    FieldSchema(name="category_type", dtype=DataType.VARCHAR, max_length=65535),
    FieldSchema(name="content_hash", dtype=DataType.VARCHAR, max_length=65535),
    FieldSchema(name="content_length", dtype=DataType.INT64),
    FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535),
    FieldSchema(name="pk", dtype=DataType.INT64, is_primary=False, auto_id=True),
    FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=384)
]
```

#### Field Descriptions

| Field | Type | Description | Content Type Usage |
|-------|------|-------------|-------------------|
| `document_id` | VARCHAR(65535) | Primary identifier | File ID / URL ID / Email message_id |
| `source` | VARCHAR(65535) | Content source location | File path / URL / email:{from_addr} |
| `page` | INT64 | Page or chunk number | Document page / URL chunk index / Email chunk index |
| `chunk_id` | VARCHAR(65535) | Unique chunk identifier | Generated composite ID |
| `topic` | VARCHAR(65535) | Main subject or title | Filename / Page title / Email subject |
| `category` | VARCHAR(65535) | Content type classification | "document" / "url" / "email" |
| `content_hash` | VARCHAR(65535) | Hash for deduplication | Generated based on content |
| `content_length` | INT64 | Character count of text | Length of text field |
| `text` | VARCHAR(65535) | Searchable text content | Extracted text chunk |
| `pk` | INT64 | Auto-generated primary key | Milvus internal ID |
| `vector` | FLOAT_VECTOR(384) | Embedding vector | Generated by mxbai-embed-large |

### Index Configuration

```python
# Vector similarity index
index_params = {
    "index_type": "IVF_FLAT",
    "metric_type": "COSINE",
    "params": {"nlist": 128}
}
```

## Schema Mapping Rules

### Document Upload → Milvus
```python
{
    "document_id": document_id,
    "source": f"file:{filename}",
    "page": page_number,
    "chunk_id": f"{document_id}:{chunk_index}",
    "topic": filename,
    "category": element_type,  # From unstructured library
    "category_type": "document",
    "content_hash": f"doc_{document_id}_{chunk_index}",
    "content_length": len(text_chunk),
    "text": text_chunk
}
```

### URL Crawl → Milvus
When `urls.snapshot_enabled = true`, use the snapshot id to ensure point-in-time traceability; otherwise fall back to the URL id.

```python
# If snapshots enabled for this URL
{
    "document_id": snapshot_id,  # from url_snapshots.id
    "source": url,
    "page": chunk_index,
    "chunk_id": f"{snapshot_id}:{chunk_index}",
    "topic": page_title,
    "category": "NarrativeText",  # Or other element types
    "category_type": "url",
    "content_hash": f"url_{snapshot_id}_{chunk_index}",
    "content_length": len(text_chunk),
    "text": text_chunk
}

# If snapshots disabled for this URL
{
    "document_id": url_id,
    "source": url,
    "page": chunk_index,
    "chunk_id": f"{url_id}:{chunk_index}",
    "topic": page_title,
    "category": "NarrativeText",  # Or other element types
    "category_type": "url",
    "content_hash": f"url_{url_id}_{chunk_index}",
    "content_length": len(text_chunk),
    "text": text_chunk
}
```

### Email Processing → Milvus
```python
{
    "document_id": message_id,
    "source": f"email:{from_addr}",
    "page": chunk_index,
    "chunk_id": f"{message_id}:{chunk_index}",
    "topic": subject,
    "category": "NarrativeText",  # Or other element types
    "category_type": "email",
    "content_hash": f"email_{message_id}_{chunk_index}",
    "content_length": len(text_chunk),
    "text": text_chunk
}
```

## Migration and Maintenance

### Schema Migrations

1. **PostgreSQL Migrations**: Use standard SQL migration scripts
2. **Milvus Schema Changes**: Require collection recreation (no in-place migration)
3. **Data Migration**: Export from old schema, transform, import to new schema

### Backup Strategy

1. **PostgreSQL**: Standard pg_dump for metadata backup
2. **Milvus**: Collection export using bulkwriter for vector data
3. **Consistency**: Coordinate backups to maintain referential integrity

### Performance Considerations

1. **PostgreSQL Indexes**: Create indexes on frequently queried JSONB fields
2. **Milvus Optimization**: Tune index parameters based on data size and query patterns
3. **Connection Pooling**: Use pgbouncer for PostgreSQL connections
4. **Monitoring**: Track query performance and optimize accordingly

## Troubleshooting

### Common Schema Issues

1. **Milvus Schema Mismatch**: Error indicates field count mismatch - check mapping logic
2. **PostgreSQL Foreign Key Violations**: Ensure referenced records exist before insertion
3. **Encoding Issues**: Use UTF-8 encoding consistently across all text fields
4. **Vector Dimension Mismatch**: Verify embedding model outputs correct dimension (384)

### Debugging Queries

```sql
-- Check PostgreSQL table schemas
\d+ table_name

-- View Milvus collection info
-- Use pymilvus Collection.describe() method

-- Check data consistency
SELECT COUNT(*) FROM email_messages WHERE account_id NOT IN (SELECT id FROM email_accounts);
```
