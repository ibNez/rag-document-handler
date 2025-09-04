# Database Schema Documentation

This document provides comprehensive documentation of all database schemas used in the RAG Knowledgebase Manager system.

## Overview

The system uses a dual-database architecture with refactored data access:
- **PostgreSQL**: Relational metadata storage managed through `ingestion/core/postgres_manager.py`
- **Milvus**: Vector embeddings storage accessed via `rag_manager/managers/milvus_manager.py`

## Data Access Architecture

The refactored system provides clean separation of database concerns:
- **Core Layer**: `ingestion/core/database_manager.py` provides unified database abstraction
- **PostgreSQL Layer**: `ingestion/core/postgres_manager.py` handles connection pooling and operations
- **Vector Layer**: `rag_manager/managers/milvus_manager.py` manages embeddings and RAG search
- **Domain Layers**: Email, URL, and document managers use core abstractions

## Schema Design Principles

**Consistent ID Naming Convention:**
- Every table has `id` field as UUID primary key
- Foreign keys follow `{tablename}_id` pattern
- Same UUID values used across PostgreSQL ↔ Milvus for one-to-one mapping

**Document Types:**
- Files and URLs are both stored as "documents" with `document_type` field
- Emails are separate entities with their own tables and collections

## PostgreSQL Schema

### Tables Overview

| Table | Purpose | Primary Key | Foreign Keys |
|-------|---------|-------------|--------------|
| `documents` | File upload and URL metadata | `id` (UUID) | None |
| `document_chunks` | Document text chunks for search | `id` (UUID) | `document_id` → `documents.id` |
| `urls` | URL-specific metadata and crawling settings | `id` (UUID) | `parent_url_id` → `urls.id` |
| `emails` | Email message metadata | `id` (UUID) | None |
| `email_chunks` | Email text chunks for search | `id` (UUID) | `email_id` → `emails.id` |

### Documents Table

Stores metadata for uploaded documents AND crawled URLs (unified approach).

```sql
CREATE TABLE documents (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    document_type VARCHAR(50) NOT NULL DEFAULT 'file', -- 'file' or 'url'
    title TEXT,
    content_preview TEXT,
    file_path TEXT, -- filename for files, URL for URLs
    content_type VARCHAR(100),
    file_size BIGINT,
    word_count INTEGER,
    page_count INTEGER,
    chunk_count INTEGER,
    avg_chunk_chars REAL,
    median_chunk_chars REAL,
    top_keywords TEXT[],
    processing_time_seconds REAL,
    processing_status VARCHAR(50) DEFAULT 'pending',
    file_hash VARCHAR(64),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    indexed_at TIMESTAMP WITH TIME ZONE
);
```

#### Field Descriptions

| Field | Type | Description | Constraints |
|-------|------|-------------|-------------|
| `id` | UUID | Unique identifier for the document | PRIMARY KEY, DEFAULT uuid_generate_v4() |
| `document_type` | VARCHAR(50) | Type of document: 'file' or 'url' | NOT NULL, DEFAULT 'file' |
| `title` | TEXT | Document title or page title | Optional |
| `content_preview` | TEXT | Preview of document content | Optional |
| `file_path` | TEXT | Filename for files, URL for web pages | Optional |

> **Architecture Note:** PostgreSQL is the single source of truth for all document metadata. Milvus stores only minimal fields needed for vector operations (document_id, page, content_hash). All metadata queries should use PostgreSQL with UUID relationships.

| `content_type` | VARCHAR(100) | MIME type (e.g., 'application/pdf') | Optional |
| `file_size` | BIGINT | File size in bytes | Optional |
| `word_count` | INTEGER | Total word count | Optional |
| `page_count` | INTEGER | Number of pages | Optional |
| `chunk_count` | INTEGER | Number of text chunks | Optional |
| `processing_status` | VARCHAR(50) | Processing state: 'pending', 'completed', 'failed' | DEFAULT 'pending' |
| `created_at` | TIMESTAMP | Record creation time | DEFAULT NOW() |
| `updated_at` | TIMESTAMP | Last record update | DEFAULT NOW() |

### Document Chunks Table

Stores text chunks from documents for hybrid retrieval and search.

```sql
CREATE TABLE document_chunks (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    document_id UUID NOT NULL,
    chunk_text TEXT NOT NULL,
    chunk_ordinal INTEGER NOT NULL,
    page_start INTEGER NULL,
    page_end INTEGER NULL,
    section_path TEXT NULL,
    element_types TEXT[] NULL,
    token_count INTEGER,
    chunk_hash VARCHAR(64),
    embedding_version VARCHAR(50) DEFAULT 'mxbai-embed-large',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    CONSTRAINT fk_document_chunks_document FOREIGN KEY (document_id) REFERENCES documents(id) ON DELETE CASCADE,
    CONSTRAINT uk_document_chunks_position UNIQUE(document_id, chunk_ordinal)
);
```

#### Field Descriptions

| Field | Type | Description | Constraints |
|-------|------|-------------|-------------|
| `id` | UUID | Unique identifier for the chunk | PRIMARY KEY, DEFAULT uuid_generate_v4() |
| `document_id` | UUID | Reference to parent document | FK → documents.id |
| `chunk_text` | TEXT | The actual text content of the chunk | NOT NULL |
| `chunk_ordinal` | INTEGER | Sequential chunk number within document | NOT NULL |
| `page_start` | INTEGER | Starting page number | Optional |
| `page_end` | INTEGER | Ending page number | Optional |
| `section_path` | TEXT | Hierarchical section path | Optional |
| `element_types` | TEXT[] | Types of elements from unstructured library | Optional |
| `token_count` | INTEGER | Number of tokens in chunk | Optional |
| `chunk_hash` | VARCHAR(64) | Content hash for deduplication | Optional |

### URLs Table

Stores URL-specific metadata and crawling configuration separate from the unified documents table.

```sql
CREATE TABLE urls (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    url TEXT NOT NULL UNIQUE,
    title TEXT,
    description TEXT,
    status VARCHAR(50) DEFAULT 'active',
    refresh_interval_minutes INTEGER,
    last_crawled TIMESTAMP WITH TIME ZONE,
    last_update_status VARCHAR(50),
    last_refresh_started TIMESTAMP WITH TIME ZONE,
    is_refreshing BOOLEAN DEFAULT FALSE,
    crawl_domain BOOLEAN DEFAULT FALSE,
    ignore_robots BOOLEAN DEFAULT FALSE,
    snapshot_retention_days INTEGER DEFAULT 30,
    snapshot_max_snapshots INTEGER DEFAULT 10,
    parent_url_id UUID REFERENCES urls(id) ON DELETE CASCADE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);
```

#### Field Descriptions

| Field | Type | Description | Constraints |
|-------|------|-------------|-------------|
| `id` | UUID | Unique identifier for the URL | PRIMARY KEY, DEFAULT gen_random_uuid() |
| `url` | TEXT | The actual URL to crawl | NOT NULL, UNIQUE |
| `title` | TEXT | Page title extracted from crawl | Optional |
| `description` | TEXT | Description or metadata about URL | Optional |
| `status` | VARCHAR(50) | URL status: 'active', 'inactive' | DEFAULT 'active' |
| `refresh_interval_minutes` | INTEGER | Minutes between crawls | Optional (null = no scheduling) |
| `last_crawled` | TIMESTAMP | When URL was last successfully crawled | Optional |
| `last_update_status` | VARCHAR(50) | Result of last crawl attempt | Optional |
| `last_refresh_started` | TIMESTAMP | When current/last refresh started | Optional |
| `is_refreshing` | BOOLEAN | Whether URL is currently being processed | DEFAULT FALSE |
| `crawl_domain` | BOOLEAN | Whether to discover child URLs | DEFAULT FALSE |
| `ignore_robots` | BOOLEAN | Whether to ignore robots.txt rules | DEFAULT FALSE |
| `snapshot_retention_days` | INTEGER | Days to keep snapshots (0 = forever) | DEFAULT 30 |
| `snapshot_max_snapshots` | INTEGER | Max snapshots to keep (0 = unlimited) | DEFAULT 10 |
| `parent_url_id` | UUID | Reference to parent URL for discovered children | REFERENCES urls(id) |
| `created_at` | TIMESTAMP | Record creation time | DEFAULT NOW() |
| `updated_at` | TIMESTAMP | Last record update | DEFAULT NOW() |

#### Key Features

- **Snapshot Configuration**: Snapshots are **always enabled** for all URLs (removed snapshot_enabled toggle for consistency)
- **Parent-Child Relationships**: URLs discovered via domain crawling reference their parent via `parent_url_id`
- **Processing Protection**: `is_refreshing` flag prevents overlapping crawl sessions
- **Scheduling Prevention**: Parent URLs cannot be re-scheduled while children are still processing
- **Robots.txt Support**: Optional compliance with crawl delays and access rules

**Note:** URLs are now stored in the unified `documents` table with `document_type = 'url'`. This provides consistent ID management and metadata handling across all document types.

### URL Snapshots Table

Stores point-in-time captures for crawled URLs and links to stored artifacts.

```sql
CREATE TABLE url_snapshots (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    document_id UUID NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
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
| `document_id` | UUID | Associated document (URL type) | NOT NULL, FK → documents.id |
| `snapshot_ts` | TIMESTAMP | Capture timestamp (UTC) | DEFAULT CURRENT_TIMESTAMP |
| `pdf_document_id` | VARCHAR | Reference to PDF in documents | Optional |
| `mhtml_document_id` | VARCHAR | Reference to MHTML in documents | Optional |
| `sha256` | TEXT | Hash of canonical page text used for embeddings | Optional |
| `notes` | TEXT | Additional capture details | Optional |

Notes:
- Artifact binaries (PDF/MHTML) are stored on disk; `documents` table tracks metadata and paths.
- Milvus `document_id` for URL embeddings should be set to `url_snapshots.id` to ensure point-in-time traceability.
- Snapshot capture is controlled per-URL via document metadata; retention can be enforced via configuration.

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

## Milvus Schema

### Collections Overview

| Collection | Purpose | Primary Identifier | Content Types |
|------------|---------|-------------------|---------------|
| `documents` | Document and URL embeddings | `document_id` (UUID) | Files and URLs |
| `emails` | Email embeddings | `email_id` (UUID) | Email messages |

### Documents Collection

The Milvus collection stores vector embeddings with **minimal metadata**. PostgreSQL is the single source of truth for all document metadata.

```python
# Collection: documents
# Dimension: 384 (mxbai-embed-large model)
fields = [
    FieldSchema(name="document_id", dtype=DataType.VARCHAR, max_length=65535),
    FieldSchema(name="page", dtype=DataType.INT64),
    FieldSchema(name="content_hash", dtype=DataType.VARCHAR, max_length=65535),
    FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535),
    FieldSchema(name="pk", dtype=DataType.INT64, is_primary=True, auto_id=True),
    FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=384)
]
```

#### Field Descriptions (Clean Architecture)

| Field | Type | Description | Content Usage |
|-------|------|-------------|---------------|
| `document_id` | VARCHAR(65535) | UUID from documents.id | Reference to PostgreSQL document |
| `page` | INT64 | Page or chunk number | Basic navigation reference |
| `content_hash` | VARCHAR(65535) | Hash for deduplication | Generated based on content |
| `text` | VARCHAR(65535) | Searchable text content | Extracted text chunk |
| `pk` | INT64 | Auto-generated primary key | Milvus internal ID |
| `vector` | FLOAT_VECTOR(384) | Embedding vector | Generated by mxbai-embed-large |

**Note**: All other metadata (filename, source, title, document_type, etc.) is stored in PostgreSQL and retrieved via `document_id` UUID relationship. This eliminates redundancy and ensures single source of truth.

### Emails Collection

The Milvus collection stores vector embeddings for email content with minimal metadata.

```python
# Collection: emails  
# Dimension: 384 (mxbai-embed-large model)
fields = [
    FieldSchema(name="email_id", dtype=DataType.VARCHAR, max_length=65535),
    FieldSchema(name="page", dtype=DataType.INT64),
    FieldSchema(name="content_hash", dtype=DataType.VARCHAR, max_length=65535),
    FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535),
    FieldSchema(name="pk", dtype=DataType.INT64, is_primary=True, auto_id=True),
    FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=384)
]
    FieldSchema(name="content_hash", dtype=DataType.VARCHAR, max_length=65535),
    FieldSchema(name="content_length", dtype=DataType.INT64),
    FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535),
    FieldSchema(name="pk", dtype=DataType.INT64, is_primary=False, auto_id=True),
    FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=384)
]
```

## Schema Mapping Rules

### File Upload → PostgreSQL + Milvus
```python
# PostgreSQL documents table
{
    "id": document_uuid,  # Auto-generated UUID
    "document_type": "file",
    "title": extracted_title,
    "file_path": filename,
    "content_type": "application/pdf"
}

# PostgreSQL document_chunks table  
{
    "id": chunk_id,  # Auto-generated UUID
    "document_id": document_id,  # FK to documents.id
    "chunk_text": text_content,
    "chunk_ordinal": chunk_index
}

# Milvus documents collection (clean minimal schema)
{
    "document_id": document_id,  # ID reference to PostgreSQL
    "page": page_number,
    "content_hash": "abc123...",
    "text": text_chunk
    # Note: All other metadata retrieved from PostgreSQL via document_id
}
```

### URL Crawl → PostgreSQL + Milvus
```python
# PostgreSQL documents table
{
    "id": document_uuid,  # Auto-generated UUID  
    "document_type": "url",
    "title": page_title,
    "file_path": url,
    "content_type": "text/html"
}

# PostgreSQL document_chunks table
{
    "id": chunk_id,  # Auto-generated UUID
    "document_id": document_id,  # FK to documents.id
    "chunk_text": text_content,
    "chunk_ordinal": chunk_index
}

# Milvus documents collection (clean minimal schema)
{
    "document_id": document_id,  # ID reference to PostgreSQL
    "page": chunk_index,
    "content_hash": "def456...",
    "text": text_chunk
    # Note: URL, page_title, document_type retrieved from PostgreSQL
}
```

### Email Processing → PostgreSQL + Milvus
```python
# PostgreSQL emails table
{
    "id": email_uuid,  # Auto-generated UUID
    "subject": email_subject,
    "from_addr": sender_address,
    "message_id": original_message_id
}

# PostgreSQL email_chunks table
{
    "id": chunk_id,  # Auto-generated UUID
    "email_id": email_id,  # FK to emails.id
    "chunk_text": text_content,
    "chunk_ordinal": chunk_index
}

# Milvus emails collection (clean minimal schema)
{
    "email_id": email_id,  # ID reference to PostgreSQL
    "page": chunk_index,
    "content_hash": "ghi789...",
    "text": text_chunk
    # Note: subject, from_addr, etc. retrieved from PostgreSQL
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

## Unified Duplicate Handling (Current Implementation)

The current system uses a SIMPLE, deterministic duplicate prevention strategy at upload time:

- Files: Duplicate prevention = filename presence in staging directory (filesystem check) BEFORE saving
- URLs: Uniqueness enforced by `urls.url` UNIQUE constraint and (optionally) by checking existence in unified `documents` (document_type='url') when creating a processed snapshot
- Emails: Deduplication enforced by UNIQUE `message_id` and `header_hash`
- Chunks: Per-parent uniqueness enforced via composite unique constraints (`(document_id, chunk_ordinal)` / `(email_id, chunk_index)`)

Content hashing roles:

- `file_hash`: A UNIQUE constraint on `documents.file_hash` at upload time. If a user uploads the exact same file contents under a different filename, the INSERT will raise a unique‑violation and the upload is rejected. We currently do the hash computation after saving the file, then attempt the insert; no pre-hash short‑circuit yet.
- `chunk_hash`: Used to prevent duplicate chunk inserts but not surfaced to the user directly.

Current duplicate scenarios:
1. Same filename, same content: Blocked early by filesystem filename existence check (never reaches DB).
2. Same filename, different content (should be rare): Detected by filename collision branch; user prompted to rename/delete.
3. Different filename, same content: Blocked by UNIQUE `file_hash` (constraint error intercepted and reported as duplicate – improvement pending for a more user-friendly message by pre-checking hash before save).
4. Different filename, different content: New row created (normal case).

Planned improvement (not yet implemented): Compute hash in-memory before writing to disk to provide a friendlier duplicate message and avoid temporary file creation.

### Summary Table of Uniqueness / Identity
| Domain | Primary Identifier | Upload-Time Duplicate Check | DB-Level Uniqueness | Notes |
|--------|--------------------|-----------------------------|---------------------|-------|
| Files | `filename` (staging) + row `id` (UUID) | Filesystem: `os.path.exists(staging/filename)` | `documents.filename` UNIQUE | `file_hash` reserved for integrity/future diff |
| URLs | `url` | Form submission prevents duplicates | `urls.url` UNIQUE; also stored in `documents` (`document_type='url'`) | Two-layer representation (structured + unified doc row) |
| Emails | `message_id` | Connector ensures no re-fetch by offset + message id | `emails.message_id`, `emails.header_hash` UNIQUE | `content_hash` optional post-ingest |
| File Chunks | `(document_id, chunk_ordinal)` | N/A (derived) | `uk_document_chunks_position` | Prevents double-chunking |
| Email Chunks | `(email_id, chunk_index)` | N/A (derived) | `uk_email_chunks_position` | Stable ordering |

### Processing Status Semantics
| Status | Applies To | Meaning |
|--------|-----------|---------|
| `pending` | Files (initial upload), URLs (queued), Emails (not chunked yet) | Record exists but not processed |
| `processing` | Files | Background worker loading/chunking |
| `embedding` | Files | Generating embeddings / enrichment |
| `storing` | Files | Persisting chunks/metadata |
| `completed` | Files, URLs (after snapshot/process) | Fully processed and queryable |
| `failed` | Files, URLs, Emails | Terminal error state |

### Lifecycle Overview
1. User uploads file → filesystem staging + `documents` row (pending)
2. User clicks Process → background worker finds existing row (NO new insert)
3. Worker updates row with chunk/page/word stats + sets `completed`
4. Chunks stored in PostgreSQL + vectors in Milvus
5. URLs follow a parallel path but originate from URL management panel instead of upload
6. Emails ingest via orchestrator: account → fetch → store raw → chunk → embed

## Future Extension: Chat Retention (Planned)
A forthcoming `chats` (or `conversations`) domain will follow the SAME patterns:
- Primary table with UUID `id`
- Chunk table (if needed) with `(chat_id, message_index)` uniqueness
- Consistent status fields (likely always `completed` after write)
- Potential retention policy table to mirror snapshot retention

> When adding chat retention, mirror this document: extend the Summary Uniqueness table and Lifecycle section.

## Implementation Reference Crosswalk
| Concern | Canonical Implementation File |
|---------|-------------------------------|
| Schema creation | `ingestion/core/postgres_manager.py` |
| File upload route | `rag_manager/web/routes.py` |
| Background file processing | `rag_manager/app.py` (`_process_document_background`) |
| URL snapshot + doc linkage | `ingestion/url/utils/snapshot_service.py` |
| URL metadata manager | `ingestion/url/manager.py` |
| Email ingestion orchestrator | `ingestion/email/orchestrator.py` |
| Email chunk storage | `ingestion/email/processor.py` |
| Hybrid retrieval | `rag_manager/managers/milvus_manager.py` |
| Stats aggregation | `rag_manager/web/panels/*.py` |

## Consistency Guardrails
To prevent regressions when modifying one domain:
- NEVER introduce a new identifier pattern (always UUID primary keys for core tables)
- Reuse status vocabulary (see table above)
- Keep uniqueness enforcement declarative (DB constraints) + minimal pre-checks (filesystem for files)
- Add any new domain to: schema doc tables, stats panels, and hybrid retrieval only if logically required
- Prefer UPDATE over INSERT during processing phases (no duplicate logical rows)

# End of augmented schema section.
