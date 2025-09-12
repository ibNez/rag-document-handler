# Database Schema Documentation

This document provides comprehensive documentation of all database schemas used in the RAG Knowledgebase Manager system.

## Overview

The system uses a dual-database architecture with clean data access:
- **PostgreSQL**: Relational metadata storage managed through `rag_manager/managers/postgres_manager.py`
- **Milvus**: Vector embeddings storage accessed via `rag_manager/managers/milvus_manager.py`

## Data Access Architecture

The system provides clean separation of database concerns:
- **Core Layer**: `rag_manager/core/` provides configuration and models
- **PostgreSQL Layer**: `rag_manager/managers/postgres_manager.py` handles connection management
- **Vector Layer**: `rag_manager/managers/milvus_manager.py` manages embeddings and search
- **Data Access**: Modular data managers use core abstractions for each domain

## Schema Design Principles

**Consistent ID Naming Convention:**
- Every table has `id` field as UUID primary key
- Foreign keys follow `{tablename}_id` pattern
- Same UUID values used across PostgreSQL ↔ Milvus for one-to-one mapping

**Document Types:**
- Files and URLs are both stored as "documents" with `document_type` field
- Emails are separate entities with their own tables and collections

**Milvus Collection Schema Design:**
- **Documents Collection**: Follows minimal metadata principle - only stores essential fields for vector operations (document_id, content_hash, page, snapshot_id)
- **Email Collection**: Stores more metadata for operational efficiency - includes email-specific fields for search and filtering
- Both collections store text content as vector embeddings rather than searchable text metadata
- PostgreSQL remains the single source of truth for all non-vector metadata
- Schemas are auto-created by LangChain based on actual metadata from processing code

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
    file_path TEXT, -- Full path to file for files, URL for URLs
    filename TEXT UNIQUE, -- Just the filename (e.g., 'document.pdf') for files and snapshot pdf filename for URLs
    content_type VARCHAR(100),
    file_size BIGINT,
    word_count INTEGER,
    page_count INTEGER,
    chunk_count INTEGER,
    avg_chunk_chars REAL,
    median_chunk_chars REAL,
    keywords TEXT, -- Comma-separated keywords for efficient search: "keyword1, keyword2, keyword3"
    processing_time_seconds REAL,
    processing_status VARCHAR(50) DEFAULT 'pending',
    file_hash VARCHAR(64) UNIQUE,
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
| `file_path` | TEXT | Full path to file for files, URL for URLs | Optional |
| `filename` | TEXT | Just the filename for files, snapshot pdf filename for URLs | UNIQUE |
| `content_type` | VARCHAR(100) | MIME type (e.g., 'application/pdf') | Optional |
| `file_size` | BIGINT | File size in bytes | Optional |
| `word_count` | INTEGER | Total word count | Optional |
| `page_count` | INTEGER | Number of pages | Optional |
| `chunk_count` | INTEGER | Number of text chunks | Optional |
| `avg_chunk_chars` | REAL | Average characters per chunk | Optional |
| `median_chunk_chars` | REAL | Median characters per chunk | Optional |
| `keywords` | TEXT | Comma-separated top keywords for search | Optional |
| `processing_time_seconds` | REAL | Time taken to process document | Optional |
| `processing_status` | VARCHAR(50) | Processing state: 'pending', 'completed', 'failed' | DEFAULT 'pending' |
| `file_hash` | VARCHAR(64) | Hash of file content for deduplication | UNIQUE |
| `created_at` | TIMESTAMP | Record creation time | DEFAULT NOW() |
| `updated_at` | TIMESTAMP | Last record update | DEFAULT NOW() |
| `indexed_at` | TIMESTAMP | When document was indexed | Optional |

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

Stores text chunks from documents for retrieval and search.

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
    topics TEXT NULL,
    embedding_version VARCHAR(50) DEFAULT 'mxbai-embed-large',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    CONSTRAINT fk_document_chunks_document FOREIGN KEY (document_id) REFERENCES documents(id) ON DELETE CASCADE,
    CONSTRAINT uk_document_chunks_position UNIQUE(document_id, chunk_ordinal)
);

-- Full-text search index for multi-topic queries
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_document_chunks_topics_gin 
ON document_chunks USING GIN (topics gin_trgm_ops);
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
| `topics` | TEXT | LLM-generated comma-separated multi-topics for search discoverability | Optional |

### URLs Table

Stores URL-specific metadata and crawling configuration separate from the unified documents table.

```sql
CREATE TABLE urls (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    url TEXT UNIQUE NOT NULL,
    title TEXT,
    description TEXT,
    status VARCHAR(50) DEFAULT 'pending',
    content_type VARCHAR(100),
    content_length BIGINT,
    last_crawled TIMESTAMP WITH TIME ZONE,
    crawl_depth INTEGER DEFAULT 0,
    refresh_interval_minutes INTEGER DEFAULT 1440,
    crawl_domain BOOLEAN DEFAULT FALSE,
    ignore_robots BOOLEAN DEFAULT FALSE,
    snapshot_retention_days INTEGER DEFAULT 0,
    snapshot_max_snapshots INTEGER DEFAULT 0,
    is_refreshing BOOLEAN DEFAULT FALSE,
    last_refresh_started TIMESTAMP WITH TIME ZONE,
    last_content_hash TEXT,
    last_update_status VARCHAR(50),
    parent_url_id UUID REFERENCES urls(id) ON DELETE CASCADE,
    child_url_count INTEGER DEFAULT 0,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);
```

#### Field Descriptions

| Field | Type | Description | Constraints |
|-------|------|-------------|-------------|
| `id` | UUID | Unique identifier for the URL | PRIMARY KEY, DEFAULT uuid_generate_v4() |
| `url` | TEXT | The actual URL to crawl | NOT NULL, UNIQUE |
| `title` | TEXT | Page title extracted from crawl | Optional |
| `description` | TEXT | Description or metadata about URL | Optional |
| `status` | VARCHAR(50) | URL status: 'pending', 'active', 'inactive' | DEFAULT 'pending' |
| `content_type` | VARCHAR(100) | MIME type of the URL content | Optional |
| `content_length` | BIGINT | Content length in bytes | Optional |
| `last_crawled` | TIMESTAMP | When URL was last successfully crawled | Optional |
| `crawl_depth` | INTEGER | Depth level for domain crawling | DEFAULT 0 |
| `refresh_interval_minutes` | INTEGER | Minutes between crawls | DEFAULT 1440 |
| `crawl_domain` | BOOLEAN | Whether to discover child URLs | DEFAULT FALSE |
| `ignore_robots` | BOOLEAN | Whether to ignore robots.txt rules | DEFAULT FALSE |
| `snapshot_retention_days` | INTEGER | Days to keep snapshots (0 = forever) | DEFAULT 0 |
| `snapshot_max_snapshots` | INTEGER | Max snapshots to keep (0 = unlimited) | DEFAULT 0 |
| `is_refreshing` | BOOLEAN | Whether URL is currently being processed | DEFAULT FALSE |
| `last_refresh_started` | TIMESTAMP | When current/last refresh started | Optional |
| `last_content_hash` | TEXT | Hash of last crawled content | Optional |
| `last_update_status` | VARCHAR(50) | Result of last crawl attempt | Optional |
| `parent_url_id` | UUID | Reference to parent URL for discovered children | REFERENCES urls(id) |
| `child_url_count` | INTEGER | Number of child URLs discovered | DEFAULT 0 |
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
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    document_id UUID NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    url_id UUID REFERENCES urls(id) ON DELETE CASCADE,
    snapshot_ts TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    pdf_document_id UUID, -- optional link to documents table for stored PDF artifact
    mhtml_document_id UUID, -- optional link to documents table for stored MHTML artifact
    sha256 TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);
```

#### Field Descriptions

| Field | Type | Description | Constraints |
|-------|------|-------------|-------------|
| `id` | UUID | Snapshot identifier | PRIMARY KEY, DEFAULT uuid_generate_v4() |
| `document_id` | UUID | Associated document (URL type) | NOT NULL, FK → documents.id |
| `url_id` | UUID | Reference to the source URL | FK → urls.id |
| `snapshot_ts` | TIMESTAMP | Capture timestamp (UTC) | NOT NULL, DEFAULT NOW() |
| `pdf_document_id` | UUID | Optional link to PDF document in documents table | Optional |
| `mhtml_document_id` | UUID | Optional link to MHTML document in documents table | Optional |
| `sha256` | TEXT | Hash of canonical page text used for embeddings | Optional |
| `created_at` | TIMESTAMP | Record creation time | DEFAULT NOW() |
| `updated_at` | TIMESTAMP | Last record update | DEFAULT NOW() |

Notes:
- Artifact binaries (PDF/MHTML) are stored on disk; `documents` table tracks metadata and paths.
- Milvus `document_id` for URL embeddings should be set to `url_snapshots.id` to ensure point-in-time traceability.
- Snapshot capture is controlled per-URL via document metadata; retention can be enforced via configuration.

### Email Accounts Table

Stores email account configuration and authentication.

```sql
CREATE TABLE email_accounts (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    account_name TEXT UNIQUE NOT NULL,
    server_type TEXT NOT NULL,
    server TEXT NOT NULL,
    port INTEGER NOT NULL,
    email_address TEXT NOT NULL,
    password TEXT NOT NULL,
    mailbox TEXT,
    batch_limit INTEGER,
    use_ssl INTEGER,
    refresh_interval_minutes INTEGER,
    last_synced TIMESTAMP,
    last_update_status TEXT,
    last_synced_offset INTEGER DEFAULT 0,
    total_emails_in_mailbox INTEGER DEFAULT 0
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
| `offset_position` | INTEGER | Current processing offset position for resuming interrupted operations | DEFAULT 0 |
| `last_synced` | TIMESTAMP | Last successful sync time | Optional |
| `last_update_status` | VARCHAR | Status of last sync attempt | Optional |
| `next_run` | TIMESTAMP | Scheduled next sync time | Optional |
| `created_at` | TIMESTAMP | Record creation time | DEFAULT CURRENT_TIMESTAMP |
| `updated_at` | TIMESTAMP | Last record update | DEFAULT CURRENT_TIMESTAMP |

### Emails Table

Stores individual email message metadata and content.

```sql
CREATE TABLE emails (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    message_id TEXT UNIQUE NOT NULL,
    from_addr TEXT,
    to_addrs JSONB,
    subject TEXT,
    date_utc TIMESTAMP,
    header_hash TEXT UNIQUE NOT NULL,
    content_hash TEXT UNIQUE,
    content TEXT,
    attachments TEXT,
    headers JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

#### Field Descriptions

| Field | Type | Description | Constraints |
|-------|------|-------------|-------------|
| `id` | UUID | Unique email identifier | PRIMARY KEY, DEFAULT uuid_generate_v4() |
| `message_id` | TEXT | **REQUIRED**: Unique email Message-ID header (no fallback generation) | UNIQUE NOT NULL |
| `from_addr` | TEXT | Sender email address | Optional |
| `to_addrs` | JSONB | Recipients as JSON array | Optional |
| `subject` | TEXT | Email subject line | Optional |
| `date_utc` | TIMESTAMP | Email date in UTC | Optional |
| `header_hash` | TEXT | **NEW**: Hash of email headers for deduplication | UNIQUE NOT NULL |
| `content_hash` | TEXT | **NEW**: Hash of email content for deduplication | UNIQUE |
| `content` | TEXT | Full email body content | Optional |
| `attachments` | TEXT | Attachment information | Optional |
| `headers` | JSONB | **NEW**: Complete email headers collection for metadata analysis | Optional |
| `created_at` | TIMESTAMP | Record creation time | DEFAULT CURRENT_TIMESTAMP |
| `updated_at` | TIMESTAMP | Last record update | DEFAULT CURRENT_TIMESTAMP |

#### Headers Field Details

The `headers` JSONB field contains all email headers collected during processing:

- **Standard Headers**: From, To, Subject, Date, Message-ID, Content-Type
- **MIME Headers**: MIME-Version, Content-Transfer-Encoding  
- **Threading Headers**: In-Reply-To, References for email thread tracking
- **Routing Headers**: Received, Return-Path, Delivered-To
- **Authentication Headers**: DKIM-Signature, SPF, DMARC headers
- **Custom Headers**: X-* headers and application-specific metadata

Example headers structure:
```json
{
  "from": "sender@example.com",
  "to": "recipient@example.com", 
  "subject": "Email Subject",
  "date": "Wed, 01 Jan 2025 12:00:00 -0000",
  "message-id": "<unique-id@example.com>",
  "content-type": "text/plain; charset=utf-8",
  "x-priority": "1",
  "received": ["by mail.example.com..."],
  "dkim-signature": "v=1; a=rsa-sha256..."
}
```

### Email Chunks Table

Stores extracted text chunks from email messages for retrieval and full-text search. Email chunks are linked to the parent email via the message ID (string primary key in `emails`).

```sql
CREATE TABLE email_chunks (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    email_id TEXT NOT NULL,
    chunk_text TEXT NOT NULL,
    chunk_index INTEGER NOT NULL,
    token_count INTEGER,
    chunk_hash VARCHAR(64),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    CONSTRAINT fk_email_chunks_email FOREIGN KEY (email_id) REFERENCES emails(id) ON DELETE CASCADE,
    CONSTRAINT uk_email_chunks_position UNIQUE(email_id, chunk_index)
);
```

#### Field Descriptions

| Field | Type | Description | Constraints |
|-------|------|-------------|-------------|
| `id` | UUID | Unique identifier for the chunk | PRIMARY KEY, DEFAULT uuid_generate_v4() |
| `email_id` | UUID | Parent email identifier (matches `emails.id`) | NOT NULL, FK → emails(id) |
| `chunk_text` | TEXT | The actual text content of the email chunk | NOT NULL |
| `chunk_index` | INTEGER | Sequential chunk number within the email (0-based) | NOT NULL |
| `token_count` | INTEGER | Token count for the chunk (optional) | Optional |
| `chunk_hash` | VARCHAR(64) | Content hash used for deduplication | Optional |
| `created_at` | TIMESTAMP | When the chunk was created/stored | DEFAULT NOW() |

#### Indexes and Performance

- `idx_email_chunks_email_id` on `email_chunks(email_id)` — speeds lookups by parent email.
- `idx_email_chunks_hash` on `email_chunks(chunk_hash)` — speeds deduplication checks.
- `idx_email_chunks_position` on `(email_id, chunk_index)` — supports the unique position constraint and ordered retrieval.

#### Notes

- Email chunks use UUID foreign keys to reference parent emails via `emails.id` for consistency with document chunks architecture.
- The unique constraint on `(email_id, chunk_index)` prevents duplicate chunk inserts for the same message and position.
- For retrieval, full-text search indexes (GIN on to_tsvector(chunk_text)) are created to support fast FTS queries against email chunks.

## Milvus Schema

## Milvus Schema

### Collections Overview

| Collection | Purpose | Primary Identifier | Content Types |
|------------|---------|-------------------|---------------|
| `documents` | Document and URL embeddings | `document_id` (UUID) | Files and URLs |
| `emails` | Email embeddings | `message_id` (String) | Email messages |

### Documents Collection

The Milvus collection stores vector embeddings with **minimal metadata**. PostgreSQL is the single source of truth for all document metadata.

```python
# Collection: documents
# Dimension: 384 (mxbai-embed-large model)
# Auto-created by LangChain based on actual metadata structure from document processing
fields = [
    FieldSchema(name="document_id", dtype=DataType.VARCHAR, max_length=65535),
    FieldSchema(name="content_hash", dtype=DataType.VARCHAR, max_length=65535),
    FieldSchema(name="page", dtype=DataType.INT64),
    FieldSchema(name="snapshot_id", dtype=DataType.VARCHAR, max_length=65535),
    FieldSchema(name="pk", dtype=DataType.INT64, is_primary=True, auto_id=True),
    FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=384)
]
```

#### Document Collection Field Descriptions

| Field | Type | Description | Usage |
|-------|------|-------------|-------|
| `document_id` | VARCHAR | UUID from documents.id | Reference to PostgreSQL document |
| `content_hash` | VARCHAR | SHA1 hash for deduplication | Generated based on content (16 chars) |
| `page` | INT64 | Page or chunk number | Basic navigation reference |
| `snapshot_id` | VARCHAR | Snapshot identifier for temporal deletion | Used for URLs (empty string for files) |
| `pk` | INT64 | Auto-generated primary key | Milvus internal ID |
| `vector` | FLOAT_VECTOR | Embedding vector (384 dimensions) | Generated by mxbai-embed-large |

**Architecture Note**: This collection follows the minimal metadata principle - only fields needed for Milvus operations are stored. All other metadata (filename, source, title, document_type, etc.) is stored in PostgreSQL and retrieved via `document_id` UUID relationship.

### Emails Collection

The Milvus collection stores vector embeddings for email content with email-specific metadata.

```python
# Collection: emails  
# Dimension: 384 (mxbai-embed-large model)
# Auto-created by LangChain based on actual metadata structure from email processor
fields = [
    FieldSchema(name="message_id", dtype=DataType.VARCHAR, max_length=65535),
    FieldSchema(name="source", dtype=DataType.VARCHAR, max_length=65535),
    FieldSchema(name="subject", dtype=DataType.VARCHAR, max_length=65535),
    FieldSchema(name="from_addr", dtype=DataType.VARCHAR, max_length=65535),
    FieldSchema(name="date_utc", dtype=DataType.VARCHAR, max_length=65535),
    FieldSchema(name="chunk_id", dtype=DataType.VARCHAR, max_length=65535),
    FieldSchema(name="page", dtype=DataType.INT64),
    FieldSchema(name="content_hash", dtype=DataType.VARCHAR, max_length=65535),
    FieldSchema(name="category_type", dtype=DataType.VARCHAR, max_length=65535),
    FieldSchema(name="pk", dtype=DataType.INT64, is_primary=True, auto_id=True),
    FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=384)
]
```

#### Email Collection Field Descriptions

| Field | Type | Description | Source |
|-------|------|-------------|---------|
| `message_id` | VARCHAR | Primary email identifier from Message-ID header | Email processor |
| `source` | VARCHAR | Format: "email:{from_addr}" | Email processor |
| `subject` | VARCHAR | Email subject line | Email processor |
| `from_addr` | VARCHAR | Email sender address | Email processor |
| `date_utc` | VARCHAR | Email date in UTC format | Email processor |
| `chunk_id` | VARCHAR | Format: "{message_id}:{chunk_index}" | Email processor |
| `page` | INT64 | Chunk index (0-based) | Email processor |
| `content_hash` | VARCHAR | SHA256 hash of chunk content for deduplication | Email processor |
| `category_type` | VARCHAR | Always "email" for email chunks | Email processor |
| `pk` | INT64 | Auto-generated primary key | Milvus auto_id |
| `vector` | FLOAT_VECTOR | Text embedding vector (384 dimensions) | Embedding model |

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

# Milvus documents collection (actual implementation schema)
{
    "document_id": document_id,  # ID reference to PostgreSQL
    "content_hash": "abc123...",  # SHA1 hash for deduplication (16 chars)
    "page": page_number,          # Page or chunk number
    "snapshot_id": ""             # Empty string for file uploads
    # Note: Text content stored as vector embedding, not as metadata
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

# Milvus documents collection (actual implementation schema)
{
    "document_id": document_id,     # ID reference to PostgreSQL
    "content_hash": "def456...",    # SHA1 hash for deduplication (16 chars)
    "page": chunk_index,            # Chunk index number
    "snapshot_id": snapshot_uuid    # Snapshot ID for temporal deletion
    # Note: Text content stored as vector embedding, not as metadata
    # Note: URL, page_title, document_type retrieved from PostgreSQL
}
```

### Email Processing → PostgreSQL + Milvus
```python
# PostgreSQL emails table
{
    "id": email_uuid,  # Auto-generated UUID
    "message_id": original_message_id,  # From email headers
    "subject": email_subject,
    "from_addr": sender_address,
    "to_addrs": ["recipient1@example.com", "recipient2@example.com"],  # JSONB array
    "date_utc": email_timestamp,
    "header_hash": "abc123...",  # Hash of headers for deduplication
    "content_hash": "def456...",  # Hash of content for deduplication
    "content": full_email_content,
    "headers": {  # Complete headers collection (NEW)
        "from": "sender@example.com",
        "to": "recipient@example.com",
        "subject": "Email Subject",
        "date": "Wed, 01 Jan 2025 12:00:00 -0000",
        "message-id": "<unique-id@example.com>",
        "content-type": "text/plain; charset=utf-8",
        "x-priority": "1",
        "received": ["by mail.example.com..."]
    }
}

# PostgreSQL email_chunks table
{
    "id": chunk_uuid,  # Auto-generated UUID
    "email_id": email_uuid,  # FK to emails.id
    "chunk_text": text_content,
    "chunk_index": chunk_position,
    "token_count": 150,
    "chunk_hash": "ghi789..."
}

# Milvus emails collection (actual implementation schema)
{
    "message_id": "message_id_from_header",    # Primary email identifier
    "source": "email:sender@domain.com",      # Format: "email:{from_addr}"
    "subject": "Email Subject Line",          # Email subject
    "from_addr": "sender@domain.com",         # Sender email address
    "date_utc": "2023-01-01T00:00:00Z",      # Email date in UTC
    "chunk_id": "message_id:0",               # Format: "{message_id}:{index}"
    "page": 0,                                # Chunk index (0-based)
    "content_hash": "ghi789...",              # SHA256 hash for deduplication
    "category_type": "email"                  # Always "email" for email chunks
    # Note: Text content stored as vector embedding, not as metadata
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
- Add any new domain to: schema doc tables, stats panels, and retrieval only if logically required
- Prefer UPDATE over INSERT during processing phases (no duplicate logical rows)

# End of augmented schema section.
