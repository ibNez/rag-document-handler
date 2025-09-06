# Document Module Architecture

## Overview

The Document Module provides comprehensive document ingestion, storage, and retrieval capabilities for the RAG Knowledgebase Manager. The module uses a clean architecture that separates data operations from business logic.

## Architecture Components

### Data Access Layer

#### DocumentDataManager
Located: `rag_manager/data/document_data.py`

The DocumentDataManager provides pure data access operations for document content without any business logic. It handles all PostgreSQL and Milvus operations for document storage and retrieval.

**Key Responsibilities:**
- Document metadata storage and retrieval
- Document chunk storage and management
- Full-text search operations
- Document statistics and reporting
- Metadata enrichment for search results
- Vector operations coordination

**Core Methods:**
- `upsert_document_metadata(filename, metadata)` - Insert or update document metadata
- `get_document_metadata(filename)` - Retrieve document metadata by filename
- `delete_document_metadata(filename)` - Remove document metadata
- `store_document(file_path, filename, **kwargs)` - Store document with full metadata
- `store_document_chunk(document_id, chunk_text, **kwargs)` - Store individual chunks
- `persist_chunks(document_id, chunks)` - Batch store LangChain Document chunks
- `search_documents(query, limit)` - Full-text search across documents
- `search_document_chunks_fts(query, k)` - Full-text search across chunks
- `get_knowledgebase_metadata()` - Aggregated knowledge base statistics
- `batch_enrich_documents_from_postgres(docs)` - Enrich search results with metadata

### Business Logic Layer

#### DocumentSourceManager (Ingestion)
Located: `ingestion/document/source_manager.py`

Handles document ingestion orchestration and business logic. Delegates all data operations to DocumentDataManager while applying ingestion-specific rules and validations.

**Key Features:**
- File type filtering and validation
- Processing status management
- Ingestion business rules
- Error handling and recovery
- Metadata mapping and transformation

**Core Methods:**
- `upsert_document_metadata(filename, metadata)` - Ingestion with business rules
- `persist_chunks(document_id, chunks, trace_logger)` - Chunk ingestion orchestration
- `store_document(file_path, filename, **kwargs)` - Document storage with validation
- `update_processing_status(document_id, status)` - Status management

#### DocumentSearchManager (Retrieval)
Located: `retrieval/document/search_manager.py`

Handles document search orchestration and retrieval business logic. Provides enhanced search capabilities with result processing and metadata enrichment.

**Key Features:**
- Search query sanitization and validation
- Result post-processing and ranking
- Metadata enrichment for display
- Search performance optimization
- Result categorization and formatting

**Core Methods:**
- `search_documents(query, limit)` - Enhanced document search
- `search_document_chunks_fts(query, k)` - Enhanced chunk search
- `batch_enrich_documents_from_postgres(docs)` - Search result enrichment
- `normalize_documents_metadata(docs)` - Metadata normalization
- `get_document_statistics()` - Retrieval-optimized statistics

### Interface Layer

#### DocumentManager (Ingestion)
Located: `ingestion/document/manager.py`

Provides backward-compatible interface for ingestion operations. Delegates all operations to DocumentSourceManager.

#### DocumentManager (Retrieval)
Located: `retrieval/document/manager.py`

Provides backward-compatible interface for retrieval operations. Delegates all operations to DocumentSearchManager.

## Configuration

### Database Requirements

The Document Module requires PostgreSQL with the following tables:

**documents table:**
- `id` (UUID, Primary Key)
- `filename` (TEXT, Unique)
- `title` (TEXT)
- `content_preview` (TEXT)
- `file_path` (TEXT)
- `content_type` (TEXT)
- `file_size` (BIGINT)
- `word_count` (INTEGER)
- `page_count` (INTEGER)
- `chunk_count` (INTEGER)
- `avg_chunk_chars` (INTEGER)
- `median_chunk_chars` (INTEGER)
- `top_keywords` (TEXT[])
- `processing_time_seconds` (REAL)
- `processing_status` (TEXT)
- `document_type` (TEXT)
- `created_at` (TIMESTAMP)
- `updated_at` (TIMESTAMP)

**document_chunks table:**
- `id` (UUID, Primary Key)
- `document_id` (UUID, Foreign Key)
- `chunk_text` (TEXT)
- `chunk_ordinal` (INTEGER)
- `page_start` (INTEGER)
- `page_end` (INTEGER)
- `section_path` (TEXT)
- `element_types` (TEXT[])
- `token_count` (INTEGER)
- `chunk_hash` (TEXT)
- `topics` (TEXT)
- `embedding_version` (TEXT)
- `created_at` (TIMESTAMP)

### PostgreSQL Configuration

Full-text search requires PostgreSQL with English text search configuration:
```sql
-- Example FTS query used by the module
SELECT chunk_text, ts_rank_cd(to_tsvector('english', chunk_text), plainto_tsquery('english', %s)) as rank
FROM document_chunks 
WHERE to_tsvector('english', chunk_text) @@ plainto_tsquery('english', %s)
ORDER BY rank DESC;
```

### Vector Store Integration

Optional Milvus integration for vector operations:
- Vector storage for semantic similarity search
- Embedding management
- Vector deletion during document cleanup

## Usage Examples

### Document Ingestion

```python
from ingestion.document.manager import DocumentManager

# Initialize manager
doc_manager = DocumentManager(postgres_manager, vector_store)

# Store document metadata
doc_id = doc_manager.store_document(
    file_path="/path/to/document.pdf",
    filename="document.pdf",
    title="Important Document",
    content_type="application/pdf",
    file_size=1024000,
    word_count=5000
)

# Update processing status
doc_manager.update_processing_status(doc_id, "processing")

# Store document chunks
chunks = [...]  # LangChain Document objects
stored_count = doc_manager.persist_chunks(doc_id, chunks)

# Mark as completed
doc_manager.update_processing_status(doc_id, "completed")
```

### Document Retrieval

```python
from retrieval.document.manager import DocumentManager

# Initialize manager
doc_manager = DocumentManager(postgres_manager)

# Search documents
results = doc_manager.search_documents("machine learning", limit=20)

# Search chunks with FTS
chunk_results = doc_manager.search_document_chunks_fts("neural networks", k=10)

# Enrich search results
docs = [...]  # LangChain Document objects from vector search
doc_manager.batch_enrich_documents_from_postgres(docs)

# Get statistics
stats = doc_manager.get_document_statistics()
print(f"Total documents: {stats['total_documents']}")
print(f"Search ready: {stats['search_ready_documents']}")
```

### Direct Data Operations

```python
from rag_manager.data.document_data import DocumentDataManager

# Initialize data manager
data_manager = DocumentDataManager(postgres_manager, milvus_manager)

# Direct metadata operations
metadata = {
    'title': 'Research Paper',
    'content_preview': 'This paper discusses...',
    'file_path': '/documents/research.pdf',
    'content_type': 'application/pdf',
    'processing_status': 'completed'
}
data_manager.upsert_document_metadata("research.pdf", metadata)

# Direct chunk operations
chunk_data = {
    'chunk_id': 'uuid-here',
    'document_id': 'doc-uuid-here',
    'chunk_text': 'This is the chunk content...',
    'chunk_index': 1,
    'chunk_metadata': {'page': 1, 'section': 'Introduction'}
}
data_manager.upsert_document_chunk(chunk_data)
```

## Processing Status Values

The module uses standardized processing status values:

- `pending` - Document queued for processing
- `processing` - Document currently being processed
- `completed` - Document successfully processed and indexed
- `error` - Processing failed with errors
- `skipped` - Document skipped due to business rules

## Search Capabilities

### Full-Text Search
- PostgreSQL native FTS with English language support
- Ranking based on tf-idf scoring
- Search across document titles, content previews, and chunk text
- Configurable result limits and performance optimization

### Metadata Search
- Filename-based document lookup
- Document type filtering
- Processing status filtering
- Date range queries

### Hybrid Search Support
- Integration with vector similarity search
- Metadata enrichment for vector results
- Ranking fusion capabilities
- Reranking support for improved relevance

## Performance Considerations

### Connection Pooling
- Uses shared PostgreSQL connection pools
- Configurable pool sizes for concurrent operations
- Automatic connection management and cleanup

### Query Optimization
- Indexed columns for common search patterns
- Batch operations for bulk data processing
- Efficient metadata enrichment with minimal queries

### Memory Management
- Streaming for large document processing
- Configurable chunk sizes
- Lazy loading for search results

## Error Handling

The module implements comprehensive error handling:

- Database connection failures with automatic retry
- Malformed document handling with detailed logging
- Processing timeout management
- Graceful degradation for optional features

All errors are logged with appropriate levels and include contextual information for debugging and monitoring.

## Migration and Compatibility

The refactored module maintains backward compatibility with existing interfaces while providing enhanced capabilities through the new architecture. Existing code using DocumentManager interfaces will continue to work without modification.
