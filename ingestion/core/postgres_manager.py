"""
PostgreSQL Database Manager for RAG Knowledgebase Manager
Handles metadata storage and complex queries for document management.
"""

import json
import logging
import os
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
import psycopg2
from psycopg2.extras import RealDictCursor
from psycopg2.pool import ThreadedConnectionPool
import psycopg2.extensions
from contextlib import contextmanager

logger = logging.getLogger(__name__)

@dataclass
class PostgreSQLConfig:
    """Configuration for PostgreSQL connection."""
    host: str = os.getenv('POSTGRES_HOST', 'localhost')
    port: int = int(os.getenv('POSTGRES_PORT', '5432'))
    database: str = os.getenv('POSTGRES_DB', 'rag_metadata')
    user: str = os.getenv('POSTGRES_USER', 'rag_user')
    password: str = os.getenv('POSTGRES_PASSWORD', 'secure_password')
    min_connections: int = 2
    max_connections: int = 20  # Increased from 10 to handle more concurrent operations

class PostgreSQLManager:
    """PostgreSQL manager for RAG document metadata and analytics."""
    
    def __init__(self, config_or_pool=None):
        """
        Initialize PostgreSQL manager with connection pooling.
        
        Args:
            config_or_pool: Either a PostgreSQLConfig object or a connection pool.
                          If None, creates default config.
        """
        # Check if it's a connection pool (duck typing)
        if hasattr(config_or_pool, 'getconn') and hasattr(config_or_pool, 'putconn'):
            # It's a connection pool - use it directly
            self.pool = config_or_pool
            self.config = None
            logger.info("PostgreSQL manager initialized with existing pool")
        else:
            # It's a config object (or None)
            self.config = config_or_pool or PostgreSQLConfig()
            self._initialize_pool()
            self._ensure_schema()
    
    def _initialize_pool(self) -> None:
        """Initialize connection pool."""
        if not self.config:
            raise ValueError("Cannot initialize pool without config")
            
        try:
            self.pool = ThreadedConnectionPool(
                self.config.min_connections,
                self.config.max_connections,
                host=self.config.host,
                port=self.config.port,
                database=self.config.database,
                user=self.config.user,
                password=self.config.password,
                cursor_factory=RealDictCursor
            )
            logger.info(f"PostgreSQL connection pool initialized for {self.config.database}")
        except Exception as e:
            logger.error(f"Failed to initialize PostgreSQL connection pool: {e}")
            raise
    
    @contextmanager
    def get_connection(self):
        """Context manager for database connections."""
        if not self.pool:
            raise RuntimeError("No connection pool available")
            
        conn = None
        try:
            # Log connection attempt (but avoid accessing private pool attributes)
            logger.debug("Requesting database connection from pool")
            
            conn = self.pool.getconn()
            conn.autocommit = True  # Enable autocommit for read operations
            yield conn
        except Exception as e:
            if conn:
                try:
                    conn.rollback()
                except:
                    pass  # Ignore rollback errors in autocommit mode
            logger.error(f"Database operation failed: {e} (type: {type(e)}) (args: {e.args})")
            raise
        finally:
            if conn:
                self.pool.putconn(conn)
    
    def _ensure_schema(self) -> None:
        """Create necessary tables and indexes. Skip if using external pool."""
        if not self.config:
            # Skip schema creation when using external pool - assume it's already set up
            logger.debug("Skipping schema creation for external pool")
            return
            
        schema_sql = """
        -- Enable required extensions
        CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
        
        -- Documents table for metadata storage (files AND URLs)
        CREATE TABLE IF NOT EXISTS documents (
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
            top_keywords TEXT[], -- PostgreSQL array of strings
            processing_time_seconds REAL,
            processing_status VARCHAR(50) DEFAULT 'pending',
            file_hash VARCHAR(64) UNIQUE,
            created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
            updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
            indexed_at TIMESTAMP WITH TIME ZONE
        );
        
        -- Email documents table (migrated from SQLite)
        CREATE TABLE IF NOT EXISTS emails (
            id SERIAL PRIMARY KEY,
            message_id TEXT UNIQUE NOT NULL,
            from_addr TEXT,
            to_addrs JSONB,
            subject TEXT,
            date_utc TIMESTAMP,
            header_hash TEXT UNIQUE NOT NULL,
            content_hash TEXT UNIQUE,
            content TEXT,
            attachments JSONB,
            headers JSONB,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        
                
        -- Email chunks table for hybrid retrieval
        CREATE TABLE IF NOT EXISTS email_chunks (
            chunk_id VARCHAR(255) PRIMARY KEY,
            email_id TEXT NOT NULL,
            chunk_text TEXT NOT NULL,
            chunk_index INTEGER NOT NULL,
            token_count INTEGER,
            chunk_hash VARCHAR(64),
            created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
            CONSTRAINT fk_email_chunks_email FOREIGN KEY (email_id) REFERENCES emails(message_id) ON DELETE CASCADE,
            CONSTRAINT uk_email_chunks_position UNIQUE(email_id, chunk_index)
        );

        -- Document chunks table for hybrid retrieval (similar to email_chunks)
        CREATE TABLE IF NOT EXISTS document_chunks (
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
        
                -- URLs table (migrated from SQLite)
        CREATE TABLE IF NOT EXISTS urls (
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
        
        -- Email accounts table (compatible with EmailAccountManager interface)
        CREATE TABLE IF NOT EXISTS email_accounts (
            id SERIAL PRIMARY KEY,
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
        
        -- Add total_emails_in_mailbox column if it doesn't exist (migration)
        ALTER TABLE email_accounts ADD COLUMN IF NOT EXISTS total_emails_in_mailbox INTEGER DEFAULT 0;
        
        -- Create indexes for performance
        CREATE INDEX IF NOT EXISTS idx_documents_status ON documents(processing_status);
        CREATE INDEX IF NOT EXISTS idx_documents_created ON documents(created_at);
        CREATE INDEX IF NOT EXISTS idx_documents_keywords ON documents USING GIN(top_keywords);
        
        -- Email indexes
        CREATE INDEX IF NOT EXISTS idx_emails_message_id ON emails(message_id);
        CREATE INDEX IF NOT EXISTS idx_emails_header_hash ON emails(header_hash);
        CREATE INDEX IF NOT EXISTS idx_emails_from_addr ON emails(from_addr);
        CREATE INDEX IF NOT EXISTS idx_emails_date_utc ON emails(date_utc);
        CREATE INDEX IF NOT EXISTS idx_emails_content_hash ON emails(content_hash);
        
        -- Email chunks indexes
        CREATE INDEX IF NOT EXISTS idx_email_chunks_email_id ON email_chunks(email_id);
        CREATE INDEX IF NOT EXISTS idx_email_chunks_hash ON email_chunks(chunk_hash);
        CREATE INDEX IF NOT EXISTS idx_email_chunks_position ON email_chunks(email_id, chunk_index);
        
        -- Document chunks indexes
        CREATE INDEX IF NOT EXISTS idx_document_chunks_document_id ON document_chunks(document_id);
        CREATE INDEX IF NOT EXISTS idx_document_chunks_hash ON document_chunks(chunk_hash);
        CREATE INDEX IF NOT EXISTS idx_document_chunks_position ON document_chunks(document_id, chunk_ordinal);
        CREATE INDEX IF NOT EXISTS idx_document_chunks_page ON document_chunks(page_start, page_end);
        CREATE INDEX IF NOT EXISTS idx_document_chunks_element_types ON document_chunks USING GIN(element_types);
        
        -- Email accounts indexes
        CREATE INDEX IF NOT EXISTS idx_email_accounts_offset ON email_accounts(last_synced_offset);
        
        CREATE INDEX IF NOT EXISTS idx_urls_url ON urls(url);
        CREATE INDEX IF NOT EXISTS idx_urls_status ON urls(status);
        CREATE INDEX IF NOT EXISTS idx_urls_parent_url_id ON urls(parent_url_id);
        
        CREATE INDEX IF NOT EXISTS idx_email_accounts_email ON email_accounts(email_address);
        CREATE INDEX IF NOT EXISTS idx_email_accounts_name ON email_accounts(account_name);
        
        -- Full-text search indexes
        CREATE INDEX IF NOT EXISTS idx_documents_fts ON documents 
        USING GIN(to_tsvector('english', COALESCE(title, '') || ' ' || COALESCE(content_preview, '')));
        
        -- Email FTS index
        CREATE INDEX IF NOT EXISTS idx_emails_fts ON emails 
        USING GIN(to_tsvector('english', COALESCE(subject, '') || ' ' || COALESCE(content, '')));
        
        -- Email chunks FTS index
        CREATE INDEX IF NOT EXISTS idx_email_chunks_fts ON email_chunks 
        USING GIN(to_tsvector('english', chunk_text));
        
        -- Document chunks FTS index
        CREATE INDEX IF NOT EXISTS idx_document_chunks_fts ON document_chunks 
        USING GIN(to_tsvector('english', chunk_text));
        """
        
        with self.get_connection() as conn:
            with conn.cursor() as cur:
                try:
                    cur.execute(schema_sql)
                    conn.commit()
                    logger.info("PostgreSQL schema initialized successfully")
                except Exception as e:
                    conn.rollback()
                    logger.error(f"Failed to initialize schema: {e}")
                    raise
    
    def store_document(self, file_path: str, filename: str, 
                      title: Optional[str] = None, content_preview: Optional[str] = None,
                      content_type: Optional[str] = None, file_size: Optional[int] = None,
                      word_count: Optional[int] = None, document_type: str = 'file', **kwargs) -> str:
        """
        Store document metadata and return the UUID.
        
        Args:
            title: Document title (optional)
            content_preview: Preview of document content (optional)
            file_path: REQUIRED - Full path to file for files, URL for URLs
            filename: REQUIRED - Filename for files, descriptive name for URLs
            content_type: MIME type (optional)
            file_size: File size in bytes (optional)
            word_count: Number of words in document (optional)
            document_type: Type of document ('file' or 'url')
            **kwargs: Additional metadata (ignored)
            
        Returns:
            The UUID of the stored document
            
        Raises:
            ValueError: If required fields are missing or invalid
        """
        # Validate required fields according to DEVELOPMENT_RULES
        if not file_path or not file_path.strip():
            logger.error(f"Missing required field: file_path")
            raise ValueError("file_path is required and cannot be empty")
        
        if not filename or not filename.strip():
            logger.error(f"Missing required field: filename")
            raise ValueError("filename is required and cannot be empty")
        
        logger.info(f"Storing document: file_path='{file_path}', filename='{filename}', type='{document_type}'")
        
        query = """
            INSERT INTO documents (title, content_preview, file_path, filename,
                                 content_type, file_size, word_count, document_type, updated_at)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, NOW())
            RETURNING id
        """
        
        with self.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(query, [
                    title, content_preview, file_path, filename,
                    content_type, file_size, word_count, document_type
                ])
                result = cur.fetchone()
                conn.commit()
                document_id = str(result['id'])
                logger.info(f"Stored document metadata: {document_id}")
                return document_id
    
    def update_processing_status(self, document_id: str, status: str) -> None:
        """Update document processing status by ID."""
        query = """
            UPDATE documents 
            SET processing_status = %s, 
                indexed_at = CASE WHEN %s = 'completed' THEN NOW() ELSE indexed_at END,
                updated_at = NOW()
            WHERE id = %s
        """
        
        with self.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(query, [status, status, document_id])
                conn.commit()
                logger.debug(f"Updated document {document_id} status to {status}")
    
    def search_documents(self, query: str, limit: int = 50) -> List[Dict[str, Any]]:
        """Full-text search across documents."""
        search_sql = """
            SELECT id, title, content_preview, content_type, document_type,
                   page_count, chunk_count, word_count, 
                   avg_chunk_chars, median_chunk_chars, top_keywords,
                   processing_time_seconds, processing_status,
                   created_at, updated_at,
                   ts_rank(to_tsvector('english', COALESCE(title, '') || ' ' || COALESCE(content_preview, '')), 
                          plainto_tsquery('english', %s)) as relevance
            FROM documents
            WHERE to_tsvector('english', COALESCE(title, '') || ' ' || COALESCE(content_preview, '')) 
                  @@ plainto_tsquery('english', %s)
            ORDER BY relevance DESC, created_at DESC
            LIMIT %s
        """
        
        with self.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(search_sql, [query, query, limit])
                results = cur.fetchall()
                return [dict(row) for row in results]
    
    def get_document_analytics(self) -> Dict[str, Any]:
        """Get document processing analytics."""
        analytics_sql = """
            WITH stats AS (
                SELECT 
                    COUNT(*) as total_documents,
                    COUNT(*) FILTER (WHERE processing_status = 'completed') as completed,
                    COUNT(*) FILTER (WHERE processing_status = 'pending') as pending,
                    COUNT(*) FILTER (WHERE processing_status = 'failed') as failed,
                    AVG(word_count) as avg_word_count,
                    SUM(file_size) as total_size
                FROM documents
            ),
            daily_stats AS (
                SELECT 
                    DATE_TRUNC('day', created_at) as date,
                    COUNT(*) as daily_count
                FROM documents
                WHERE created_at >= NOW() - INTERVAL '30 days'
                GROUP BY DATE_TRUNC('day', created_at)
                ORDER BY date DESC
                LIMIT 30
            )
            SELECT 
                json_build_object(
                    'total_documents', s.total_documents,
                    'completed', s.completed,
                    'pending', s.pending,
                    'failed', s.failed,
                    'avg_word_count', ROUND(s.avg_word_count::numeric, 2),
                    'total_size_mb', ROUND((s.total_size / 1024.0 / 1024.0)::numeric, 2),
                    'daily_counts', json_agg(
                        json_build_object('date', ds.date, 'count', ds.daily_count)
                        ORDER BY ds.date DESC
                    )
                ) as analytics
            FROM stats s
            CROSS JOIN daily_stats ds
            GROUP BY s.total_documents, s.completed, s.pending, s.failed, s.avg_word_count, s.total_size
        """
        
        with self.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(analytics_sql)
                result = cur.fetchone()
                return result['analytics'] if result else {}
    
    def get_version_info(self) -> Dict[str, Any]:
        """Get PostgreSQL version and connection info."""
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("SELECT version()")
                    version_result = cur.fetchone()
                    version_str = version_result['version'] if version_result else "Unknown"
                    
                    # Extract just the version number (e.g., "PostgreSQL 15.4")
                    version_short = version_str.split(' on ')[0] if ' on ' in version_str else version_str
                    
                    return {
                        "connected": True,
                        "version": version_short,
                        "full_version": version_str
                    }
        except Exception as e:
            logger.error(f"Failed to get PostgreSQL version: {e}")
            return {
                "connected": False,
                "error": str(e)
            }
    
    def store_document_chunk(self, document_id: str, chunk_text: str, 
                           chunk_ordinal: int, page_start: Optional[int] = None, 
                           page_end: Optional[int] = None, section_path: Optional[str] = None,
                           element_types: Optional[List[str]] = None, token_count: Optional[int] = None,
                           chunk_hash: Optional[str] = None, embedding_version: str = 'mxbai-embed-large') -> str:
        """
        Store document chunk for hybrid retrieval.
        
        Args:
            document_id: Parent document identifier
            chunk_text: Text content of the chunk
            chunk_ordinal: Sequential chunk number within document
            page_start: Starting page number (optional)
            page_end: Ending page number (optional)
            section_path: Hierarchical section path (e.g., "H1 > H2 > List")
            element_types: List of element types from Unstructured
            token_count: Number of tokens in chunk
            chunk_hash: Content hash for deduplication
            embedding_version: Version/model used for embeddings
            
        Returns:
            The UUID of the created chunk
        """
        query = """
            INSERT INTO document_chunks (
                document_id, chunk_text, chunk_ordinal, page_start, page_end,
                section_path, element_types, token_count, chunk_hash, embedding_version
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (document_id, chunk_ordinal) 
            DO UPDATE SET 
                chunk_text = EXCLUDED.chunk_text,
                page_start = EXCLUDED.page_start,
                page_end = EXCLUDED.page_end,
                section_path = EXCLUDED.section_path,
                element_types = EXCLUDED.element_types,
                token_count = EXCLUDED.token_count,
                chunk_hash = EXCLUDED.chunk_hash,
                embedding_version = EXCLUDED.embedding_version
            RETURNING id
        """
        
        with self.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(query, [
                    document_id, chunk_text, chunk_ordinal, page_start, page_end,
                    section_path, element_types, token_count, chunk_hash, embedding_version
                ])
                result = cur.fetchone()
                conn.commit()
                chunk_id = str(result['id'])
                logger.debug(f"Stored document chunk: {chunk_id}")
                return chunk_id
    
    def get_document_chunks(self, document_id: str) -> List[Dict[str, Any]]:
        """
        Get all chunks for a document.
        
        Args:
            document_id: Document identifier
            
        Returns:
            List of chunk dictionaries with metadata
        """
        query = """
            SELECT id, document_id, chunk_text, chunk_ordinal, page_start, page_end,
                   section_path, element_types, token_count, chunk_hash, embedding_version,
                   created_at
            FROM document_chunks 
            WHERE document_id = %s 
            ORDER BY chunk_ordinal
        """
        
        with self.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(query, [document_id])
                return [dict(row) for row in cur.fetchall()]
    
    def search_document_chunks_fts(self, query: str, limit: int = 20, 
                                 document_id: Optional[str] = None,
                                 filetype_filter: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Full-text search across document chunks with optional filtering.
        
        Args:
            query: Search query
            limit: Maximum number of results
            document_id: Optional document ID filter
            filetype_filter: Optional content type filter
            
        Returns:
            List of chunks with FTS scores and metadata
        """
        base_query = """
            SELECT 
                dc.id,
                dc.document_id,
                dc.chunk_text,
                dc.chunk_ordinal,
                dc.page_start,
                dc.page_end,
                dc.section_path,
                dc.element_types,
                dc.token_count,
                ts_rank(to_tsvector('english', dc.chunk_text), plainto_tsquery('english', %s)) as fts_score,
                d.title,
                d.content_type,
                d.file_path,
                d.created_at
            FROM document_chunks dc
            JOIN documents d ON dc.document_id = d.id
            WHERE to_tsvector('english', dc.chunk_text) @@ plainto_tsquery('english', %s)
        """
        
        params = [query, query]
        
        # Add optional filters
        if document_id:
            base_query += " AND dc.document_id = %s"
            params.append(document_id)
            
        if filetype_filter:
            base_query += " AND d.content_type = %s"
            params.append(filetype_filter)
        
        base_query += " ORDER BY fts_score DESC LIMIT %s"
        params.append(str(limit))
        
        with self.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(base_query, params)
                return [dict(row) for row in cur.fetchall()]
    
    def delete_document_chunks(self, document_id: str) -> int:
        """
        Delete all chunks for a document.
        
        Args:
            document_id: Document identifier
            
        Returns:
            Number of deleted chunks
        """
        query = "DELETE FROM document_chunks WHERE document_id = %s"
        
        with self.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(query, [document_id])
                deleted_count = cur.rowcount
                conn.commit()
                logger.info(f"Deleted {deleted_count} chunks for document: {document_id}")
                return deleted_count
    
    def close(self) -> None:
        """Close connection pool."""
        if self.pool:
            self.pool.closeall()
            logger.info("PostgreSQL connection pool closed")
