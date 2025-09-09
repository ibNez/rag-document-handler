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
    
    def __init__(self, config: Optional[PostgreSQLConfig] = None):
        """
        Initialize PostgreSQL manager with connection pooling.
        
        Args:
            config: PostgreSQL configuration object. If None, creates default config.
        """
        self.config = config or PostgreSQLConfig()
        self._initialize_pool()
        self._ensure_schema()
    
    def _initialize_pool(self) -> None:
        """Initialize connection pool."""
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
        """Create necessary tables and indexes."""
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
            keywords TEXT, -- Comma-separated keywords for efficient search
            processing_time_seconds REAL,
            processing_status VARCHAR(50) DEFAULT 'pending',
            file_hash VARCHAR(64) UNIQUE,
            created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
            updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
            indexed_at TIMESTAMP WITH TIME ZONE
        );
        
        -- Email documents table
        CREATE TABLE IF NOT EXISTS emails (
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
        
                
        -- Email chunks table for retrieval
        CREATE TABLE IF NOT EXISTS email_chunks (
            id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
            email_id UUID NOT NULL,
            chunk_text TEXT NOT NULL,
            chunk_index INTEGER NOT NULL,
            token_count INTEGER,
            chunk_hash VARCHAR(64),
            created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
            CONSTRAINT fk_email_chunks_email FOREIGN KEY (email_id) REFERENCES emails(id) ON DELETE CASCADE,
            CONSTRAINT uk_email_chunks_position UNIQUE(email_id, chunk_index)
        );

        -- URLs table: used to store URL Parent information
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

        -- URL snapshots table for point-in-time captures of crawled pages (MUST come before document_chunks)
        CREATE TABLE IF NOT EXISTS url_snapshots (
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

    -- Document chunks table for retrieval (MUST come after url_snapshots for FK reference)
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
            topics TEXT NULL,
            embedding_version VARCHAR(50) DEFAULT 'mxbai-embed-large',
            snapshot_id UUID REFERENCES url_snapshots(id) ON DELETE SET NULL, -- NULL for file documents, populated for URL documents
            created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
            CONSTRAINT fk_document_chunks_document FOREIGN KEY (document_id) REFERENCES documents(id) ON DELETE CASCADE,
            CONSTRAINT uk_document_chunks_position UNIQUE(document_id, chunk_ordinal)
        );

        -- Indexes for snapshots
        CREATE INDEX IF NOT EXISTS idx_url_snapshots_document_id ON url_snapshots(document_id);
        CREATE INDEX IF NOT EXISTS idx_url_snapshots_url_id ON url_snapshots(url_id);
        CREATE INDEX IF NOT EXISTS idx_url_snapshots_snapshot_ts ON url_snapshots(snapshot_ts);
        
        -- Index for document chunks snapshot relationship
        CREATE INDEX IF NOT EXISTS idx_document_chunks_snapshot_id ON document_chunks(snapshot_id);
        
        -- Email accounts table 
        CREATE TABLE IF NOT EXISTS email_accounts (
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
        
        -- Create indexes for performance
        CREATE INDEX IF NOT EXISTS idx_documents_status ON documents(processing_status);
        CREATE INDEX IF NOT EXISTS idx_documents_created ON documents(created_at);
        CREATE INDEX IF NOT EXISTS idx_documents_keywords_search ON documents 
            USING GIN(to_tsvector('english', keywords));
        CREATE INDEX IF NOT EXISTS idx_documents_content_search ON documents 
            USING GIN(to_tsvector('english', title || ' ' || COALESCE(content_preview, '')));
        
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
        CREATE INDEX IF NOT EXISTS idx_document_chunks_topics_gin ON document_chunks USING gin(to_tsvector('english', topics)) WHERE topics IS NOT NULL;
        
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

    # Document storage and management - MOVED TO ingestion/document/manager.py
    # Methods moved: store_document, update_processing_status, store_document_chunk, delete_document_chunks
    
    def search_documents(self, query: str, limit: int = 50) -> List[Dict[str, Any]]:
        """Full-text search across documents."""
        search_sql = """
            SELECT id, title, content_preview, content_type, document_type,
                   page_count, chunk_count, word_count, 
                   avg_chunk_chars, median_chunk_chars, keywords,
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

    # Document chunk methods - MOVED TO ingestion/document/manager.py
    # Methods moved: store_document_chunk, delete_document_chunks

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
                dc.id AS document_chunk_id,
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
                rows = [dict(row) for row in cur.fetchall()]
                return rows

    def close_pool(self) -> None:
        """Close the connection pool."""
        if self.pool:
            self.pool.closeall()
            self.pool = None
            logger.info("PostgreSQL connection pool closed")
    
    def close(self) -> None:
        """Close connection pool."""
        if self.pool:
            self.pool.closeall()
            logger.info("PostgreSQL connection pool closed")
    
    def get_pool_status(self) -> dict:
        """
        Get connection pool status information.
        
        Returns:
            Dictionary with pool status details
        """
        if not self.pool:
            return {'status': 'not_initialized'}
            
        # Note: psycopg2 ThreadedConnectionPool doesn't expose internal stats
        # This is a basic status check
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("SELECT 1")
                    result = cur.fetchone()
            
            return {
                'status': 'healthy',
                'database': self.config.database,
                'host': self.config.host,
                'port': self.config.port,
                'test_query': 'success' if result else 'failed'
            }
        except Exception as e:
            return {
                'status': 'error',
                'error': str(e),
                'database': self.config.database,
                'host': self.config.host,
                'port': self.config.port
            }


# Global instance for easy access
_postgres_manager: Optional['PostgreSQLManager'] = None


def get_postgres_manager(config: Optional[PostgreSQLConfig] = None) -> 'PostgreSQLManager':
    """
    Get the global PostgreSQL manager instance.
    
    Args:
        config: Optional configuration for first-time initialization
        
    Returns:
        PostgreSQL manager instance
    """
    global _postgres_manager
    
    if _postgres_manager is None:
        _postgres_manager = PostgreSQLManager(config)
    
    return _postgres_manager


def close_postgres_connections() -> None:
    """Close all PostgreSQL connections."""
    global _postgres_manager
    
    if _postgres_manager:
        _postgres_manager.close_pool()
        _postgres_manager = None


__all__ = ["PostgreSQLManager", "PostgreSQLConfig", "get_postgres_manager", "close_postgres_connections"]
