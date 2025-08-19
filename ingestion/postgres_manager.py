"""
PostgreSQL Database Manager for RAG Document Handler
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
    min_connections: int = 1
    max_connections: int = 10

class PostgreSQLManager:
    """PostgreSQL manager for RAG document metadata and analytics."""
    
    def __init__(self, config: Optional[PostgreSQLConfig] = None):
        """Initialize PostgreSQL manager with connection pooling."""
        self.config = config or PostgreSQLConfig()
        self.pool: Optional[ThreadedConnectionPool] = None
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
        conn = None
        try:
            conn = self.pool.getconn()
            conn.autocommit = True  # Enable autocommit for read operations
            yield conn
        except Exception as e:
            if conn:
                try:
                    conn.rollback()
                except:
                    pass  # Ignore rollback errors in autocommit mode
            logger.error(f"Database operation failed: {e}")
            raise
        finally:
            if conn:
                self.pool.putconn(conn)
    
    def _ensure_schema(self) -> None:
        """Create necessary tables and indexes."""
        schema_sql = """
        -- Enable required extensions
        CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
        
        -- Documents table for metadata storage
        CREATE TABLE IF NOT EXISTS documents (
            id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
            document_id VARCHAR(255) UNIQUE NOT NULL,
            title TEXT,
            content_preview TEXT,
            file_path TEXT,
            content_type VARCHAR(100),
            file_size BIGINT,
            word_count INTEGER,
            processing_status VARCHAR(50) DEFAULT 'pending',
            metadata JSONB DEFAULT '{}',
            created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
            updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
            indexed_at TIMESTAMP WITH TIME ZONE
        );
        
        -- Email documents table (migrated from SQLite)
        CREATE TABLE IF NOT EXISTS emails (
            id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
            message_id VARCHAR(255) UNIQUE NOT NULL,
            subject TEXT,
            sender VARCHAR(255),
            recipient VARCHAR(255),
            date_sent TIMESTAMP WITH TIME ZONE,
            body_text TEXT,
            body_html TEXT,
            attachments JSONB DEFAULT '[]',
            header_hash VARCHAR(64) UNIQUE,
            server_type VARCHAR(50),
            account_id VARCHAR(255),
            folder VARCHAR(255),
            metadata JSONB DEFAULT '{}',
            processed_at TIMESTAMP WITH TIME ZONE,
            created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
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
            metadata JSONB DEFAULT '{}',
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
            last_update_status TEXT
        );
        
        -- Create indexes for performance
        CREATE INDEX IF NOT EXISTS idx_documents_document_id ON documents(document_id);
        CREATE INDEX IF NOT EXISTS idx_documents_status ON documents(processing_status);
        CREATE INDEX IF NOT EXISTS idx_documents_created ON documents(created_at);
        CREATE INDEX IF NOT EXISTS idx_documents_metadata ON documents USING GIN(metadata);
        
        CREATE INDEX IF NOT EXISTS idx_emails_message_id ON emails(message_id);
        CREATE INDEX IF NOT EXISTS idx_emails_header_hash ON emails(header_hash);
        CREATE INDEX IF NOT EXISTS idx_emails_sender ON emails(sender);
        CREATE INDEX IF NOT EXISTS idx_emails_date_sent ON emails(date_sent);
        CREATE INDEX IF NOT EXISTS idx_emails_metadata ON emails USING GIN(metadata);
        
        CREATE INDEX IF NOT EXISTS idx_urls_url ON urls(url);
        CREATE INDEX IF NOT EXISTS idx_urls_status ON urls(status);
        CREATE INDEX IF NOT EXISTS idx_urls_metadata ON urls USING GIN(metadata);
        
        CREATE INDEX IF NOT EXISTS idx_email_accounts_email ON email_accounts(email_address);
        CREATE INDEX IF NOT EXISTS idx_email_accounts_name ON email_accounts(account_name);
        
        -- Full-text search indexes
        CREATE INDEX IF NOT EXISTS idx_documents_fts ON documents 
        USING GIN(to_tsvector('english', COALESCE(title, '') || ' ' || COALESCE(content_preview, '')));
        
        CREATE INDEX IF NOT EXISTS idx_emails_fts ON emails 
        USING GIN(to_tsvector('english', COALESCE(subject, '') || ' ' || COALESCE(body_text, '')));
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
    
    def store_document(self, document_id: str, title: Optional[str] = None, content_preview: Optional[str] = None,
                      file_path: Optional[str] = None, content_type: Optional[str] = None, file_size: Optional[int] = None,
                      word_count: Optional[int] = None, metadata: Optional[Dict[str, Any]] = None) -> str:
        """Store document metadata."""
        metadata = metadata or {}
        
        query = """
            INSERT INTO documents (document_id, title, content_preview, file_path, 
                                 content_type, file_size, word_count, metadata, updated_at)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, NOW())
            ON CONFLICT (document_id) 
            DO UPDATE SET 
                title = EXCLUDED.title,
                content_preview = EXCLUDED.content_preview,
                file_path = EXCLUDED.file_path,
                content_type = EXCLUDED.content_type,
                file_size = EXCLUDED.file_size,
                word_count = EXCLUDED.word_count,
                metadata = documents.metadata || EXCLUDED.metadata,
                updated_at = NOW()
            RETURNING id
        """
        
        with self.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(query, [
                    document_id, title, content_preview, file_path,
                    content_type, file_size, word_count, json.dumps(metadata)
                ])
                result = cur.fetchone()
                conn.commit()
                logger.info(f"Stored document metadata: {document_id}")
                return str(result['id'])
    
    def update_processing_status(self, document_id: str, status: str) -> None:
        """Update document processing status."""
        query = """
            UPDATE documents 
            SET processing_status = %s, 
                indexed_at = CASE WHEN %s = 'completed' THEN NOW() ELSE indexed_at END,
                updated_at = NOW()
            WHERE document_id = %s
        """
        
        with self.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(query, [status, status, document_id])
                conn.commit()
                logger.debug(f"Updated document {document_id} status to {status}")
    
    def search_documents(self, query: str, limit: int = 50) -> List[Dict[str, Any]]:
        """Full-text search across documents."""
        search_sql = """
            SELECT document_id, title, content_preview, content_type, 
                   metadata, created_at, updated_at,
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
    
    def close(self) -> None:
        """Close connection pool."""
        if self.pool:
            self.pool.closeall()
            logger.info("PostgreSQL connection pool closed")
