"""
Database migration to enhance document metadata according to POC specification.

This migration adds missing fields for enhanced document metadata:
- authors, tags, lang, content_hash to documents table
- tsvector column for full-text search to document_chunks table

Following DEVELOPMENT_RULES.md for all development requirements.
"""

import logging
from typing import Optional
from contextlib import contextmanager
import psycopg2
from psycopg2.extras import RealDictCursor

logger = logging.getLogger(__name__)


class DocumentMetadataEnhancementMigration:
    """
    Database migration to enhance document metadata.
    
    Adds missing fields from POC specification for better
    document management and search capabilities.
    """
    
    def __init__(self, postgres_manager):
        """
        Initialize migration with PostgreSQL manager.
        
        Args:
            postgres_manager: PostgreSQLManager instance
        """
        self.postgres_manager = postgres_manager
        
    def run_migration(self) -> bool:
        """
        Run the complete migration.
        
        Returns:
            True if migration successful, False otherwise
        """
        try:
            logger.info("Starting document metadata enhancement migration...")
            
            # Check if migration is needed
            if self._is_migration_needed():
                logger.info("Migration needed, proceeding with schema updates...")
                
                # Run migration steps
                self._add_document_metadata_fields()
                self._add_document_chunks_fts()
                self._create_indexes()
                
                logger.info("Document metadata enhancement migration completed successfully")
                return True
            else:
                logger.info("Migration not needed, schema is already up to date")
                return True
                
        except Exception as e:
            logger.error(f"Migration failed: {e}")
            return False
    
    def _is_migration_needed(self) -> bool:
        """Check if migration is needed by checking for missing columns."""
        try:
            with self.postgres_manager.get_connection() as conn:
                with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                    # Check if authors column exists in documents table
                    cursor.execute("""
                        SELECT column_name 
                        FROM information_schema.columns 
                        WHERE table_name = 'documents' 
                        AND column_name = 'authors'
                    """)
                    authors_exists = cursor.fetchone() is not None
                    
                    # Check if tsv column exists in document_chunks table
                    cursor.execute("""
                        SELECT column_name 
                        FROM information_schema.columns 
                        WHERE table_name = 'document_chunks' 
                        AND column_name = 'tsv'
                    """)
                    tsv_exists = cursor.fetchone() is not None
                    
                    return not (authors_exists and tsv_exists)
                    
        except Exception as e:
            logger.error(f"Error checking migration status: {e}")
            return False
    
    def _add_document_metadata_fields(self) -> None:
        """Add missing metadata fields to documents table."""
        logger.info("Adding missing metadata fields to documents table...")
        
        migration_sql = """
        -- Add missing metadata fields to documents table if they don't exist
        
        -- Add source_uri (aliased to file_path for now)
        DO $$ 
        BEGIN 
            IF NOT EXISTS (
                SELECT 1 FROM information_schema.columns 
                WHERE table_name = 'documents' AND column_name = 'source_uri'
            ) THEN
                ALTER TABLE documents ADD COLUMN source_uri TEXT;
                -- Copy existing file_path values to source_uri
                UPDATE documents SET source_uri = file_path WHERE file_path IS NOT NULL;
            END IF;
        END $$;
        
        -- Add authors array
        DO $$ 
        BEGIN 
            IF NOT EXISTS (
                SELECT 1 FROM information_schema.columns 
                WHERE table_name = 'documents' AND column_name = 'authors'
            ) THEN
                ALTER TABLE documents ADD COLUMN authors TEXT[];
            END IF;
        END $$;
        
        -- Add tags array
        DO $$ 
        BEGIN 
            IF NOT EXISTS (
                SELECT 1 FROM information_schema.columns 
                WHERE table_name = 'documents' AND column_name = 'tags'
            ) THEN
                ALTER TABLE documents ADD COLUMN tags TEXT[];
            END IF;
        END $$;
        
        -- Add language with default
        DO $$ 
        BEGIN 
            IF NOT EXISTS (
                SELECT 1 FROM information_schema.columns 
                WHERE table_name = 'documents' AND column_name = 'lang'
            ) THEN
                ALTER TABLE documents ADD COLUMN lang TEXT DEFAULT 'english';
            END IF;
        END $$;
        
        -- Add content_hash (separate from file_hash)
        DO $$ 
        BEGIN 
            IF NOT EXISTS (
                SELECT 1 FROM information_schema.columns 
                WHERE table_name = 'documents' AND column_name = 'content_hash'
            ) THEN
                ALTER TABLE documents ADD COLUMN content_hash TEXT;
            END IF;
        END $$;
        """
        
        with self.postgres_manager.get_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute(migration_sql)
                conn.commit()
                
        logger.info("Document metadata fields added successfully")
    
    def _add_document_chunks_fts(self) -> None:
        """Add full-text search tsvector column to document_chunks table."""
        logger.info("Adding full-text search column to document_chunks table...")
        
        fts_sql = """
        -- Add tsvector column for full-text search
        DO $$ 
        BEGIN 
            IF NOT EXISTS (
                SELECT 1 FROM information_schema.columns 
                WHERE table_name = 'document_chunks' AND column_name = 'tsv'
            ) THEN
                -- Add the tsvector column as a generated column
                ALTER TABLE document_chunks 
                ADD COLUMN tsv tsvector 
                GENERATED ALWAYS AS (to_tsvector('english', coalesce(chunk_text, ''))) STORED;
                
                -- Update existing rows by forcing regeneration
                UPDATE document_chunks SET chunk_text = chunk_text;
            END IF;
        END $$;
        """
        
        with self.postgres_manager.get_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute(fts_sql)
                conn.commit()
                
        logger.info("Full-text search column added successfully")
    
    def _create_indexes(self) -> None:
        """Create indexes for new metadata fields."""
        logger.info("Creating indexes for enhanced metadata...")
        
        index_sql = """
        -- Create indexes for new metadata fields
        
        -- GIN index for authors array
        CREATE INDEX IF NOT EXISTS idx_documents_authors 
        ON documents USING GIN(authors);
        
        -- GIN index for tags array
        CREATE INDEX IF NOT EXISTS idx_documents_tags 
        ON documents USING GIN(tags);
        
        -- B-tree index for language
        CREATE INDEX IF NOT EXISTS idx_documents_lang 
        ON documents(lang);
        
        -- B-tree index for content_hash
        CREATE INDEX IF NOT EXISTS idx_documents_content_hash 
        ON documents(content_hash);
        
        -- GIN index for full-text search on document_chunks.tsv
        CREATE INDEX IF NOT EXISTS idx_document_chunks_tsv 
        ON document_chunks USING GIN(tsv);
        
        -- Combined index for filetype and modified_at (from POC spec)
        CREATE INDEX IF NOT EXISTS idx_documents_filetype_modified 
        ON documents(content_type, updated_at);
        """
        
        with self.postgres_manager.get_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute(index_sql)
                conn.commit()
                
        logger.info("Indexes created successfully")
    
    def get_migration_info(self) -> dict:
        """Get information about the migration status."""
        try:
            with self.postgres_manager.get_connection() as conn:
                with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                    # Check documents table columns
                    cursor.execute("""
                        SELECT column_name, data_type 
                        FROM information_schema.columns 
                        WHERE table_name = 'documents'
                        AND column_name IN ('source_uri', 'authors', 'tags', 'lang', 'content_hash')
                        ORDER BY column_name
                    """)
                    documents_columns = cursor.fetchall()
                    
                    # Check document_chunks table columns
                    cursor.execute("""
                        SELECT column_name, data_type 
                        FROM information_schema.columns 
                        WHERE table_name = 'document_chunks'
                        AND column_name = 'tsv'
                    """)
                    chunks_columns = cursor.fetchall()
                    
                    # Check indexes
                    cursor.execute("""
                        SELECT indexname 
                        FROM pg_indexes 
                        WHERE tablename IN ('documents', 'document_chunks')
                        AND indexname LIKE 'idx_document%'
                        ORDER BY indexname
                    """)
                    indexes = cursor.fetchall()
                    
                    return {
                        'documents_enhanced_columns': [col['column_name'] for col in documents_columns],
                        'document_chunks_fts_enabled': len(chunks_columns) > 0,
                        'enhanced_indexes': [idx['indexname'] for idx in indexes],
                        'migration_needed': self._is_migration_needed()
                    }
                    
        except Exception as e:
            logger.error(f"Error getting migration info: {e}")
            return {'error': str(e)}


def run_document_metadata_migration(postgres_manager) -> bool:
    """
    Convenience function to run document metadata enhancement migration.
    
    Args:
        postgres_manager: PostgreSQLManager instance
        
    Returns:
        True if migration successful, False otherwise
    """
    migration = DocumentMetadataEnhancementMigration(postgres_manager)
    return migration.run_migration()


if __name__ == "__main__":
    # Example usage for testing
    import sys
    import os
    
    # Add parent directory to path for imports
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    from ingestion.core.postgres_manager import PostgreSQLManager, PostgreSQLConfig
    
    # Initialize PostgreSQL manager
    config = PostgreSQLConfig()
    postgres_manager = PostgreSQLManager(config)
    
    # Run migration
    migration = DocumentMetadataEnhancementMigration(postgres_manager)
    
    # Show current status
    print("Migration Info:", migration.get_migration_info())
    
    # Run migration if needed
    success = migration.run_migration()
    print(f"Migration {'successful' if success else 'failed'}")
    
    # Show final status
    print("Final Migration Info:", migration.get_migration_info())
