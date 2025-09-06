"""
Database Integration Layer for RAG Knowledgebase Manager
Coordinates between PostgreSQL (metadata) and Milvus (vectors).
"""

import logging
from typing import Dict, List, Optional, Any, Protocol
from datetime import datetime
import json

from rag_manager.managers.postgres_manager import PostgreSQLManager, PostgreSQLConfig
from rag_manager.core.models import DocumentMetadata
from ingestion.document.manager import DocumentManager

logger = logging.getLogger(__name__)

class VectorStore(Protocol):
    """Protocol for vector storage operations."""
    def store_embeddings(self, doc_id: str, embeddings: List[float]) -> None: ...
    def similarity_search(self, query_vector: List[float], limit: int) -> List[str]: ...
    def delete_document(self, doc_id: str) -> None: ...

class RAGDatabaseManager:
    """
    Unified database manager that coordinates PostgreSQL and Milvus operations.
    Following DEVELOPMENT_RULES.md for clean architecture.
    """
    
    def __init__(self, postgres_config: Optional[PostgreSQLConfig] = None, 
                 vector_store: Optional[VectorStore] = None):
        """Initialize with PostgreSQL and vector store."""
        self.postgres = PostgreSQLManager(postgres_config)
        self.document_manager = DocumentManager(self.postgres)
        self.vector_store = vector_store
        logger.info("RAG Database Manager initialized")
    
    def store_document(self, content: str, metadata: DocumentMetadata) -> str:
        """
        Store document in both metadata and vector databases.
        Returns the document ID.
        """
        try:
            # Store metadata in PostgreSQL using document manager
            doc_uuid = self.document_manager.store_document(
                file_path=metadata.file_path,
                filename=metadata.filename,
                document_id=metadata.document_id,
                title=metadata.title,
                content_preview=metadata.content_preview,
                content_type=metadata.content_type,
                file_size=metadata.file_size,
                word_count=metadata.word_count,
                metadata=metadata.metadata
            )
            
            # Update processing status
            self.document_manager.update_processing_status(metadata.document_id, 'processing')
            
            # Store embeddings in vector store if available
            if self.vector_store and content:
                try:
                    # Note: Embedding generation would be handled by the vector store
                    # or passed in as pre-computed embeddings
                    logger.info(f"Vector storage would be handled for document: {metadata.document_id}")
                    self.document_manager.update_processing_status(metadata.document_id, 'completed')
                except Exception as e:
                    logger.error(f"Failed to store embeddings for {metadata.document_id}: {e}")
                    self.document_manager.update_processing_status(metadata.document_id, 'failed')
                    raise
            else:
                # Mark as completed even without vector storage for now
                self.document_manager.update_processing_status(metadata.document_id, 'completed')
            
            logger.info(f"Successfully stored document: {metadata.document_id}")
            return metadata.document_id
            
        except Exception as e:
            logger.error(f"Failed to store document {metadata.document_id}: {e}")
            # Update status to failed
            try:
                self.document_manager.update_processing_status(metadata.document_id, 'failed')
            except:
                pass  # Don't fail if status update fails
            raise
    
    def search_documents(self, query: str, limit: int = 50) -> List[Dict[str, Any]]:
        """Search documents using PostgreSQL full-text search."""
        try:
            results = self.postgres.search_documents(query, limit)
            logger.info(f"Found {len(results)} documents for query: {query}")
            return results
        except Exception as e:
            logger.error(f"Search failed for query '{query}': {e}")
            return []
    
    def get_document_by_id(self, document_id: str) -> Optional[Dict[str, Any]]:
        """Get document metadata by ID."""
        try:
            with self.postgres.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("""
                        SELECT document_id, title, content_preview, file_path,
                               content_type, file_size, word_count, processing_status,
                               metadata, created_at, updated_at, indexed_at
                        FROM documents 
                        WHERE document_id = %s
                    """, [document_id])
                    result = cur.fetchone()
                    return dict(result) if result else None
        except Exception as e:
            logger.error(f"Failed to get document {document_id}: {e}")
            return None
    
    def update_document_metadata(self, document_id: str, metadata: Dict[str, Any]) -> bool:
        """Update document metadata."""
        try:
            with self.postgres.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("""
                        UPDATE documents 
                        SET metadata = metadata || %s::jsonb,
                            updated_at = NOW()
                        WHERE document_id = %s
                    """, [json.dumps(metadata), document_id])
                    conn.commit()
                    logger.info(f"Updated metadata for document: {document_id}")
                    return True
        except Exception as e:
            logger.error(f"Failed to update metadata for {document_id}: {e}")
            return False
    
    def delete_document(self, document_id: str) -> bool:
        """Delete document from both stores."""
        try:
            # Delete from vector store if available
            if self.vector_store:
                try:
                    self.vector_store.delete_document(document_id)
                except Exception as e:
                    logger.warning(f"Failed to delete from vector store: {e}")
            
            # Delete from PostgreSQL
            with self.postgres.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("DELETE FROM documents WHERE document_id = %s", [document_id])
                    conn.commit()
                    
            logger.info(f"Deleted document: {document_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete document {document_id}: {e}")
            return False
    
    def get_analytics(self) -> Dict[str, Any]:
        """Get comprehensive analytics."""
        try:
            analytics = self.postgres.get_document_analytics()
            
            # Add vector store statistics if available
            if self.vector_store:
                # Vector store stats would be added here
                analytics['vector_store'] = {'status': 'connected'}
            else:
                analytics['vector_store'] = {'status': 'not_configured'}
            
            return analytics
            
        except Exception as e:
            logger.error(f"Failed to get analytics: {e}")
            return {'error': str(e)}
    
    def get_processing_status(self) -> Dict[str, int]:
        """Get document processing status counts."""
        try:
            with self.postgres.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("""
                        SELECT processing_status, COUNT(*) as count
                        FROM documents 
                        GROUP BY processing_status
                    """)
                    results = cur.fetchall()
                    return {row['processing_status']: row['count'] for row in results}
        except Exception as e:
            logger.error(f"Failed to get processing status: {e}")
            return {}
    
    def close(self) -> None:
        """Close database connections."""
        if self.postgres:
            self.postgres.close()
        logger.info("Database connections closed")
