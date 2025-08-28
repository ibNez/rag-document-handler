#!/usr/bin/env python
"""
PostgreSQL Full-Text Search Retriever for Documents
Following DEVELOPMENT_RULES.md for all development requirements

This module provides PostgreSQL-based full-text search retrieval for document chunks.
"""

import logging
from typing import List, Any, Dict, Optional
from langchain_core.documents import Document

from ingestion.core.postgres_manager import PostgreSQLManager

logger = logging.getLogger(__name__)


class DocumentPostgresFTSRetriever:
    """
    PostgreSQL Full-Text Search retriever for document chunks.
    
    Uses PostgreSQL's built-in FTS capabilities with GIN indexes
    for fast text search across document content.
    """
    
    def __init__(self, postgres_manager: PostgreSQLManager) -> None:
        """
        Initialize PostgreSQL FTS retriever for documents.
        
        Args:
            postgres_manager: PostgreSQL manager instance
        """
        self.db_manager = postgres_manager
        logger.info("Document PostgreSQL FTS retriever initialized")
    
    def search(self, query: str, k: int = 10, 
              document_id: Optional[str] = None,
              filetype_filter: Optional[str] = None,
              page_range: Optional[tuple[int, int]] = None) -> List[Document]:
        """
        Search document chunks using PostgreSQL FTS.
        
        Args:
            query: Search query
            k: Number of results to return
            document_id: Optional document ID filter
            filetype_filter: Optional content type filter (e.g., 'application/pdf')
            page_range: Optional page range filter (start_page, end_page)
            
        Returns:
            List of Document objects with FTS scores and metadata
        """
        if not query.strip():
            logger.warning("Empty query provided to document FTS search")
            return []
        
        try:
            logger.info(f"Performing PostgreSQL FTS search for documents: '{query[:50]}...' (k={k})")
            
            with self.db_manager.get_connection() as conn:
                with conn.cursor() as cur:
                    # Build dynamic query with optional filters
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
                            d.created_at as document_created,
                            dc.created_at as chunk_created
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
                        
                    if page_range:
                        start_page, end_page = page_range
                        base_query += " AND (dc.page_start >= %s AND dc.page_end <= %s)"
                        params.extend([str(start_page), str(end_page)])
                    
                    base_query += " ORDER BY fts_score DESC LIMIT %s"
                    params.append(str(k))
                    
                    cur.execute(base_query, params)
                    results = cur.fetchall()
                    logger.info(f"Document PostgreSQL FTS returned {len(results)} results")
                    
                    documents = []
                    for row in results:
                        # Extract fields directly from database
                        id = row['id']
                        document_id = row['document_id']
                        chunk_text = row['chunk_text']
                        chunk_ordinal = row['chunk_ordinal']
                        page_start = row['page_start']
                        page_end = row['page_end']
                        section_path = row['section_path']
                        element_types = row['element_types']
                        token_count = row['token_count']
                        fts_score = row['fts_score']
                        title = row['title']
                        content_type = row['content_type']
                        file_path = row['file_path']
                        document_created = row['document_created']
                        chunk_created = row['chunk_created']
                        
                        # Create LangChain Document with comprehensive metadata
                        doc = Document(
                            page_content=chunk_text,
                            metadata={
                                'id': id,
                                'document_id': document_id,
                                'chunk_ordinal': chunk_ordinal,
                                'page_start': page_start,
                                'page_end': page_end,
                                'section_path': section_path,
                                'element_types': element_types or [],
                                'token_count': token_count,
                                'fts_score': float(fts_score),
                                'title': title,
                                'content_type': content_type,
                                'file_path': file_path,
                                'document_created': document_created,
                                'chunk_created': chunk_created,
                                'category': 'document',
                                'retrieval_method': 'fts'
                            }
                        )
                        documents.append(doc)
                    
                    return documents
                    
        except Exception as e:
            logger.error(f"Document FTS search failed: {e}")
            return []
    
    def search_with_filters(self, query: str, k: int = 10, 
                           filters: Optional[Dict[str, Any]] = None) -> List[Document]:
        """
        Search with complex filtering options.
        
        Args:
            query: Search query
            k: Number of results to return
            filters: Dictionary with filter options:
                - document_ids: List of document IDs to include
                - content_types: List of content types to include
                - element_types: List of element types to include
                - page_range: Tuple of (min_page, max_page)
                - keywords: List of keywords that must be present
                - date_range: Tuple of (start_date, end_date)
                
        Returns:
            List of Document objects with FTS scores and metadata
        """
        if not query.strip():
            return []
            
        filters = filters or {}
        
        try:
            with self.db_manager.get_connection() as conn:
                with conn.cursor() as cur:
                    # Build complex query with multiple filters
                    base_query = """
                        SELECT 
                            dc.id, dc.document_id, dc.chunk_text, dc.chunk_ordinal,
                            dc.page_start, dc.page_end, dc.section_path, dc.element_types,
                            dc.token_count,
                            ts_rank(to_tsvector('english', dc.chunk_text), plainto_tsquery('english', %s)) as fts_score,
                            d.title, d.content_type, d.file_path, d.created_at,
                            d.top_keywords
                        FROM document_chunks dc
                        JOIN documents d ON dc.document_id = d.id
                        WHERE to_tsvector('english', dc.chunk_text) @@ plainto_tsquery('english', %s)
                    """
                    
                    params = [query, query]
                    
                    # Apply filters
                    if filters.get('document_ids'):
                        placeholders = ','.join(['%s'] * len(filters['document_ids']))
                        base_query += f" AND dc.document_id IN ({placeholders})"
                        params.extend(filters['document_ids'])
                        
                    if filters.get('content_types'):
                        placeholders = ','.join(['%s'] * len(filters['content_types']))
                        base_query += f" AND d.content_type IN ({placeholders})"
                        params.extend(filters['content_types'])
                        
                    if filters.get('element_types'):
                        base_query += " AND dc.element_types && %s"
                        params.append(filters['element_types'])
                        
                    if filters.get('page_range'):
                        min_page, max_page = filters['page_range']
                        base_query += " AND dc.page_start >= %s AND dc.page_end <= %s"
                        params.extend([str(min_page), str(max_page)])
                        
                    if filters.get('keywords'):
                        base_query += " AND d.top_keywords && %s"
                        params.append(filters['keywords'])
                        
                    if filters.get('date_range'):
                        start_date, end_date = filters['date_range']
                        base_query += " AND d.created_at BETWEEN %s AND %s"
                        params.extend([start_date, end_date])
                    
                    base_query += " ORDER BY fts_score DESC LIMIT %s"
                    params.append(str(k))
                    
                    cur.execute(base_query, params)
                    results = cur.fetchall()
                    
                    documents = []
                    for row in results:
                        # Extract fields directly from database
                        id = row['id']
                        document_id = row['document_id']
                        chunk_text = row['chunk_text']
                        chunk_ordinal = row['chunk_ordinal']
                        page_start = row['page_start']
                        page_end = row['page_end']
                        section_path = row['section_path']
                        element_types = row['element_types']
                        token_count = row['token_count']
                        fts_score = row['fts_score']
                        title = row['title']
                        content_type = row['content_type']
                        file_path = row['file_path']
                        created_at = row['created_at']
                        top_keywords = row['top_keywords']
                        
                        doc = Document(
                            page_content=chunk_text,
                            metadata={
                                'id': id,
                                'document_id': document_id,
                                'chunk_ordinal': chunk_ordinal,
                                'page_start': page_start,
                                'page_end': page_end,
                                'section_path': section_path,
                                'element_types': element_types or [],
                                'token_count': token_count,
                                'fts_score': float(fts_score),
                                'title': title,
                                'content_type': content_type,
                                'file_path': file_path,
                                'created_at': created_at,
                                'top_keywords': top_keywords or [],
                                'category': 'document',
                                'retrieval_method': 'fts_filtered',
                                'applied_filters': list(filters.keys())
                            }
                        )
                        documents.append(doc)
                    
                    logger.info(f"Document FTS search with filters returned {len(documents)} results")
                    return documents
                    
        except Exception as e:
            logger.error(f"Document FTS search with filters failed: {e}")
            return []
    
    def get_search_statistics(self, query: str) -> Dict[str, Any]:
        """
        Get statistics about search results for a query.
        
        Args:
            query: Search query
            
        Returns:
            Dictionary with search statistics
        """
        try:
            with self.db_manager.get_connection() as conn:
                with conn.cursor() as cur:
                    # Get result counts by content type
                    stats_query = """
                        SELECT 
                            d.content_type,
                            COUNT(*) as chunk_count,
                            COUNT(DISTINCT dc.document_id) as document_count,
                            AVG(ts_rank(to_tsvector('english', dc.chunk_text), plainto_tsquery('english', %s))) as avg_score
                        FROM document_chunks dc
                        JOIN documents d ON dc.document_id = d.id
                        WHERE to_tsvector('english', dc.chunk_text) @@ plainto_tsquery('english', %s)
                        GROUP BY d.content_type
                        ORDER BY chunk_count DESC
                    """
                    
                    cur.execute(stats_query, [query, query])
                    type_stats = cur.fetchall()
                    
                    # Get total statistics
                    total_query = """
                        SELECT 
                            COUNT(*) as total_chunks,
                            COUNT(DISTINCT dc.document_id) as total_documents,
                            MAX(ts_rank(to_tsvector('english', dc.chunk_text), plainto_tsquery('english', %s))) as max_score
                        FROM document_chunks dc
                        WHERE to_tsvector('english', dc.chunk_text) @@ plainto_tsquery('english', %s)
                    """
                    
                    cur.execute(total_query, [query, query])
                    total_stats = cur.fetchone()
                    
                    return {
                        'query': query,
                        'total_chunks': total_stats['total_chunks'] if total_stats else 0,
                        'total_documents': total_stats['total_documents'] if total_stats else 0,
                        'max_score': float(total_stats['max_score']) if total_stats and total_stats['max_score'] else 0.0,
                        'by_content_type': [
                            {
                                'content_type': row['content_type'],
                                'chunk_count': row['chunk_count'],
                                'document_count': row['document_count'],
                                'avg_score': float(row['avg_score'])
                            }
                            for row in type_stats
                        ]
                    }
                    
        except Exception as e:
            logger.error(f"Failed to get search statistics: {e}")
            return {
                'query': query,
                'error': str(e),
                'total_chunks': 0,
                'total_documents': 0,
                'max_score': 0.0,
                'by_content_type': []
            }
