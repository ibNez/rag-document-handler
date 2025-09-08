#!/usr/bin/env python
"""
Email Database Manager
Following DEVELOPMENT_RULES.md for all development requirements

This module handles all PostgreSQL and Milvus database operations specific to email retrieval.
Provides a clean interface between processors and database managers for email data.
"""

import json
import logging
from datetime import datetime, UTC
from typing import Any, Dict, List, Optional

from langchain_core.documents import Document

logger = logging.getLogger(__name__)


class EmailManager:
    """
    Email database manager for PostgreSQL and Milvus operations.
    
    Handles all database interactions for email retrieval functionality,
    providing a clean separation between business logic and data access.
    """
    
    def __init__(self, postgres_manager: Any, milvus_manager: Optional[Any] = None) -> None:
        """
        Initialize email database manager.
        
        Args:
            postgres_manager: PostgreSQL manager instance with connection pool
            milvus_manager: Optional Milvus manager for vector operations
        """
        self.db_manager = postgres_manager
        logger.info("Email Manager initialized for retrieval operations")
        self.milvus_manager = milvus_manager
        
        # Hybrid retrieval components (initialized on demand)
        self.postgres_fts_retriever: Optional[Any] = None
        self.hybrid_retriever: Optional[Any] = None
        
        logger.info("EmailManager initialized for database operations")

    # =============================================================================
    # PostgreSQL Email Operations
    # =============================================================================
    
    def upsert_email(self, record: Dict[str, Any]) -> str:
        """
        Upsert an email record into PostgreSQL database.
        
        Args:
            record: Email record dictionary with required fields
            
        Returns:
            The database-generated UUID id for the stored email
        """
        required_fields = ["message_id"]  # Only message_id is truly required - content can be empty
        missing_fields = [field for field in required_fields if field not in record or record.get(field) is None]
        
        if missing_fields:
            raise ValueError(f"Email record missing required fields: {missing_fields}")
        
        message_id = record["message_id"]
        logger.info(f"Upserting email record for message_id: {message_id}")
        
        try:
            # Use the email data manager for the actual database operation
            from rag_manager.data.email_data import EmailDataManager
            email_data_manager = EmailDataManager(self.db_manager)
            email_id = email_data_manager.upsert_email(record)
            
            logger.debug(f"Successfully upserted email {message_id} with ID: {email_id}")
            return email_id
            
        except Exception as e:
            logger.error(f"Failed to upsert email {message_id}: {e}")
            raise

    def get_email_statistics(self, email_address: str) -> Dict[str, int]:
        """
        Get email statistics for a specific email address.
        
        Args:
            email_address: Email address to get statistics for
            
        Returns:
            Dictionary with email counts and statistics
        """
        try:
            with self.db_manager.get_connection() as conn:
                with conn.cursor() as cur:
                    # Get email counts for this address
                    cur.execute("""
                        SELECT 
                            COUNT(*) as total_emails,
                            COUNT(CASE WHEN content IS NOT NULL AND content != '' THEN 1 END) as synced_emails
                        FROM emails
                        WHERE from_addr = %s OR to_addrs::jsonb ? %s
                    """, (email_address, email_address))
                    
                    result = cur.fetchone()
                    if result:
                        total_emails, synced_emails = result
                    else:
                        total_emails, synced_emails = 0, 0
                    
                    # Get chunk statistics
                    cur.execute("""
                        SELECT COUNT(*) as total_chunks
                        FROM email_chunks ec
                        JOIN emails e ON ec.email_id = e.message_id
                        WHERE e.from_addr = %s OR e.to_addrs::jsonb ? %s
                    """, (email_address, email_address))
                    
                    chunk_result = cur.fetchone()
                    total_chunks = chunk_result[0] if chunk_result else 0
                    
                    return {
                        'total_emails': total_emails,
                        'synced_emails': synced_emails,
                        'total_chunks': total_chunks
                    }
        except Exception as e:
            logger.error(f"Failed to get email statistics for {email_address}: {e}")
            return {
                'total_emails': 0,
                'synced_emails': 0,
                'total_chunks': 0
            }

    def update_total_emails_in_mailbox(self, account_id: str, total_emails: int) -> None:
        """
        Update the total number of emails in the mailbox for an account.
        
        Args:
            account_id: Email account ID (UUID string)
            total_emails: Total number of emails found in the mailbox
        """
        try:
            with self.db_manager.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("""
                        UPDATE email_accounts 
                        SET total_emails_in_mailbox = %s
                        WHERE id = %s
                    """, (total_emails, account_id))
                conn.commit()
                logger.debug(f"Updated total emails count for account {account_id}: {total_emails}")
                
        except Exception as e:
            logger.error(f"Failed to update total emails count for account {account_id}: {e}")
            raise

    # =============================================================================
    # Email Chunk Operations
    # =============================================================================
    
    def search_chunks_for_email(self, email_id: str) -> List[Dict[str, Any]]:
        """
        Retrieve all chunks for a specific email.
        
        Args:
            email_id: Email message ID
            
        Returns:
            List of chunk dictionaries
        """
        try:
            with self.db_manager.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("""
                        SELECT id AS email_chunk_id, email_id, chunk_text, chunk_index, 
                               token_count, chunk_hash, created_at, updated_at
                        FROM email_chunks
                        WHERE email_id = %s
                        ORDER BY chunk_index
                    """, (email_id,))
                    
                    results = cur.fetchall()
                    
                    chunks = []
                    for row in results:
                        email_chunk_id, email_id, chunk_text, chunk_index, \
                        token_count, chunk_hash, created_at, updated_at = row
                        
                        chunks.append({
                            'email_chunk_id': email_chunk_id,
                            'email_id': email_id,
                            'chunk_text': chunk_text,
                            'chunk_index': chunk_index,
                            'token_count': token_count,
                            'chunk_hash': chunk_hash,
                            'created_at': created_at,
                            'updated_at': updated_at
                        })
                    
                    return chunks
                    
        except Exception as e:
            logger.error(f"Failed to retrieve chunks for email {email_id}: {e}")
            return []

    def get_chunk_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about stored email chunks.
        
        Returns:
            Dictionary with chunk statistics
        """
        try:
            with self.db_manager.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("""
                        SELECT 
                            COUNT(*) as total_chunks,
                            COUNT(DISTINCT email_id) as unique_emails,
                            AVG(token_count) as avg_token_count,
                            MIN(token_count) as min_token_count,
                            MAX(token_count) as max_token_count
                        FROM email_chunks
                    """)
                    
                    result = cur.fetchone()
                    if result:
                        total_chunks, unique_emails, avg_token_count, min_token_count, max_token_count = result
                        return {
                            'total_chunks': total_chunks or 0,
                            'unique_emails': unique_emails or 0,
                            'avg_token_count': float(avg_token_count) if avg_token_count else 0.0,
                            'min_token_count': min_token_count or 0,
                            'max_token_count': max_token_count or 0
                        }
                    else:
                        return {
                            'total_chunks': 0,
                            'unique_emails': 0,
                            'avg_token_count': 0.0,
                            'min_token_count': 0,
                            'max_token_count': 0
                        }
                        
        except Exception as e:
            logger.error(f"Failed to get chunk statistics: {e}")
            return {
                'total_chunks': 0,
                'unique_emails': 0,
                'avg_token_count': 0.0,
                'min_token_count': 0,
                'max_token_count': 0
            }

    def search_email_chunks_fts(self, query: str, k: int = 10) -> List[Document]:
        """
        Search email chunks using PostgreSQL FTS.
        
        Args:
            query: Search query
            k: Number of results to return
            
        Returns:
            List of Document objects with FTS scores
        """
        if not query.strip():
            logger.warning("Empty query provided to FTS search")
            return []
        
        try:
            logger.info(f"Performing PostgreSQL FTS search for: '{query[:50]}...' (k={k})")
            
            with self.db_manager.get_connection() as conn:
                with conn.cursor() as cur:
                    # Use PostgreSQL FTS with ranking
                    # ts_rank provides relevance scoring
                    cur.execute("""
                        SELECT 
                            ec.id AS email_chunk_id,
                            ec.email_id,
                            ec.chunk_text,
                            ec.chunk_index,
                            ec.token_count,
                            ts_rank(to_tsvector('english', ec.chunk_text), plainto_tsquery('english', %s)) as fts_score,
                            e.subject,
                            e.from_addr,
                            e.to_addrs,
                            e.date_utc,
                            e.message_id
                        FROM email_chunks ec
                        JOIN emails e ON ec.email_id = e.message_id
                        WHERE to_tsvector('english', ec.chunk_text) @@ plainto_tsquery('english', %s)
                        ORDER BY fts_score DESC
                        LIMIT %s
                    """, (query, query, k))
                    
                    results = cur.fetchall()
                    logger.info(f"PostgreSQL FTS returned {len(results)} results")
                    
                    documents = []
                    for row in results:
                        # Support both tuple rows and RealDictCursor dict rows.
                        if isinstance(row, dict):
                            email_chunk_id = row.get('email_chunk_id')
                            email_id = row.get('email_id')
                            chunk_text = row.get('chunk_text')
                            chunk_index = row.get('chunk_index')
                            token_count = row.get('token_count')
                            fts_score = row.get('fts_score')
                            subject = row.get('subject')
                            from_addr = row.get('from_addr')
                            to_addrs = row.get('to_addrs')
                            date_utc = row.get('date_utc')
                            message_id = row.get('message_id')
                        else:
                            email_chunk_id, email_id, chunk_text, chunk_index, token_count, fts_score, \
                            subject, from_addr, to_addrs, date_utc, message_id = row

                        # Ensure fts_score is present — fail fast if DB returned NULL unexpectedly
                        if fts_score is None:
                            # Log the problematic row for diagnostics before failing
                            logger.error(
                                "FTS returned NULL score for email_chunk_id=%s; row=%r",
                                email_chunk_id,
                                row,
                            )
                            raise RuntimeError(f"FTS returned NULL score for email_chunk_id={email_chunk_id}")
                        fts_score_value = float(fts_score)

                        # Create LangChain Document with metadata using the table-specific id key
                        doc = Document(
                            page_content=chunk_text,
                            metadata={
                                'email_chunk_id': email_chunk_id,
                                'email_id': email_id,
                                'message_id': message_id,
                                'chunk_index': chunk_index,
                                'token_count': token_count,
                                'fts_score': fts_score_value,
                                'subject': subject,
                                'from_addr': from_addr,
                                'to_addrs': to_addrs,  # JSONB field with recipient addresses
                                'date_utc': date_utc,
                                'category': 'email',
                                'category_type': 'email',
                                'retrieval_method': 'postgresql_fts'
                            }
                        )
                        documents.append(doc)
                    
                    return documents
                    
        except Exception as e:
            logger.error(f"PostgreSQL FTS search failed: {e}")
            return []

    # =============================================================================
    # Hybrid Retrieval Operations
    # =============================================================================
    
    def initialize_hybrid_retrieval(self, email_vector_store: Any) -> None:
        """
        Initialize retrieval system combining vector search and PostgreSQL FTS.
        
        Args:
            email_vector_store: Milvus email vector store from MilvusManager
        """
        try:
            # Import here to avoid circular dependencies
            from retrieval.email.postgres_fts_retriever import PostgresFTSRetriever
            from retrieval.email.processor import EmailProcessor
            
            # Initialize PostgreSQL FTS retriever
            self.postgres_fts_retriever = PostgresFTSRetriever(self.db_manager)
            
            # Initialize retriever combining vector + FTS
            self.hybrid_retriever = EmailProcessor(
                vector_retriever=email_vector_store.as_retriever(),
                fts_retriever=self.postgres_fts_retriever
            )
            
            logger.info("Email retrieval system initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize email retrieval: {e}")
            raise RuntimeError(f"Email retrieval initialization failed: {e}")

    def search_emails_hybrid(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Hybrid email search combining vector similarity and PostgreSQL FTS.
        
        Args:
            query: Search query
            top_k: Maximum number of results to return
            
        Returns:
            List of email chunks with relevance scores and metadata
        """
        if not self.hybrid_retriever:
            raise RuntimeError("Hybrid retriever not initialized. Call initialize_hybrid_retrieval() first.")
        
        try:
            # Perform search using RRF fusion
            results = self.hybrid_retriever.search(query, k=top_k)
            
            # Convert to consistent format
            formatted_results = []
            for result in results:
                formatted_results.append({
                    'chunk_text': result.page_content,
                    'metadata': result.metadata,
                    'similarity_score': result.metadata.get('combined_score', 0.0)
                })
            
            logger.info(f"Hybrid email search returned {len(formatted_results)} results for query: {query[:50]}...")
            return formatted_results
            
        except Exception as e:
            logger.error(f"Hybrid email search failed: {e}")
            raise RuntimeError(f"Email search failed: {e}")

    def format_email_context(self, results: List[Dict[str, Any]]) -> tuple:
        """
        Format search results for LLM context.
        
        Args:
            results: List of email search results
            
        Returns:
            Tuple of (context_text, sources) for LLM processing
        """
        unique_emails = {}
        sources = []
        
        # Group chunks by email_id to create complete email context
        for result in results:
            metadata = result['metadata']
            chunk_text = result['chunk_text']
            email_id = metadata.get('message_id')
            
            if not email_id:
                logger.error(f"Email chunk missing required 'message_id' metadata: {metadata}")
                continue
            
            if email_id not in unique_emails:
                unique_emails[email_id] = {
                    'ref_num': len(unique_emails) + 1,
                    'subject': metadata.get('subject', metadata.get('topic', '')),
                    'sender': metadata.get('from_addr', ''),
                    'recipient': metadata.get('to_addrs', ''),
                    'date': metadata.get('date_utc', metadata.get('date', '')),
                    'chunks': []
                }
            
            unique_emails[email_id]['chunks'].append(chunk_text)
        
        # Build context for LLM
        context_parts = []
        for email_id, email_data in unique_emails.items():
            ref_num = email_data['ref_num']
            
            # Combine all chunks for this email
            full_content = '\n'.join(email_data['chunks'])
            
            context_part = f"""Email [{ref_num}]:
Subject: {email_data['subject']}
From: {email_data['sender']}
To: {email_data['recipient']}
Date: {email_data['date']}

Content:
{full_content}

Email ID: {email_id}"""
            
            context_parts.append(context_part)
            
            # Add to sources for display
            sources.append({
                'filename': f"Email: {email_data['subject']}",
                'category_type': 'email',
                'email_subject': email_data['subject'],
                'email_sender': email_data['sender'],
                'email_recipient': email_data['recipient'],
                'email_date': email_data['date'],
                'email_id': email_id,
                'ref_num': ref_num,
                'page': 'N/A',
                'similarity_score': 1.0  # Default score for emails
            })
        
        context_text = "\n\n".join(context_parts)
        return context_text, sources
