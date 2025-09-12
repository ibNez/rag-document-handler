#!/usr/bin/env python
"""
Email Data Manager
Following DEVELOPMENT_RULES.md for all development requirements

This module provides pure data access operations for email content.
No business logic - only database operations.
"""

import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
from langchain_core.documents import Document

from .base_data import BaseDataManager

logger = logging.getLogger(__name__)


class EmailDataManager(BaseDataManager):
    """
    Pure data access manager for email operations.
    
    Handles all PostgreSQL and Milvus operations for email content
    without any business logic or orchestration.
    """
    
    def __init__(self, postgres_manager: Any, milvus_manager: Optional[Any] = None) -> None:
        """
        Initialize email data manager.
        
        Args:
            postgres_manager: PostgreSQL connection manager
            milvus_manager: Optional Milvus vector store manager
        """
        super().__init__(postgres_manager, milvus_manager)
        logger.info("EmailDataManager initialized for pure data operations")
    
    # =============================================================================
    # Email Record Operations
    # =============================================================================
    
    def upsert_email(self, record: Dict[str, Any]) -> str:
        """
        Insert or update email record in PostgreSQL.
        
        Args:
            record: Email record dictionary with required fields
            
        Returns:
            The UUID of the upserted email record
        """
        required_fields = ["message_id"]  # Only message_id is truly required
        missing_fields = [field for field in required_fields if field not in record or record.get(field) is None]
        
        if missing_fields:
            raise ValueError(f"Email record missing required fields: {missing_fields}")
        
        message_id = record["message_id"]
        logger.debug(f"Upserting email record for message_id: {message_id}")
        
        try:
            # Convert to_addrs to JSON if it's a list or string
            to_addrs = record.get("to_addrs", [])
            if isinstance(to_addrs, str):
                to_addrs = [to_addrs]
            
            query = """
                INSERT INTO emails (
                    message_id, from_addr, to_addrs, subject, date_utc, 
                    header_hash, content_hash, content, attachments, headers
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (message_id) 
                DO UPDATE SET
                    from_addr = EXCLUDED.from_addr,
                    to_addrs = EXCLUDED.to_addrs,
                    subject = EXCLUDED.subject,
                    date_utc = EXCLUDED.date_utc,
                    header_hash = EXCLUDED.header_hash,
                    content_hash = EXCLUDED.content_hash,
                    content = EXCLUDED.content,
                    attachments = EXCLUDED.attachments,
                    headers = EXCLUDED.headers,
                    updated_at = CURRENT_TIMESTAMP
                RETURNING id
            """
            
            params = (
                message_id,
                record.get("from_addr", ""),
                json.dumps(to_addrs) if to_addrs else None,
                record.get("subject", ""),
                record.get("date_utc"),
                record.get("header_hash", ""),
                record.get("content_hash", ""),
                record.get("content", ""),  # Use 'content' field instead of 'body_text'
                json.dumps(record.get("attachments", [])) if record.get("attachments") else None,
                json.dumps(record.get("headers", {})) if record.get("headers") else None
            )
            
            result = self.execute_query(query, params, fetch_one=True)
            email_id = str(result['id']) if result else None
            
            if not email_id:
                raise RuntimeError(f"Failed to get email ID for message {message_id}")
                
            logger.debug(f"Successfully upserted email record: {message_id} with ID: {email_id}")
            return email_id
            
        except Exception as e:
            logger.error(f"Failed to upsert email record {message_id}: {e}")
            raise
     
    def delete_email(self, message_id: str) -> bool:
        """
        Delete email record by message ID.
        
        Args:
            message_id: Email message ID
            
        Returns:
            True if deleted, False if not found
        """
        try:
            query = "DELETE FROM emails WHERE message_id = %s"
            with self.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(query, (message_id,))
                    deleted = cur.rowcount > 0
                conn.commit()
            
            if deleted:
                logger.debug(f"Deleted email record: {message_id}")
            else:
                logger.warning(f"Email record not found for deletion: {message_id}")
            
            return deleted
            
        except Exception as e:
            logger.error(f"Failed to delete email record {message_id}: {e}")
            raise
    
    # =============================================================================
    # Email Statistics Operations
    # =============================================================================
        
    def get_global_email_statistics(self) -> Dict[str, int]:
        """
        Get global email statistics across all accounts.
        
        Returns:
            Dictionary with global email statistics
        """
        try:
            query = """
                SELECT 
                    COUNT(DISTINCT message_id) as total_emails,
                    COUNT(DISTINCT from_addr) as unique_senders,
                    COUNT(*) as total_chunks,
                    COUNT(*) FILTER (WHERE created_at >= CURRENT_DATE) as today_emails,
                    COUNT(*) FILTER (WHERE created_at >= CURRENT_DATE - INTERVAL '7 days') as week_emails,
                    COUNT(*) FILTER (WHERE created_at >= CURRENT_DATE - INTERVAL '30 days') as month_emails
                FROM emails
            """
            
            result = self.execute_query(query, fetch_one=True)
            
            if result:
                return dict(result)
            else:
                return {
                    'total_emails': 0,
                    'unique_senders': 0,
                    'total_chunks': 0,
                    'today_emails': 0,
                    'week_emails': 0,
                    'month_emails': 0
                }
                
        except Exception as e:
            logger.error(f"Failed to get global email statistics: {e}")
            raise
    
    # =============================================================================
    # Email Account Operations
    # =============================================================================
    
    def update_total_emails_in_mailbox(self, account_id: str, total_emails: int) -> None:
        """
        Update the total number of emails in the mailbox for an account.
        
        Args:
            account_id: Email account ID (UUID string)
            total_emails: Total number of emails found in the mailbox
        """
        try:
            query = """
                UPDATE email_accounts 
                SET total_emails_in_mailbox = %s
                WHERE id = %s
            """
            self.execute_query(query, (total_emails, account_id))
            logger.debug(f"Updated total emails count for account {account_id}: {total_emails}")
            
        except Exception as e:
            logger.error(f"Failed to update total emails count for account {account_id}: {e}")
            raise

    # =============================================================================
    # Email Search Operations  
    # =============================================================================
    
    def search_email_chunks_fts(self, query: str, k: int = 10) -> List[Document]:
        """
        Search email chunks using PostgreSQL Full Text Search.
        
        Args:
            query: Search query string
            k: Maximum number of results to return
            
        Returns:
            List of Document objects with search results
        """
        try:
            # Use PostgreSQL FTS with ranking
            search_query = """
                SELECT 
                    e.message_id,
                    e.subject,
                    e.from_addr,
                    e.to_addrs,
                    e.date_utc,
                    e.content,
                    ts_rank_cd(to_tsvector('english', e.content), plainto_tsquery('english', %s)) as rank
                FROM emails e
                WHERE to_tsvector('english', e.content) @@ plainto_tsquery('english', %s)
                ORDER BY rank DESC
                LIMIT %s
            """
            
            results = self.execute_query(search_query, (query, query, k), fetch_all=True)
            
            documents = []
            if results:
                for row in results:
                    # Create Document with email metadata
                    doc = Document(
                        page_content=row['content'],
                        metadata={
                            'message_id': row['message_id'],
                            'subject': row['subject'],
                            'from_addr': row['from_addr'],
                            'to_addrs': row['to_addrs'],
                            'date_utc': row['date_utc'].isoformat() if row['date_utc'] else None,
                            'fts_rank': float(row['rank']),
                            'email_chunk_id': row['message_id'],  # For RRF fusion compatibility
                            'content_type': 'email'
                        }
                    )
                    documents.append(doc)
            
            logger.debug(f"FTS search returned {len(documents)} results for query: {query[:50]}...")
            return documents
            
        except Exception as e:
            logger.error(f"FTS search failed for query '{query}': {e}")
            raise
    
    
    def get_chunk_statistics(self) -> Dict[str, Any]:
        """
        Get email chunk statistics.
        
        Returns:
            Dictionary with chunk statistics
        """
        try:
            query = """
                SELECT 
                    COUNT(*) as total_chunks,
                    COUNT(DISTINCT message_id) as unique_emails,
                    AVG(LENGTH(content)) as avg_chunk_length,
                    MIN(LENGTH(content)) as min_chunk_length,
                    MAX(LENGTH(content)) as max_chunk_length
                FROM emails
            """
            
            result = self.execute_query(query, fetch_one=True)
            
            if result:
                return {
                    'total_chunks': result['total_chunks'],
                    'unique_emails': result['unique_emails'],
                    'avg_chunk_length': float(result['avg_chunk_length']) if result['avg_chunk_length'] else 0.0,
                    'min_chunk_length': result['min_chunk_length'] or 0,
                    'max_chunk_length': result['max_chunk_length'] or 0
                }
            else:
                return {
                    'total_chunks': 0,
                    'unique_emails': 0,
                    'avg_chunk_length': 0.0,
                    'min_chunk_length': 0,
                    'max_chunk_length': 0
                }
                
        except Exception as e:
            logger.error(f"Failed to get chunk statistics: {e}")
            raise
    
    # =============================================================================
    # Email Search and Retrieval Operations
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
            self.postgres_fts_retriever = PostgresFTSRetriever(self.postgres_manager)
            
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
        if not hasattr(self, 'hybrid_retriever') or not self.hybrid_retriever:
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
    
