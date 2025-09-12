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
