#!/usr/bin/env python
"""
PostgreSQL Email Account Manager
Following DEVELOPMENT_RULES.md for all development requirements

This module provides PostgreSQL-based email account management functionality
to replace SQLite-based email operations when using PostgreSQL backend.
Also handles hybrid email search combining vector and FTS retrieval.
"""

import json
import logging
from datetime import datetime, UTC
from typing import Any, Dict, List, Optional

from ingestion.utils.db_utils import PostgreSQLManager
from ingestion.utils.crypto import encrypt, decrypt

logger = logging.getLogger(__name__)


class PostgreSQLEmailManager:
    """PostgreSQL-based email account manager with hybrid search capabilities."""

    def __init__(self, postgres_manager: Any) -> None:
        """Initialize with PostgreSQL manager for pure PostgreSQL-based email account management."""
        self.db_manager = PostgreSQLManager(postgres_manager.pool)
        self.postgres_pool = postgres_manager.pool
        # Add pool attribute for compatibility with EmailProcessor
        self.pool = postgres_manager.pool
        
        # Hybrid retrieval components (initialized on demand)
        self.postgres_fts_retriever: Optional[Any] = None
        self.hybrid_retriever: Optional[Any] = None
        
        logger.info("PostgreSQL EmailManager initialized with PostgreSQL-based email statistics")

    def create_account(self, record: Dict[str, Any]) -> int:
        """Create a new email account (PostgreSQL implementation)."""
        required = {
            "account_name",
            "server_type", 
            "server",
            "port",
            "email_address",
            "password",
        }
        missing = required - record.keys()
        if missing:
            raise ValueError(f"record missing required fields: {', '.join(sorted(missing))}")

        # Encrypt the password before persisting
        record = dict(record)
        if "password" in record:
            record["password"] = encrypt(str(record["password"]))

        logger.info(
            "Creating email account '%s' (%s) on %s:%s for %s",
            record.get("account_name"),
            record.get("server_type"),
            record.get("server"),
            record.get("port"),
            record.get("email_address"),
        )

        try:
            with self.db_manager.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("""
                        INSERT INTO email_accounts (
                            account_name, server_type, server, port, email_address, password,
                            mailbox, batch_limit, use_ssl, refresh_interval_minutes
                        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                        RETURNING id
                    """, (
                        record.get("account_name"),
                        record.get("server_type"),
                        record.get("server"),
                        record.get("port"),
                        record.get("email_address"),
                        record.get("password"),
                        record.get("mailbox", "INBOX"),
                        record.get("batch_limit", 50),
                        record.get("use_ssl", 1),
                        record.get("refresh_interval_minutes", 60)
                    ))
                    account_id = cur.fetchone()['id']
                conn.commit()
                logger.info("Created email account %s (%s)", account_id, record.get("account_name"))
                return account_id
        except Exception as e:
            logger.error("Failed to create email account: %s", e)
            raise

    def list_accounts(self, include_password: bool = False) -> List[Dict[str, Any]]:
        """List all email accounts (PostgreSQL implementation) with email statistics from PostgreSQL."""
        with self.db_manager.get_connection() as conn:
            with conn.cursor() as cur:
                if include_password:
                    accounts_query = """
                        SELECT ea.id, ea.account_name, ea.server_type, ea.server, ea.port, ea.email_address, ea.password,
                               ea.mailbox, ea.batch_limit, ea.use_ssl, ea.refresh_interval_minutes, 
                               ea.last_synced::text as last_synced, ea.last_update_status,
                               COALESCE(ea.last_synced_offset, 0) as last_synced_offset,
                               CASE
                                   WHEN ea.refresh_interval_minutes IS NOT NULL AND ea.refresh_interval_minutes > 0 AND ea.last_synced IS NOT NULL
                                   THEN (ea.last_synced + INTERVAL '1 minute' * ea.refresh_interval_minutes)::text
                                   ELSE NULL
                               END AS next_run,
                               COALESCE(ea.total_emails_in_mailbox, 0) as total_emails,
                               COALESCE(email_stats.synced_emails, 0) as synced_emails,
                               COALESCE(email_stats.total_chunks, 0) as total_chunks
                        FROM email_accounts ea
                        LEFT JOIN (
                            SELECT 
                                from_addr,
                                COUNT(DISTINCT message_id) as synced_emails,  -- Count unique emails, not chunks
                                COUNT(*) as total_chunks  -- This counts chunks for chunk statistics
                            FROM emails 
                            GROUP BY from_addr
                        ) email_stats ON ea.email_address = email_stats.from_addr
                        ORDER BY ea.account_name
                    """
                else:
                    accounts_query = """
                        SELECT ea.id, ea.account_name, ea.server_type, ea.server, ea.port, ea.email_address,
                               ea.mailbox, ea.batch_limit, ea.use_ssl, ea.refresh_interval_minutes, 
                               ea.last_synced::text as last_synced, ea.last_update_status,
                               COALESCE(ea.last_synced_offset, 0) as last_synced_offset,
                               CASE
                                   WHEN ea.refresh_interval_minutes IS NOT NULL AND ea.refresh_interval_minutes > 0 AND ea.last_synced IS NOT NULL
                                   THEN (ea.last_synced + INTERVAL '1 minute' * ea.refresh_interval_minutes)::text
                                   ELSE NULL
                               END AS next_run,
                               COALESCE(ea.total_emails_in_mailbox, 0) as total_emails,
                               COALESCE(email_stats.synced_emails, 0) as synced_emails,
                               COALESCE(email_stats.total_chunks, 0) as total_chunks
                        FROM email_accounts ea
                        LEFT JOIN (
                            SELECT 
                                from_addr,
                                COUNT(DISTINCT message_id) as synced_emails,  -- Count unique emails, not chunks
                                COUNT(*) as total_chunks  -- This counts chunks for chunk statistics
                            FROM emails 
                            GROUP BY from_addr
                        ) email_stats ON ea.email_address = email_stats.from_addr
                        ORDER BY ea.account_name
                    """
                cur.execute(accounts_query)
                accounts: List[Dict[str, Any]] = []
                for row in cur.fetchall():
                    account = dict(row)
                    if include_password and "password" in account:
                        try:
                            account["password"] = decrypt(account["password"])
                        except Exception:
                            account["password"] = ""
                    accounts.append(account)
                logger.info("Retrieved %d email accounts with PostgreSQL-based statistics", len(accounts))
                return accounts

    def update_account(self, account_id: int, updates: Dict[str, Any]) -> None:
        """Update an email account (PostgreSQL implementation)."""
        if not updates:
            return
        
        try:
            with self.db_manager.get_connection() as conn:
                with conn.cursor() as cur:
                    # Build dynamic update query
                    set_clauses = []
                    values = []
                    for key, value in updates.items():
                        set_clauses.append(f"{key} = %s")
                        values.append(value)
                    
                    values.append(account_id)
                    query = f"UPDATE email_accounts SET {', '.join(set_clauses)} WHERE id = %s"
                    cur.execute(query, values)
                conn.commit()
                logger.info(f"Updated email account {account_id}")
        except Exception as e:
            logger.error(f"Failed to update email account {account_id}: {e}")
            raise

    def delete_account(self, account_id: int) -> None:
        """Delete an email account (PostgreSQL implementation)."""
        try:
            with self.db_manager.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("DELETE FROM email_accounts WHERE id = %s", (account_id,))
                conn.commit()
                logger.info(f"Deleted email account {account_id}")
        except Exception as e:
            logger.error(f"Failed to delete email account {account_id}: {e}")
            raise

    def get_account_count(self) -> int:
        """Get the total number of email accounts (PostgreSQL implementation)."""
        logger.debug("get_account_count called.")
        try:
            with self.db_manager.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("SELECT COUNT(*) as count FROM email_accounts")
                    result = cur.fetchone()
                    logger.debug("Executing query to count email accounts.")
                    logger.debug("Query result: %s", result)
                    return result['count'] if result else 0
        except Exception as e:
            logger.error(f"Failed to get email account count: {e}")
            return 0

    def get_account_stats(self) -> Dict[str, Any]:
        """Get email account statistics (PostgreSQL implementation)."""
        try:
            with self.db_manager.get_connection() as conn:
                with conn.cursor() as cur:
                    # Basic account stats
                    cur.execute("""
                        SELECT 
                            COUNT(*) as total_accounts,
                            COUNT(*) FILTER (WHERE last_synced IS NOT NULL) as synced_accounts,
                            COUNT(*) FILTER (WHERE refresh_interval_minutes > 0) as active_accounts
                        FROM email_accounts
                    """)
                    account_stats = cur.fetchone()
                    
                    # Email stats from emails table
                    cur.execute("""
                        SELECT 
                            COUNT(*) as total_messages,
                            COUNT(DISTINCT from_addr) as unique_senders
                        FROM emails
                    """)
                    email_stats = cur.fetchone()
                    
                    return {
                        'total_accounts': account_stats['total_accounts'] if account_stats else 0,
                        'synced_accounts': account_stats['synced_accounts'] if account_stats else 0,
                        'active_accounts': account_stats['active_accounts'] if account_stats else 0,
                        'total_messages': email_stats['total_messages'] if email_stats else 0,
                        'processed_messages': email_stats['total_messages'] if email_stats else 0,  # All stored = processed
                        'unique_senders': email_stats['unique_senders'] if email_stats else 0
                    }
        except Exception as e:
            logger.error(f"Failed to get email account stats: {e}")
            return {
                'total_accounts': 0,
                'synced_accounts': 0,
                'active_accounts': 0,
                'total_messages': 0,
                'processed_messages': 0,
                'unique_senders': 0
            }

    def get_email_statistics_for_account(self, email_address: str) -> Dict[str, int]:
        """Get email statistics for a specific account from PostgreSQL."""
        try:
            with self.db_manager.get_connection() as conn:
                with conn.cursor() as cur:
                    # Count emails for this account
                    cur.execute("""
                        SELECT 
                            COUNT(*) as total_emails,
                            COUNT(*) as total_chunks  -- For now, 1 email = 1 chunk
                        FROM emails 
                        WHERE from_addr = %s
                    """, (email_address,))
                    
                    result = cur.fetchone()
                    if result:
                        return {
                            'total_emails': result['total_emails'],
                            'synced_emails': result['total_emails'],  # All stored emails are synced
                            'total_chunks': result['total_chunks']
                        }
                    else:
                        return {
                            'total_emails': 0,
                            'synced_emails': 0,
                            'total_chunks': 0
                        }
        except Exception as e:
            logger.error(f"Failed to get email statistics for {email_address}: {e}")
            return {
                'total_emails': 0,
                'synced_emails': 0,
                'total_chunks': 0
            }

    def initialize_hybrid_retrieval(self, email_vector_store: Any) -> None:
        """
        Initialize hybrid retrieval system combining vector search and PostgreSQL FTS.
        
        Args:
            email_vector_store: Milvus email vector store from MilvusManager
        """
        try:
            # Import here to avoid circular dependencies
            from retrieval.email.postgres_fts_retriever import PostgresFTSRetriever
            from retrieval.email.hybrid_retriever import HybridRetriever
            
            # Initialize PostgreSQL FTS retriever
            self.postgres_fts_retriever = PostgresFTSRetriever(self.postgres_pool)
            
            # Initialize hybrid retriever combining vector + FTS
            self.hybrid_retriever = HybridRetriever(
                vector_retriever=email_vector_store.as_retriever(),
                fts_retriever=self.postgres_fts_retriever
            )
            
            logger.info("Email hybrid retrieval system initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize email hybrid retrieval: {e}")
            raise RuntimeError(f"Email hybrid retrieval initialization failed: {e}")

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
            # Perform hybrid search using RRF fusion
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
        """Format hybrid search results for LLM context."""
        unique_emails = {}
        sources = []
        
        # Group chunks by email_id to create complete email context
        for result in results:
            metadata = result['metadata']
            chunk_text = result['chunk_text']
            email_id = metadata.get('message_id', metadata.get('source', 'unknown'))
            chunk_id = metadata.get('chunk_id')  # Get the real chunk_id from metadata
            
            if email_id not in unique_emails:
                unique_emails[email_id] = {
                    'ref_num': len(unique_emails) + 1,
                    'subject': metadata.get('subject', metadata.get('topic', '')),
                    'sender': metadata.get('from_addr', metadata.get('source', '')),
                    'recipient': metadata.get('to_addrs', ''),
                    'date': metadata.get('date_utc', metadata.get('date', '')),
                    'chunks': [],
                    'chunk_ids': []  # Track chunk_ids for each email
                }
            
            unique_emails[email_id]['chunks'].append(chunk_text)
            if chunk_id:
                unique_emails[email_id]['chunk_ids'].append(chunk_id)
        
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
                'chunk_id': ','.join(email_data.get('chunk_ids', [])) or f"email_{ref_num}",  # Use real chunk_ids or fallback
                'similarity_score': 1.0  # Default score for emails
            })
        
        context_text = "\n\n".join(context_parts)
        return context_text, sources

    def upsert_email(self, record: Dict[str, Any]) -> None:
        """
        Upsert an email record into PostgreSQL database.
        
        Args:
            record: Email record dictionary with required fields
        """
        required_fields = ["message_id", "subject", "body_text"]
        missing_fields = [field for field in required_fields if not record.get(field)]
        
        if missing_fields:
            raise ValueError(f"Email record missing required fields: {missing_fields}")
        
        message_id = record["message_id"]
        logger.info(f"Upserting email record for message_id: {message_id}")
        
        try:
            with self.db_manager.get_connection() as conn:
                with conn.cursor() as cur:
                    # Convert to_addrs to JSON if it's a list or string
                    to_addrs = record.get("to_addrs", [])
                    if isinstance(to_addrs, str):
                        to_addrs = [to_addrs]
                    
                    # Upsert email record using correct schema
                    cur.execute("""
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
                    """, (
                        message_id,
                        record.get("from_addr", ""),
                        json.dumps(to_addrs) if to_addrs else None,
                        record.get("subject", ""),
                        record.get("date_utc"),
                        record.get("header_hash", ""),
                        record.get("content_hash", ""),
                        record.get("body_text", ""),
                        json.dumps(record.get("attachments", [])) if record.get("attachments") else None,
                        json.dumps(record.get("headers", {})) if record.get("headers") else None
                    ))
                conn.commit()
                logger.info(f"Successfully upserted email record: {message_id}")
                
        except Exception as e:
            logger.error(f"Failed to upsert email record {message_id}: {e}")
            raise

    def update_total_emails_in_mailbox(self, account_id: int, total_emails: int) -> None:
        """
        Update the total number of emails in the mailbox for an account.
        
        Args:
            account_id: Email account ID
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
