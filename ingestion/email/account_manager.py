#!/usr/bin/env python
"""
Email Account Manager
Following DEVELOPMENT_RULES.md for all development requirements

This module provides email account management functionality
for creating, updating, and managing email accounts.
Delegates email data operations to the shared EmailDataManager.
"""

import logging
from datetime import datetime, UTC
from typing import Any, Dict, List, Optional

from rag_manager.managers.postgres_manager import PostgreSQLManager
from rag_manager.data.email_data import EmailDataManager
from ingestion.utils.crypto import encrypt, decrypt

logger = logging.getLogger(__name__)


class EmailAccountManager:
    """Email account manager with search capabilities."""

    def __init__(self, postgres_manager: Any) -> None:
        """Initialize with PostgreSQL manager for pure PostgreSQL-based email account management."""
        self.db_manager = PostgreSQLManager()
        
        # Initialize email data manager for database operations
        self.email_data_manager = EmailDataManager(postgres_manager)
        
        # Hybrid retrieval components (initialized on demand)
        self.postgres_fts_retriever: Optional[Any] = None
        self.hybrid_retriever: Optional[Any] = None
        
        logger.info("Email Account Manager initialized with shared EmailDataManager")

    def create_account(self, record: Dict[str, Any]) -> str:
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
                account_id = str(account_id)
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
                        SELECT ea.id AS email_account_id, ea.account_name, ea.server_type, ea.server, ea.port, ea.email_address, ea.password,
                               ea.mailbox, ea.batch_limit, ea.use_ssl, ea.refresh_interval_minutes, 
                               ea.last_synced::text as last_synced, ea.last_update_status,
                               COALESCE(ea.last_synced_offset, 0) as last_synced_offset,
                               CASE
                                   WHEN ea.refresh_interval_minutes IS NOT NULL AND ea.refresh_interval_minutes > 0 AND ea.last_synced IS NOT NULL
                                   THEN (ea.last_synced + INTERVAL '1 minute' * ea.refresh_interval_minutes)::text
                                   ELSE NULL
                               END AS next_run,
                               COALESCE(ea.total_emails_in_mailbox, 0) as total_emails,
                               COALESCE(global_stats.synced_emails, 0) as synced_emails,
                               COALESCE(global_stats.total_chunks, 0) as total_chunks
                        FROM email_accounts ea
                        CROSS JOIN (
                            SELECT 
                                COUNT(DISTINCT message_id) as synced_emails,
                                COUNT(DISTINCT ec.id) as total_chunks
                            FROM emails e
                            LEFT JOIN email_chunks ec ON e.id = ec.email_id
                        ) global_stats
                        ORDER BY ea.account_name
                    """
                else:
                    accounts_query = """
                        SELECT ea.id AS email_account_id, ea.account_name, ea.server_type, ea.server, ea.port, ea.email_address,
                               ea.mailbox, ea.batch_limit, ea.use_ssl, ea.refresh_interval_minutes, 
                               ea.last_synced::text as last_synced, ea.last_update_status,
                               COALESCE(ea.last_synced_offset, 0) as last_synced_offset,
                               CASE
                                   WHEN ea.refresh_interval_minutes IS NOT NULL AND ea.refresh_interval_minutes > 0 AND ea.last_synced IS NOT NULL
                                   THEN (ea.last_synced + INTERVAL '1 minute' * ea.refresh_interval_minutes)::text
                                   ELSE NULL
                               END AS next_run,
                               COALESCE(ea.total_emails_in_mailbox, 0) as total_emails,
                               COALESCE(global_stats.synced_emails, 0) as synced_emails,
                               COALESCE(global_stats.total_chunks, 0) as total_chunks
                        FROM email_accounts ea
                        CROSS JOIN (
                            SELECT 
                                COUNT(DISTINCT message_id) as synced_emails,
                                COUNT(DISTINCT ec.id) as total_chunks
                            FROM emails e
                            LEFT JOIN email_chunks ec ON e.id = ec.email_id
                        ) global_stats
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

    def update_account(self, account_id: str, updates: Dict[str, Any]) -> None:
        """Update an email account."""
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

    def delete_account(self, account_id: str) -> None:
        """Delete an email account."""
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
        """Get the total number of email accounts."""
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
        """Get email account statistics."""
        try:
            with self.db_manager.get_connection() as conn:
                with conn.cursor() as cur:
                    # Basic account stats
                    cur.execute("""
                        SELECT 
                            COUNT(*) as total_accounts,
                            COUNT(*) FILTER (WHERE last_synced IS NOT NULL) as synced_accounts,
                            COUNT(*) FILTER (WHERE refresh_interval_minutes > 0) as active_accounts,
                            SUM(COALESCE(total_emails_in_mailbox, 0)) as total_emails_in_all_mailboxes
                        FROM email_accounts
                    """)
                    account_stats = cur.fetchone()
                    
                    # Email stats from emails table
                    cur.execute("""
                        SELECT 
                            COUNT(DISTINCT message_id) as processed_messages
                        FROM emails
                    """)
                    email_stats = cur.fetchone()
                    
                    return {
                        'total_accounts': account_stats['total_accounts'] if account_stats else 0,
                        'synced_accounts': account_stats['synced_accounts'] if account_stats else 0,
                        'active_accounts': account_stats['active_accounts'] if account_stats else 0,
                        'total_messages': account_stats['total_emails_in_all_mailboxes'] if account_stats else 0,
                        'processed_messages': email_stats['processed_messages'] if email_stats else 0,
                    }
        except Exception as e:
            logger.error(f"Failed to get email account stats: {e}")
            return {
                'total_accounts': 0,
                'synced_accounts': 0,
                'active_accounts': 0,
                'total_messages': 0,
                'processed_messages': 0,
            }


