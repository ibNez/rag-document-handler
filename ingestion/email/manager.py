#!/usr/bin/env python
"""
PostgreSQL Email Account Manager
Following DEVELOPMENT_RULES.md for all development requirements

This module provides PostgreSQL-based email account management functionality
to replace SQLite-based email operations when using PostgreSQL backend.
"""

import logging
from datetime import datetime, UTC
from typing import Any, Dict, List, Optional

from ingestion.utils.db_utils import PostgreSQLManager
from ingestion.utils.crypto import encrypt, decrypt

logger = logging.getLogger(__name__)


class PostgreSQLEmailManager:
    """PostgreSQL-based email account manager compatible with EmailAccountManager interface."""

    def __init__(self, postgres_manager: Any) -> None:
        """Initialize with PostgreSQL manager."""
        self.db_manager = PostgreSQLManager(postgres_manager.pool)
        logger.info("PostgreSQL EmailManager initialized")

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
        """List all email accounts (PostgreSQL implementation)."""
        try:
            with self.db_manager.get_connection() as conn:
                with conn.cursor() as cur:
                    if include_password:
                        cur.execute("""
                            SELECT id, account_name, server_type, server, port, email_address, password,
                                   mailbox, batch_limit, use_ssl, refresh_interval_minutes, 
                                   last_synced::text as last_synced, last_update_status,
                                   CASE
                                       WHEN refresh_interval_minutes IS NOT NULL AND refresh_interval_minutes > 0 AND last_synced IS NOT NULL
                                       THEN (last_synced + INTERVAL '1 minute' * refresh_interval_minutes)::text
                                       ELSE NULL
                                   END AS next_run
                            FROM email_accounts ORDER BY account_name
                        """)
                    else:
                        cur.execute("""
                            SELECT id, account_name, server_type, server, port, email_address,
                                   mailbox, batch_limit, use_ssl, refresh_interval_minutes, 
                                   last_synced::text as last_synced, last_update_status,
                                   CASE
                                       WHEN refresh_interval_minutes IS NOT NULL AND refresh_interval_minutes > 0 AND last_synced IS NOT NULL
                                       THEN (last_synced + INTERVAL '1 minute' * refresh_interval_minutes)::text
                                       ELSE NULL
                                   END AS next_run
                            FROM email_accounts ORDER BY account_name
                        """)
                    
                    accounts = []
                    for row in cur.fetchall():
                        # Convert RealDictRow to regular dict
                        account = dict(row)
                        
                        # Handle password decryption if needed
                        if include_password and "password" in account:
                            try:
                                account["password"] = decrypt(account["password"])
                            except Exception:
                                account["password"] = ""
                        
                        accounts.append(account)
                    
                    logger.info(
                        "Retrieved %d email accounts",
                        len(accounts),
                    )
                    return accounts
        except Exception as e:
            logger.error(f"Failed to list email accounts: {e}")
            return []

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
                    
                    return {
                        'total_accounts': account_stats['total_accounts'] if account_stats else 0,
                        'synced_accounts': account_stats['synced_accounts'] if account_stats else 0,
                        'active_accounts': account_stats['active_accounts'] if account_stats else 0,
                        'total_messages': 0,  # Would need email_messages table for this
                        'processed_messages': 0
                    }
        except Exception as e:
            logger.error(f"Failed to get email account stats: {e}")
            return {
                'total_accounts': 0,
                'synced_accounts': 0,
                'active_accounts': 0,
                'total_messages': 0,
                'processed_messages': 0
            }
