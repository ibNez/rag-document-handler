"""PostgreSQL-native email message manager for RAG system.

This module provides PostgreSQL-based email message storage and retrieval,
completely separate from SQLite dependencies.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


def _mask(value: Optional[str]) -> str:
    """Mask sensitive values for logging."""
    if not value or len(value) <= 4:
        return "***"
    return value[:2] + "*" * (len(value) - 4) + value[-2:]


def compute_header_hash(record: Dict[str, Any]) -> str:
    """Compute a hash from email headers for deduplication.
    
    Args:
        record: Email record dictionary
        
    Returns:
        String hash of key email headers
    """
    import hashlib
    
    # Use key headers that uniquely identify an email - using email protocol names
    key_headers = [
        record.get("message_id", ""),
        record.get("from_addr", ""),
        record.get("subject", ""),
        str(record.get("date_utc", "")),
    ]
    
    combined = "|".join(key_headers)
    return hashlib.sha256(combined.encode()).hexdigest()


class PostgreSQLEmailManager:
    """PostgreSQL-native CRUD helper for email messages."""

    def __init__(self, postgresql_manager: Any) -> None:
        """Initialize with PostgreSQL manager.
        
        Args:
            postgresql_manager: PostgreSQL manager instance
        """
        self.postgresql_manager = postgresql_manager
        self._ensure_table()

    def _ensure_table(self) -> None:
        """Create the emails table if it does not exist."""
        try:
            with self.postgresql_manager.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        CREATE TABLE IF NOT EXISTS emails (
                            id SERIAL PRIMARY KEY,
                            message_id TEXT UNIQUE NOT NULL,
                            from_addr TEXT,
                            subject TEXT,
                            date_utc TIMESTAMP,
                            header_hash TEXT UNIQUE NOT NULL,
                            content_hash TEXT,
                            content TEXT,
                            attachments JSONB,
                            headers JSONB,
                            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                        )
                        """
                    )
                    conn.commit()
                    logger.info("PostgreSQL emails table created successfully")
                    
                    # Create indexes for performance
                    cur.execute("CREATE INDEX IF NOT EXISTS idx_emails_message_id ON emails(message_id)")
                    cur.execute("CREATE INDEX IF NOT EXISTS idx_emails_header_hash ON emails(header_hash)")
                    cur.execute("CREATE INDEX IF NOT EXISTS idx_emails_from_addr ON emails(from_addr)")
                    cur.execute("CREATE INDEX IF NOT EXISTS idx_emails_date_utc ON emails(date_utc)")
                    cur.execute("CREATE INDEX IF NOT EXISTS idx_emails_content_hash ON emails(content_hash)")
                    
                    # Create full-text search index
                    cur.execute("""
                        CREATE INDEX IF NOT EXISTS idx_emails_fts ON emails 
                        USING GIN(to_tsvector('english', COALESCE(subject, '') || ' ' || COALESCE(content, '')))
                    """)
                    
                    conn.commit()
                    logger.info("PostgreSQL email indexes created successfully")
                    
        except Exception as e:
            logger.error(f"Failed to create emails table: {e}")
            raise

    def upsert_email(self, record: Dict[str, Any]) -> None:
        """Insert or update an email record.

        Args:
            record: Dictionary representing an email message
        """
        if "message_id" not in record or not record["message_id"]:
            raise ValueError("record missing message_id")

        try:
            # Compute header hash if not provided
            if "header_hash" not in record:
                record["header_hash"] = compute_header_hash(record)

            with self.postgresql_manager.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        INSERT INTO emails (
                            message_id, from_addr, subject, date_utc, header_hash, 
                            content_hash, content, attachments, headers, updated_at
                        )
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, CURRENT_TIMESTAMP)
                        ON CONFLICT (header_hash) DO UPDATE SET
                            message_id = EXCLUDED.message_id,
                            from_addr = EXCLUDED.from_addr,
                            subject = EXCLUDED.subject,
                            date_utc = EXCLUDED.date_utc,
                            content_hash = EXCLUDED.content_hash,
                            content = EXCLUDED.content,
                            attachments = EXCLUDED.attachments,
                            headers = EXCLUDED.headers,
                            updated_at = CURRENT_TIMESTAMP
                        """,
                        (
                            record.get("message_id"),
                            record.get("from_addr"),
                            record.get("subject"),
                            record.get("date_utc"),
                            record.get("header_hash"),
                            record.get("content_hash"),
                            record.get("content"),
                            json.dumps(record.get("attachments", [])),
                            json.dumps(record.get("headers", {}))
                        )
                    )
                    conn.commit()
                    logger.info("Upserted email %s", _mask(record["message_id"]))
        except Exception as e:
            logger.error(f"Failed to upsert email record: {e}")
            raise

    def get_email(self, message_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve an email by message_id.
        
        Args:
            message_id: Unique message identifier
            
        Returns:
            Email record dictionary or None if not found
        """
        logger.info("Fetching email for message_id %s", _mask(message_id))
        try:
            with self.postgresql_manager.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        "SELECT * FROM emails WHERE message_id = %s",
                        (message_id,)
                    )
                    row = cur.fetchone()
                    if row:
                        return self._row_to_dict(row, cur.description)
                    return None
        except Exception as e:
            logger.error(f"Failed to get email {message_id}: {e}")
            raise

    def get_email_by_hash(self, content_hash: str) -> Optional[Dict[str, Any]]:
        """Retrieve an email by content hash.
        
        Args:
            content_hash: Content hash of the email
            
        Returns:
            Email record dictionary or None if not found
        """
        logger.info("Fetching email for content_hash %s", _mask(content_hash))
        try:
            with self.postgresql_manager.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        "SELECT * FROM emails WHERE content_hash = %s",
                        (content_hash,)
                    )
                    row = cur.fetchone()
                    if row:
                        return self._row_to_dict(row, cur.description)
                    return None
        except Exception as e:
            logger.error(f"Failed to get email by content hash: {e}")
            raise

    def get_email_by_header_hash(self, header_hash: str) -> Optional[Dict[str, Any]]:
        """Retrieve an email by header hash.
        
        Args:
            header_hash: Header hash of the email
            
        Returns:
            Email record dictionary or None if not found
        """
        logger.info("Fetching email for header_hash %s", _mask(header_hash))
        try:
            with self.postgresql_manager.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        "SELECT * FROM emails WHERE header_hash = %s",
                        (header_hash,)
                    )
                    row = cur.fetchone()
                    if row:
                        return self._row_to_dict(row, cur.description)
                    return None
        except Exception as e:
            logger.error(f"Failed to get email by header hash: {e} (type: {type(e)}) (args: {e.args})")
            raise

    def delete_email(self, message_id: str) -> None:
        """Delete an email by message_id.
        
        Args:
            message_id: Unique message identifier
        """
        logger.info("Deleting email for message_id %s", _mask(message_id))
        try:
            with self.postgresql_manager.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        "DELETE FROM emails WHERE message_id = %s",
                        (message_id,)
                    )
                    conn.commit()
                    logger.info("Deleted email %s", _mask(message_id))
        except Exception as e:
            logger.error(f"Failed to delete email {message_id}: {e}")
            raise

    def _row_to_dict(self, row: Any, description: Any = None) -> Dict[str, Any]:
        """Convert a database row to a dictionary.
        
        Args:
            row: Database row tuple
            description: Cursor description with column info
            
        Returns:
            Dictionary representation of the row
        """
        if not description:
            return {}
            
        result = {}
        for i, col in enumerate(description):
            col_name = col[0]  # Column name is first element
            value = row[i]
            
            # Parse JSON fields
            if col_name in ["attachments", "headers"] and value:
                try:
                    result[col_name] = json.loads(value)
                except (json.JSONDecodeError, TypeError):
                    result[col_name] = value
            else:
                result[col_name] = value
                
        return result

    def fetch_as_dict(self, query: str, params: tuple = ()) -> List[Dict[str, Any]]:
        """Execute a query and return results as list of dictionaries.
        
        Args:
            query: SQL query string
            params: Query parameters
            
        Returns:
            List of dictionaries representing query results
        """
        try:
            with self.postgresql_manager.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(query, params)
                    rows = cur.fetchall()
                    return [self._row_to_dict(row, cur.description) for row in rows]
        except Exception as e:
            logger.error(f"Failed to execute query: {e}")
            raise

    def get_email_count(self) -> int:
        """Get total number of emails stored.
        
        Returns:
            Total email count
        """
        try:
            with self.postgresql_manager.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("SELECT COUNT(*) FROM emails")
                    return cur.fetchone()[0]
        except Exception as e:
            logger.error(f"Failed to get email count: {e}")
            return 0

    def get_emails_by_sender(self, from_addr: str, limit: int = 100) -> List[Dict[str, Any]]:
        """Get emails from a specific sender.
        
        Args:
            from_addr: Email sender address  
            limit: Maximum number of emails to return
            
        Returns:
            List of email dictionaries
        """
        try:
            with self.postgresql_manager.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        "SELECT * FROM emails WHERE from_addr = %s ORDER BY date_utc DESC LIMIT %s",
                        (from_addr, limit)
                    )
                    rows = cur.fetchall()
                    return [self._row_to_dict(row, cur.description) for row in rows]
        except Exception as e:
            logger.error(f"Failed to get emails by sender: {e}")
            return []
