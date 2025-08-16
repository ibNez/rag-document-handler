from __future__ import annotations

"""Helpers for persisting email metadata to SQLite."""

import json
import logging
from typing import Any, Dict, Optional

import sqlite3

logger = logging.getLogger(__name__)


class EmailManager:
    """CRUD helper for the ``emails`` table."""

    def __init__(self, conn: sqlite3.Connection) -> None:
        self.conn = conn
        self._ensure_table()

    # ------------------------------------------------------------------
    def _ensure_table(self) -> None:
        """Create the ``emails`` table if it does not exist."""
        cursor = self.conn.cursor()
        cursor.execute(
            '''
            CREATE TABLE IF NOT EXISTS emails (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                message_id TEXT UNIQUE NOT NULL,
                thread_id TEXT,
                subject TEXT,
                from_addr TEXT,
                to_addrs TEXT,
                cc_addrs TEXT,
                date_utc TEXT,
                received_utc TEXT,
                in_reply_to TEXT,
                references_ids TEXT,
                is_reply INTEGER,
                is_forward INTEGER,
                raw_size_bytes INTEGER,
                body_text TEXT,
                body_html TEXT,
                language TEXT,
                has_attachments INTEGER,
                attachment_manifest TEXT,
                processed INTEGER DEFAULT 0,
                ingested_at TEXT,
                updated_at TEXT,
                content_hash TEXT,
                summary TEXT,
                keywords TEXT,
                auto_topic TEXT,
                manual_topic TEXT,
                topic_confidence REAL,
                topic_version TEXT,
                error_state TEXT,
                direction TEXT,
                participants TEXT,
                participants_hash TEXT,
                to_primary TEXT
            )
            '''
        )
        self.conn.commit()

    # ------------------------------------------------------------------
    def upsert_email(self, record: Dict[str, Any]) -> None:
        """Insert or update an email record.

        Parameters
        ----------
        record:
            Dictionary representing an email as produced by the connector.
        """
        if "message_id" not in record or not record["message_id"]:
            raise ValueError("record missing message_id")

        cols = list(record.keys())
        placeholders = ",".join(["?"] * len(cols))
        updates = ", ".join([f"{c}=excluded.{c}" for c in cols if c != "message_id"])
        sql = f"INSERT INTO emails ({','.join(cols)}) VALUES ({placeholders}) ON CONFLICT(message_id) DO UPDATE SET {updates}"

        values = []
        for c in cols:
            v = record[c]
            if isinstance(v, (list, dict)):
                values.append(json.dumps(v))
            else:
                values.append(v)

        cur = self.conn.cursor()
        cur.execute(sql, values)
        self.conn.commit()
        logger.debug("Upserted email %s", record["message_id"])

    # ------------------------------------------------------------------
    def get_email(self, message_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve an email by ``message_id``."""
        cur = self.conn.cursor()
        self.conn.row_factory = sqlite3.Row
        cur.execute("SELECT * FROM emails WHERE message_id = ?", (message_id,))
        row = cur.fetchone()
        if not row:
            return None
        return self._row_to_dict(row)

    # ------------------------------------------------------------------
    def get_email_by_hash(self, content_hash: str) -> Optional[Dict[str, Any]]:
        """Retrieve an email using its ``content_hash``."""
        cur = self.conn.cursor()
        self.conn.row_factory = sqlite3.Row
        cur.execute("SELECT * FROM emails WHERE content_hash = ?", (content_hash,))
        row = cur.fetchone()
        if not row:
            return None
        return self._row_to_dict(row)

    # ------------------------------------------------------------------
    def delete_email(self, message_id: str) -> None:
        """Delete an email by ``message_id``."""
        cur = self.conn.cursor()
        cur.execute("DELETE FROM emails WHERE message_id = ?", (message_id,))
        self.conn.commit()

    # ------------------------------------------------------------------
    def _row_to_dict(self, row: sqlite3.Row) -> Dict[str, Any]:
        data = dict(row)
        json_cols = {
            "to_addrs",
            "cc_addrs",
            "references_ids",
            "attachment_manifest",
            "participants",
            "keywords",
        }
        for col in json_cols:
            if data.get(col):
                try:
                    data[col] = json.loads(data[col])
                except Exception:
                    pass
        return data
