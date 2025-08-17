from __future__ import annotations

"""CRUD helpers for managing email account configurations."""

import logging
import sqlite3
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


class EmailAccountManager:
    """CRUD helper for the ``email_accounts`` table."""

    def __init__(self, conn: sqlite3.Connection) -> None:
        """Initialize the manager and ensure the table exists."""
        self.conn = conn
        self._ensure_table()

    # ------------------------------------------------------------------
    def _ensure_table(self) -> None:
        """Create the ``email_accounts`` table if it does not exist."""
        cur = self.conn.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS email_accounts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                account_name TEXT UNIQUE NOT NULL,
                server_type TEXT NOT NULL,
                server TEXT NOT NULL,
                port INTEGER NOT NULL,
                username TEXT NOT NULL,
                password TEXT NOT NULL,
                mailbox TEXT,
                batch_limit INTEGER,
                use_ssl INTEGER
            )
            """
        )
        self.conn.commit()

    # ------------------------------------------------------------------
    def create_account(self, record: Dict[str, Any]) -> int:
        """Create a new email account record.

        Parameters
        ----------
        record:
            Mapping of column names to values for the new account.

        Returns
        -------
        int
            The ID of the newly created account.
        """
        required = {
            "account_name",
            "server_type",
            "server",
            "port",
            "username",
            "password",
        }
        missing = required - record.keys()
        if missing:
            raise ValueError(f"record missing required fields: {', '.join(sorted(missing))}")

        cols = list(record.keys())
        placeholders = ",".join(["?"] * len(cols))
        sql = f"INSERT INTO email_accounts ({','.join(cols)}) VALUES ({placeholders})"
        cur = self.conn.cursor()
        cur.execute(sql, [record[c] for c in cols])
        self.conn.commit()
        account_id = cur.lastrowid
        logger.debug("Created email account %s", account_id)
        return account_id

    # ------------------------------------------------------------------
    def list_accounts(self) -> List[Dict[str, Any]]:
        """Return all email account records."""
        self.conn.row_factory = sqlite3.Row
        cur = self.conn.cursor()
        cur.execute("SELECT * FROM email_accounts")
        rows = cur.fetchall()
        return [self._row_to_dict(r) for r in rows]

    # ------------------------------------------------------------------
    def update_account(self, account_id: int, updates: Dict[str, Any]) -> None:
        """Update an email account record.

        Parameters
        ----------
        account_id:
            ID of the account to update.
        updates:
            Mapping of column names to new values.
        """
        if not updates:
            return
        assignments = ", ".join([f"{col} = ?" for col in updates])
        values = list(updates.values()) + [account_id]
        cur = self.conn.cursor()
        cur.execute(
            f"UPDATE email_accounts SET {assignments} WHERE id = ?",
            values,
        )
        self.conn.commit()
        logger.debug("Updated email account %s", account_id)

    # ------------------------------------------------------------------
    def delete_account(self, account_id: int) -> None:
        """Delete an email account record."""
        cur = self.conn.cursor()
        cur.execute("DELETE FROM email_accounts WHERE id = ?", (account_id,))
        self.conn.commit()
        logger.debug("Deleted email account %s", account_id)

    # ------------------------------------------------------------------
    def _row_to_dict(self, row: sqlite3.Row) -> Dict[str, Any]:
        data = dict(row)
        if "use_ssl" in data:
            data["use_ssl"] = bool(data["use_ssl"])
        return data
