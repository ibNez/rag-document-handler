from __future__ import annotations

"""CRUD helpers for managing email account configurations."""

import logging
import sqlite3
from typing import Any, Dict, List

from .crypto import decrypt, encrypt

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
                use_ssl INTEGER,
                refresh_interval_minutes INTEGER,
                last_synced TIMESTAMP
            )
            """
        )
        # Add missing columns for upgrades
        cur.execute("PRAGMA table_info(email_accounts)")
        existing = {row[1] for row in cur.fetchall()}
        if "refresh_interval_minutes" not in existing:
            cur.execute(
                "ALTER TABLE email_accounts ADD COLUMN refresh_interval_minutes INTEGER"
            )
        if "last_synced" not in existing:
            cur.execute(
                "ALTER TABLE email_accounts ADD COLUMN last_synced TIMESTAMP"
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

        # Encrypt the password before persisting
        record = dict(record)
        if "password" in record:
            record["password"] = encrypt(str(record["password"]))

        logger.info(
            "Creating email account '%s' (%s) on %s:%s for user %s",
            record.get("account_name"),
            record.get("server_type"),
            record.get("server"),
            record.get("port"),
            record.get("username"),
        )

        cols = list(record.keys())
        placeholders = ",".join(["?"] * len(cols))
        sql = f"INSERT INTO email_accounts ({','.join(cols)}) VALUES ({placeholders})"
        cur = self.conn.cursor()
        cur.execute(sql, [record[c] for c in cols])
        self.conn.commit()
        account_id = cur.lastrowid
        logger.info("Created email account %s (%s)", account_id, record.get("account_name"))
        return account_id

    # ------------------------------------------------------------------
    def list_accounts(self, include_password: bool = False) -> List[Dict[str, Any]]:
        """Return all email account records.

        Parameters
        ----------
        include_password:
            If ``True`` decrypted passwords are included in the returned
            dictionaries.  When ``False`` (default) the ``password`` field is
            omitted entirely so sensitive data is not exposed to templates or
            API responses.
        """
        self.conn.row_factory = sqlite3.Row
        cur = self.conn.cursor()
        cur.execute(
            """
            SELECT
                *,
                CASE
                    WHEN refresh_interval_minutes IS NOT NULL AND refresh_interval_minutes > 0 AND last_synced IS NOT NULL
                    THEN datetime(last_synced, '+' || refresh_interval_minutes || ' minutes')
                    ELSE NULL
                END AS next_run
            FROM email_accounts
            """
        )
        rows = cur.fetchall()

        accounts: List[Dict[str, Any]] = []
        for row in rows:
            data = self._row_to_dict(row)
            if include_password and "password" in data:
                try:
                    data["password"] = decrypt(data["password"])
                except Exception:
                    data["password"] = ""
            else:
                data.pop("password", None)
            accounts.append(data)
        return accounts

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
        # Encrypt password if present
        params = dict(updates)
        if "password" in params:
            params["password"] = encrypt(str(params["password"]))

        assignments = ", ".join([f"{col} = ?" for col in params])
        values = list(params.values()) + [account_id]
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
    def get_account_count(self) -> int:
        """Get the total number of active email accounts."""
        try:
            cur = self.conn.cursor()
            cur.execute(
                "SELECT COUNT(*) FROM email_accounts WHERE refresh_interval_minutes IS NULL OR refresh_interval_minutes > 0"
            )
            return cur.fetchone()[0]
        except Exception as exc:  # pragma: no cover - defensive
            logger.error(f"Error getting email account count: {exc}")
            return 0

    # ------------------------------------------------------------------
    def _row_to_dict(self, row: sqlite3.Row) -> Dict[str, Any]:
        data = dict(row)
        if "use_ssl" in data:
            data["use_ssl"] = bool(data["use_ssl"])
        return data
