#!/usr/bin/env python3
"""Utilities for managing IMAP email accounts.

This module provides a simple `EmailAccountManager` for creating,
listing, updating and deleting IMAP account configurations stored in
the ``knowledgebase.db`` SQLite database. Account information can then
be used by :mod:`email_ingestion` to pull messages.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class EmailAccount:
    """Data class representing an IMAP account configuration."""

    name: str
    imap_host: str
    imap_user: str
    imap_password: str
    imap_port: int = 993
    mailbox: str = "INBOX"


class EmailAccountManager:
    """Manage email account configurations in SQLite."""

    def __init__(self, db_path: str = "knowledgebase.db"):
        self.db_path = db_path
        self._init_database()

    def _init_database(self) -> None:
        """Ensure the ``email_accounts`` table exists."""
        with sqlite3.connect(self.db_path) as conn:
            cur = conn.cursor()
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS email_accounts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT UNIQUE NOT NULL,
                    imap_host TEXT NOT NULL,
                    imap_user TEXT NOT NULL,
                    imap_password TEXT NOT NULL,
                    imap_port INTEGER NOT NULL,
                    mailbox TEXT NOT NULL
                )
                """
            )
            conn.commit()

    def list_accounts(self) -> List[EmailAccount]:
        """Return all configured email accounts."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cur = conn.cursor()
            cur.execute("SELECT name, imap_host, imap_user, imap_password, imap_port, mailbox FROM email_accounts")
            rows = cur.fetchall()
            return [EmailAccount(**dict(row)) for row in rows]

    def add_account(self, account: EmailAccount) -> None:
        """Add a new account configuration."""
        with sqlite3.connect(self.db_path) as conn:
            cur = conn.cursor()
            cur.execute(
                """
                INSERT INTO email_accounts (name, imap_host, imap_user, imap_password, imap_port, mailbox)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    account.name,
                    account.imap_host,
                    account.imap_user,
                    account.imap_password,
                    account.imap_port,
                    account.mailbox,
                ),
            )
            conn.commit()

    def remove_account(self, name: str) -> None:
        """Remove an account configuration by name."""
        with sqlite3.connect(self.db_path) as conn:
            cur = conn.cursor()
            cur.execute("DELETE FROM email_accounts WHERE name = ?", (name,))
            conn.commit()

    def get_account(self, name: str) -> Optional[EmailAccount]:
        """Return a single account configuration."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cur = conn.cursor()
            cur.execute(
                "SELECT name, imap_host, imap_user, imap_password, imap_port, mailbox FROM email_accounts WHERE name = ?",
                (name,),
            )
            row = cur.fetchone()
            return EmailAccount(**dict(row)) if row else None

    def update_account(self, account: EmailAccount) -> None:
        """Update an existing account configuration."""
        with sqlite3.connect(self.db_path) as conn:
            cur = conn.cursor()
            cur.execute(
                """
                UPDATE email_accounts
                   SET imap_host = ?,
                       imap_user = ?,
                       imap_password = ?,
                       imap_port = ?,
                       mailbox = ?
                 WHERE name = ?
                """,
                (
                    account.imap_host,
                    account.imap_user,
                    account.imap_password,
                    account.imap_port,
                    account.mailbox,
                    account.name,
                ),
            )
            conn.commit()
