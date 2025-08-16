"""Tests for email ingestion."""

import os
import sys
import imaplib
import sqlite3
from email.message import EmailMessage

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from email_accounts import EmailAccount, EmailAccountManager
import email_ingestion
from email_ingestion import ingest_emails



class _DummyMilvus:
    def __init__(self, *args, **kwargs):
        pass

    def add_documents(self, documents):
        pass

    @classmethod
    def from_documents(cls, docs, embeddings, **kwargs):
        return cls()

class _MockIMAP:
    def __init__(self, messages):
        self.messages = messages

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def login(self, user, password):
        return "OK"

    def select(self, mailbox):
        return "OK", None

    def search(self, charset, criteria):
        return "OK", [b" ".join(self.messages.keys())]

    def fetch(self, num, _):
        return "OK", [(None, self.messages[num])]

    def logout(self):
        return "OK"


def _make_message(msg_id: str, subject: str) -> bytes:
    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = "a@example.com"
    msg["To"] = "b@example.com"
    msg["Date"] = "Mon, 1 Jan 2024 00:00:00 +0000"
    msg["Message-ID"] = msg_id
    msg.set_content("body")
    return msg.as_bytes()


def test_ingest_emails_deduplicates(monkeypatch, tmp_path):
    """Duplicate headers should be skipped."""
    db = tmp_path / "kb.db"
    manager = EmailAccountManager(str(db))
    manager.add_account(EmailAccount(name="acct", imap_host="h", imap_user="u", imap_password="p"))

    msg_bytes = _make_message("<1>", "Subj")
    messages = {b"1": msg_bytes, b"2": msg_bytes}
    monkeypatch.setattr(imaplib, "IMAP4_SSL", lambda *args, **kwargs: _MockIMAP(messages))
    monkeypatch.setattr(email_ingestion, "LC_Milvus", _DummyMilvus)

    processed = ingest_emails("acct", db_path=str(db))
    assert processed == 1
    with sqlite3.connect(str(db)) as conn:
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM emails")
        assert cur.fetchone()[0] == 1


def test_ingest_emails_respects_limit(monkeypatch, tmp_path):
    """Only the requested number of messages should be processed."""
    db = tmp_path / "kb.db"
    manager = EmailAccountManager(str(db))
    manager.add_account(EmailAccount(name="acct", imap_host="h", imap_user="u", imap_password="p"))

    msg1 = _make_message("<1>", "A")
    msg2 = _make_message("<2>", "B")
    messages = {b"1": msg1, b"2": msg2}
    monkeypatch.setattr(imaplib, "IMAP4_SSL", lambda *args, **kwargs: _MockIMAP(messages))
    monkeypatch.setattr(email_ingestion, "LC_Milvus", _DummyMilvus)

    processed = ingest_emails("acct", limit=1, db_path=str(db))
    assert processed == 1
    with sqlite3.connect(str(db)) as conn:
        cur = conn.cursor()
        cur.execute("SELECT subject FROM emails")
        rows = cur.fetchall()
        assert rows == [("B",)]
