"""Tests for email ingestion normalization and processing."""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock

import logging
import pytest
import sys
import os
import types

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


class _DummySplitter:
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        pass

    def split_text(self, text: str) -> List[str]:
        return [text]


class _DummyEmbeddings:
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        pass

    def embed_documents(self, docs: List[str]) -> List[List[float]]:
        return [[0.0] * 1 for _ in docs]


sys.modules["langchain_text_splitters"] = types.SimpleNamespace(RecursiveCharacterTextSplitter=_DummySplitter)
sys.modules["langchain_ollama"] = types.SimpleNamespace(OllamaEmbeddings=_DummyEmbeddings)
sys.modules.setdefault("google.oauth2.credentials", types.SimpleNamespace(Credentials=object))
sys.modules.setdefault("googleapiclient.discovery", types.SimpleNamespace(build=lambda *a, **k: None))
sys.modules.setdefault("googleapiclient.errors", types.SimpleNamespace(HttpError=Exception))
sys.modules.setdefault("google.auth.transport.requests", types.SimpleNamespace(Request=object))
sys.modules.setdefault("cryptography.fernet", types.SimpleNamespace(Fernet=object))

from ingestion.email.ingest import _normalize, run_email_ingestion
from ingestion.email.processor import EmailProcessor
import ingestion.email.orchestrator as orchestrator_module
from ingestion.email.account_manager import EmailAccountManager


postgres_config = {
    "host": "localhost",
    "port": 5432,
    "user": "test_user",
    "password": "test_password",
    "database": "test_db"
}


@pytest.fixture
def email_manager():
    return EmailAccountManager(postgres_config)


@pytest.fixture
def processor(email_manager):
    milvus = MagicMock()
    return EmailProcessor(milvus, email_manager, embedding_model=_DummyEmbeddings(), chunk_size=50, chunk_overlap=0)


@pytest.fixture
def raw_email_record() -> Dict[str, Any]:
    """Return a synthetic raw email record with HTML and signature."""
    return {
        "message_id": "msg-1",
        "thread_id": None,
        "subject": "Greetings",
        "from_addr": "alice@example.com",
        "to_addrs": ["bob@example.com"],
        "cc_addrs": [],
        "date_utc": "2024-01-01T00:00:00",
        "received_utc": None,
        "in_reply_to": None,
        "references_ids": [],
        "is_reply": 0,
        "is_forward": 0,
        "raw_size_bytes": None,
        "body_text": None,
        "body_html": "<div>Hello Bob</div>\n--\n<div>Alice</div>",
        "language": None,
        "has_attachments": 0,
        "attachment_manifest": [],
        "processed": 0,
        "ingested_at": None,
        "updated_at": None,
        "content_hash": None,
        "summary": None,
        "keywords": None,
        "auto_topic": None,
        "manual_topic": None,
        "topic_confidence": None,
        "topic_version": None,
        "error_state": None,
        "direction": None,
        "participants": [],
        "participants_hash": None,
        "to_primary": None,
        "server_type": "imap",
    }


def test_normalization_strips_html_and_signature(raw_email_record: Dict[str, Any]) -> None:
    """Body HTML should be stripped and signatures removed."""
    normalized = _normalize(raw_email_record)
    assert normalized["body_text"] == "Hello Bob"


def test_content_hash_idempotent(raw_email_record: Dict[str, Any]) -> None:
    """Normalization produces stable content hashes across runs."""
    first = _normalize(raw_email_record)
    h1 = first["content_hash"]
    second = _normalize(first)
    h2 = second["content_hash"]
    assert h1 == h2


def test_run_email_ingestion_deduplicates() -> None:
    """Duplicate emails are not processed twice."""

    class DummyConnector:
        def fetch_emails(self, since_date=None):
            base = {
                "thread_id": None,
                "subject": "Hi",
                "from_addr": "a@example.com",
                "to_addrs": ["b@example.com"],
                "cc_addrs": [],
                "date_utc": "2024-01-01",
                "received_utc": None,
                "in_reply_to": None,
                "references_ids": [],
                "is_reply": 0,
                "is_forward": 0,
                "raw_size_bytes": None,
                "body_text": None,
                "body_html": "<p>First</p>",
                "language": None,
                "has_attachments": 0,
                "attachment_manifest": [],
                "processed": 0,
                "ingested_at": None,
                "updated_at": None,
                "content_hash": None,
                "summary": None,
                "keywords": None,
                "auto_topic": None,
                "manual_topic": None,
                "topic_confidence": None,
                "topic_version": None,
                "error_state": None,
                "direction": None,
                "participants": [],
                "participants_hash": None,
                "to_primary": None,
                "server_type": "imap",
            }
            rec1 = {"message_id": "1", **base}
            rec2 = {"message_id": "1", **base, "body_html": "<p>Second</p>"}
            return [rec1, rec2]

    class DummyManager:
        def __init__(self):
            self.seen: set[str] = set()

        def get_email_by_header_hash(self, hh: str):
            return {"header_hash": hh} if hh in self.seen else None

        def get_email_by_hash(self, ch: str):
            return None

        def upsert_email(self, record: Dict[str, Any]) -> None:  # pragma: no cover - stub
            pass

    class DummyProcessor:
        def __init__(self):
            self.manager = DummyManager()
            self.processed: List[Dict[str, Any]] = []

        def process(self, record: Dict[str, Any]) -> None:
            self.manager.seen.add(record["header_hash"])
            self.processed.append(record)

    connector = DummyConnector()
    processor = DummyProcessor()
    processed, failures = run_email_ingestion(connector, processor)
    assert processed == 1
    assert failures == 0
    assert len(processor.processed) == 1


def test_run_email_ingestion_counts_failures(caplog: pytest.LogCaptureFixture) -> None:
    """Failures should be logged and counted without stopping processing."""

    class DummyConnector:
        def fetch_emails(self, since_date=None):
            base = {
                "thread_id": None,
                "subject": "Hi",
                "from_addr": "a@example.com",
                "to_addrs": ["b@example.com"],
                "cc_addrs": [],
                "date_utc": "2024-01-01",
                "received_utc": None,
                "in_reply_to": None,
                "references_ids": [],
                "is_reply": 0,
                "is_forward": 0,
                "raw_size_bytes": None,
                "body_text": None,
                "body_html": "<p>First</p>",
                "language": None,
                "has_attachments": 0,
                "attachment_manifest": [],
                "processed": 0,
                "ingested_at": None,
                "updated_at": None,
                "content_hash": None,
                "summary": None,
                "keywords": None,
                "auto_topic": None,
                "manual_topic": None,
                "topic_confidence": None,
                "topic_version": None,
                "error_state": None,
                "direction": None,
                "participants": [],
                "participants_hash": None,
                "to_primary": None,
                "server_type": "imap",
            }
            rec1 = {"message_id": "1", **base}
            rec2 = {"message_id": "2", **base}
            rec3 = {"message_id": "3", **base}
            return [rec1, rec2, rec3]

    class DummyManager:
        def get_email_by_header_hash(self, hh: str):
            return None

        def get_email_by_hash(self, ch: str):
            return None

    class DummyProcessor:
        def __init__(self):
            self.manager = DummyManager()
            self.processed: List[Dict[str, Any]] = []

        def process(self, record: Dict[str, Any]) -> None:
            if record["message_id"] == "2":
                raise ValueError("boom")
            self.processed.append(record)

    connector = DummyConnector()
    processor = DummyProcessor()
    with caplog.at_level(logging.ERROR):
        processed, failures = run_email_ingestion(connector, processor)
    assert processed == 2
    assert failures == 1
    assert [r["message_id"] for r in processor.processed] == ["1", "3"]
    assert "Failed to process email 2" in caplog.text


def test_email_processor_process_uses_dependencies(raw_email_record: Dict[str, Any]) -> None:
    """`EmailProcessor.process` should store metadata and embeddings."""

    class FakeEmbedModel:
        def embed_documents(self, docs: List[str]) -> List[List[float]]:
            return [[float(i)] for i, _ in enumerate(docs)]

    milvus = MagicMock()
    from ingestion.email.account_manager import EmailAccountManager

    postgres_config = {
        "host": "localhost",
        "port": 5432,
        "user": "test_user",
        "password": "test_password",
        "database": "test_db"
    }

    email_manager = EmailAccountManager(postgres_config)
    processor = EmailProcessor(milvus, email_manager, embedding_model=FakeEmbedModel(), chunk_size=50, chunk_overlap=0)
    processor.manager.upsert_email = MagicMock()

    record = raw_email_record.copy()
    record["body_html"] = None
    record["body_text"] = "Hello Bob"

    processor.process(record)

    processor.manager.upsert_email.assert_called_once_with(record)
    milvus.add_embeddings.assert_called_once()


def test_email_processor_logs_and_counts_missing_body(caplog: pytest.LogCaptureFixture, processor) -> None:
    """Messages lacking body text should be logged and counted."""

    record = {"message_id": "skip-1", "body_text": "   "}

    with caplog.at_level(logging.DEBUG):
        processor.process(record)

    assert "Skipping message skip-1 due to missing body text" in caplog.text
    assert processor.skipped_messages == 1


def test_email_processor_logs_and_counts_zero_chunks(caplog: pytest.LogCaptureFixture) -> None:
    """Zero chunks after splitting should be logged and counted."""

    class FakeEmbedModel:
        def embed_documents(self, docs: List[str]) -> List[List[float]]:  # pragma: no cover - stub
            return [[0.0] for _ in docs]

    milvus = MagicMock()
    sqlite_conn = sqlite3.connect(":memory:")
    processor = EmailProcessor(
        milvus, sqlite_conn, embedding_model=FakeEmbedModel(), chunk_size=50, chunk_overlap=0
    )
    processor.manager.upsert_email = MagicMock()

    # Force splitter to return only empty/whitespace strings
    processor.splitter.split_text = MagicMock(return_value=["   ", ""])

    record = {"message_id": "skip-2", "body_text": "hello"}

    with caplog.at_level(logging.DEBUG):
        processor.process(record)

    assert (
        "Skipping message skip-2 because splitter produced no chunks" in caplog.text
    )
    assert processor.skipped_messages == 1


def test_email_processor_uses_env_model(monkeypatch: pytest.MonkeyPatch) -> None:
    """Default EmailProcessor should initialize OllamaEmbeddings with env model."""

    captured: Dict[str, Any] = {}

    class DummyEmbeddings:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            captured.update(kwargs)

        def embed_documents(self, docs: List[str]) -> List[List[float]]:  # pragma: no cover - stub
            return [[0.0] for _ in docs]

    import ingestion.email.processor as processor_module

    monkeypatch.setenv("EMBEDDING_MODEL", "env-model")
    monkeypatch.setattr(processor_module, "OllamaEmbeddings", DummyEmbeddings)

    milvus = MagicMock()
    sqlite_conn = sqlite3.connect(":memory:")
    processor_module.EmailProcessor(milvus, sqlite_conn)

    assert captured.get("model") == "env-model"


def test_email_orchestrator_processes_multiple_accounts(monkeypatch: pytest.MonkeyPatch) -> None:
    """EmailOrchestrator should fetch emails for each configured account."""

    processed: list[str] = []

    class DummyIMAPConnector:
        def __init__(self, **kwargs: Any) -> None:
            self.email_address = kwargs.get("email_address")

        def fetch_emails(self):
            processed.append(self.email_address)
            return []

    class DummyGmailConnector:
        def __init__(self, **kwargs: Any) -> None:
            self.user_id = kwargs.get("user_id")

        def fetch_emails(self):
            processed.append(self.user_id)
            return []

    class DummyCreds:
        expired = False
        refresh_token = None

        def refresh(self, _):
            pass

        def to_json(self):  # pragma: no cover - stub
            return "{}"

    class DummyCredentials:
        @staticmethod
        def from_authorized_user_file(*args, **kwargs):
            return DummyCreds()

    monkeypatch.setattr(orchestrator_module, "IMAPConnector", DummyIMAPConnector)
    monkeypatch.setattr(orchestrator_module, "GmailConnector", DummyGmailConnector)
    monkeypatch.setattr(orchestrator_module, "Credentials", DummyCredentials)
    monkeypatch.setattr(orchestrator_module, "Request", lambda: None)

    class DummyAccountManager:
        def list_accounts(self, include_password: bool = False):
            return [
                {
                    "server_type": "imap",
                    "server": "s1",
                    "port": 1,
                    "email_address": "u1",
                    "password": "p",
                    "use_ssl": 1,
                },
                {
                    "server_type": "gmail",
                    "email_address": "u2",
                    "token_file": "/tmp/token.json",
                },
            ]

    class DummyConfig:
        EMAIL_ENABLED = True
        EMAIL_SYNC_INTERVAL_SECONDS = 0

    orchestrator = orchestrator_module.EmailOrchestrator(
        DummyConfig(), account_manager=DummyAccountManager()
    )
    orchestrator.run_cycle()
    assert processed == ["u1", "u2"]


def test_email_orchestrator_processes_and_persists(
    monkeypatch: pytest.MonkeyPatch, raw_email_record: Dict[str, Any]
) -> None:
    """EmailOrchestrator should process fetched emails using EmailProcessor."""

    class DummyIMAPConnector:
        def __init__(self, **kwargs: Any) -> None:
            pass

        def fetch_emails(self):
            return [raw_email_record]

    monkeypatch.setattr(orchestrator_module, "IMAPConnector", DummyIMAPConnector)

    class DummyAccountManager:
        def __init__(self) -> None:
            self.updated: Optional[Dict[str, Any]] = None

        def list_accounts(self, include_password: bool = False):
            return [
                {
                    "id": 1,
                    "server_type": "imap",
                    "server": "s1",
                    "port": 993,
                    "email_address": "u1",
                    "password": "p",
                    "use_ssl": 1,
                }
            ]

        def update_account(self, acct_id: int, data: Dict[str, Any]) -> None:
            self.updated = {"acct_id": acct_id, **data}

    class DummyConfig:
        EMAIL_ENABLED = True
        EMAIL_DEFAULT_REFRESH_MINUTES = 5

    class DummyMilvus:
        def add_embeddings(self, embeddings, ids, metadatas):
            pass

    sqlite_conn = sqlite3.connect(":memory:")
    processor = EmailProcessor(DummyMilvus(), sqlite_conn, embedding_model=_DummyEmbeddings())

    orchestrator = orchestrator_module.EmailOrchestrator(
        DummyConfig(), account_manager=DummyAccountManager(), processor=processor
    )

    orchestrator.run_cycle()

    cur = sqlite_conn.cursor()
    cur.execute("SELECT message_id FROM emails")
    row = cur.fetchone()
    assert row[0] == raw_email_record["message_id"]


def test_run_email_ingestion_header_hash_dedup(monkeypatch: pytest.MonkeyPatch) -> None:
    """Emails with duplicate header hashes should be skipped."""

    class DummyConnector:
        def fetch_emails(self, since_date=None):
            base = {
                "thread_id": None,
                "subject": "Hi",
                "from_addr": "a@example.com",
                "to_addrs": ["b@example.com"],
                "cc_addrs": [],
                "date_utc": "2024-01-01",
                "received_utc": None,
                "in_reply_to": None,
                "references_ids": [],
                "is_reply": 0,
                "is_forward": 0,
                "raw_size_bytes": None,
                "body_text": None,
                "body_html": "<p>First</p>",
                "language": None,
                "has_attachments": 0,
                "attachment_manifest": [],
                "processed": 0,
                "ingested_at": None,
                "updated_at": None,
                "content_hash": None,
                "summary": None,
                "keywords": None,
                "auto_topic": None,
                "manual_topic": None,
                "topic_confidence": None,
                "topic_version": None,
                "error_state": None,
                "direction": None,
                "participants": [],
                "participants_hash": None,
                "to_primary": None,
                "server_type": "imap",
            }
            rec1 = {"message_id": "1", **base}
            rec2 = {"message_id": "1", **base, "body_html": "<p>Second</p>"}
            return [rec1, rec2]

    seen: set[str] = set()

    def get_email_by_header_hash(hh: str):
        if hh in seen:
            return {"header_hash": hh}
        seen.add(hh)
        return None

    manager = MagicMock()
    manager.get_email_by_header_hash.side_effect = get_email_by_header_hash
    manager.get_email_by_hash.return_value = None

    class DummyProcessor:
        def __init__(self):
            self.manager = manager
            self.processed: list[Dict[str, Any]] = []

        def process(self, record: Dict[str, Any]) -> None:
            self.processed.append(record)

    connector = DummyConnector()
    processor = DummyProcessor()
    processed, failures = run_email_ingestion(connector, processor)
    assert processed == 1
    assert failures == 0
    assert len(processor.processed) == 1
    assert manager.get_email_by_header_hash.call_count == 2


def test_email_orchestrator_uses_gmail_connector(monkeypatch: pytest.MonkeyPatch) -> None:
    """Gmail accounts should invoke ``GmailConnector`` and not the IMAP connector."""

    used = {"gmail": False}

    class DummyGmailConnector:
        def __init__(self, **kwargs: Any) -> None:  # pragma: no cover - simple data holder
            used["gmail"] = True

        def fetch_emails(self):
            return []

    class DummyIMAPConnector:
        def __init__(self, **kwargs: Any) -> None:  # pragma: no cover - should not run
            raise AssertionError("IMAPConnector should not be used for Gmail accounts")

        def fetch_emails(self):
            return []

    class DummyCreds:
        expired = False
        refresh_token = None

        def refresh(self, _):
            pass

        def to_json(self):  # pragma: no cover - stub
            return "{}"

    class DummyCredentials:
        @staticmethod
        def from_authorized_user_file(*args, **kwargs):
            return DummyCreds()

    monkeypatch.setattr(orchestrator_module, "GmailConnector", DummyGmailConnector)
    monkeypatch.setattr(orchestrator_module, "IMAPConnector", DummyIMAPConnector)
    monkeypatch.setattr(orchestrator_module, "Credentials", DummyCredentials)
    monkeypatch.setattr(orchestrator_module, "Request", lambda: None)

    class DummyAccountManager:
        def list_accounts(self, include_password: bool = False):
            return [
                {
                    "server_type": "gmail",
                    "email_address": "user",
                    "token_file": "/tmp/token.json",
                }
            ]

    class DummyConfig:
        EMAIL_ENABLED = True
        EMAIL_SYNC_INTERVAL_SECONDS = 0

    orchestrator = orchestrator_module.EmailOrchestrator(
        DummyConfig(), account_manager=DummyAccountManager()
    )
    orchestrator.run_cycle()
    assert used["gmail"] is True
