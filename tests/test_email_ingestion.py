"""Tests for email ingestion normalization and processing."""

from __future__ import annotations

from typing import Any, Dict, List
from unittest.mock import MagicMock

import sqlite3
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


sys.modules.setdefault("langchain_text_splitters", types.SimpleNamespace(RecursiveCharacterTextSplitter=_DummySplitter))
sys.modules.setdefault("langchain_ollama", types.SimpleNamespace(OllamaEmbeddings=_DummyEmbeddings))

from ingestion.email.ingest import _normalize, run_email_ingestion
from ingestion.email.processor import EmailProcessor


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
    processed = run_email_ingestion(connector, processor)
    assert processed == 1
    assert len(processor.processed) == 1


def test_email_processor_process_uses_dependencies(raw_email_record: Dict[str, Any]) -> None:
    """`EmailProcessor.process` should store metadata and embeddings."""

    class FakeEmbedModel:
        def embed_documents(self, docs: List[str]) -> List[List[float]]:
            return [[float(i)] for i, _ in enumerate(docs)]

    milvus = MagicMock()
    sqlite_conn = sqlite3.connect(":memory:")
    processor = EmailProcessor(milvus, sqlite_conn, embedding_model=FakeEmbedModel(), chunk_size=50, chunk_overlap=0)
    processor.manager.upsert_email = MagicMock()

    record = raw_email_record.copy()
    record["body_html"] = None
    record["body_text"] = "Hello Bob"

    processor.process(record)

    processor.manager.upsert_email.assert_called_once_with(record)
    milvus.add_embeddings.assert_called_once()
