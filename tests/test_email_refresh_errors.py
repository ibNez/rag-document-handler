"""Tests for email refresh error handling in _refresh_email_account_background."""

from __future__ import annotations

import os
import sys
import types
import logging
import pytest

# Stub external dependencies for app import
sys.modules.setdefault("pypdf", types.SimpleNamespace(PdfReader=lambda *a, **k: None))
sys.modules.setdefault("docx", types.SimpleNamespace(Document=lambda *a, **k: None))
sys.modules.setdefault("langchain_community.document_loaders", types.SimpleNamespace(PyPDFLoader=None))
sys.modules.setdefault("langchain_unstructured", types.SimpleNamespace(UnstructuredLoader=None))
sys.modules.setdefault("langchain_text_splitters", types.SimpleNamespace(RecursiveCharacterTextSplitter=lambda *a, **k: None))
sys.modules.setdefault(
    "pymilvus",
    types.SimpleNamespace(
        connections=types.SimpleNamespace(connect=lambda *a, **k: None),
        utility=types.SimpleNamespace(get_server_version=lambda: ""),
        Collection=lambda *a, **k: None,
    ),
)
sys.modules.setdefault("werkzeug.utils", types.SimpleNamespace(secure_filename=lambda x: x))
sys.modules.setdefault("werkzeug.datastructures", types.SimpleNamespace(FileStorage=object))
sys.modules.setdefault("chardet", types.SimpleNamespace(detect=lambda *a, **k: {"encoding": "utf-8"}))
_ollama = types.ModuleType("langchain_ollama")
_ollama.OllamaEmbeddings = lambda *a, **k: None
_ollama.ChatOllama = lambda *a, **k: None
sys.modules["langchain_ollama"] = _ollama
sys.modules.setdefault(
    "langchain_community.vectorstores",
    types.SimpleNamespace(Milvus=types.SimpleNamespace(from_texts=lambda *a, **k: None)),
)
sys.modules.setdefault("langchain_core.documents", types.SimpleNamespace(Document=object))
sys.modules.setdefault(
    "langchain_core.messages",
    types.SimpleNamespace(SystemMessage=object, HumanMessage=object, AIMessage=object),
)
sys.modules.setdefault("langchain_core.tools", types.SimpleNamespace(tool=lambda f: f))
sys.modules.setdefault("dotenv", types.SimpleNamespace(load_dotenv=lambda *a, **k: None))
sys.modules.setdefault("google.oauth2.credentials", types.SimpleNamespace(Credentials=object))
sys.modules.setdefault("googleapiclient.discovery", types.SimpleNamespace(build=lambda *a, **k: None))
sys.modules.setdefault("googleapiclient.errors", types.SimpleNamespace(HttpError=Exception))
sys.modules.setdefault("google.auth.transport.requests", types.SimpleNamespace(Request=object))
sys.modules.setdefault("cryptography.fernet", types.SimpleNamespace(Fernet=object))
sys.modules.setdefault("requests", types.SimpleNamespace(get=lambda *a, **k: None))
sys.modules.setdefault("bs4", types.SimpleNamespace(BeautifulSoup=lambda *a, **k: None))
sys.modules.setdefault(
    "flask",
    types.SimpleNamespace(
        Flask=object,
        request=None,
        render_template=None,
        flash=None,
        redirect=None,
        url_for=None,
        jsonify=None,
        send_from_directory=None,
        abort=None,
    ),
)

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import app


class DummyConnector:
    def __init__(self, *a, **k):
        pass


@pytest.fixture(autouse=True)
def _stub_email_modules(monkeypatch):
    """Provide dummy email ingestion modules used by refresh function."""
    monkeypatch.setitem(
        sys.modules,
        "ingestion.email.connector",
        types.SimpleNamespace(
            IMAPConnector=DummyConnector,
            GmailConnector=DummyConnector,
            ExchangeConnector=DummyConnector,
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "ingestion.email.processor",
        types.SimpleNamespace(EmailProcessor=lambda *a, **k: object()),
    )
    monkeypatch.setitem(
        sys.modules,
        "ingestion.email.ingest",
        types.SimpleNamespace(run_email_ingestion=lambda *a, **k: True),
    )


def _make_manager(account_manager, status_store=None):
    mgr = object.__new__(app.RAGKnowledgebaseManager)
    mgr.email_account_manager = account_manager
    mgr.email_processing_status = status_store or {}
    mgr.url_manager = types.SimpleNamespace(db_path=":memory:")
    mgr.milvus_manager = object()
    return mgr


def test_update_account_failure_logs(monkeypatch, caplog):
    class DummyAccountManager:
        def list_accounts(self, include_password=False):
            return [
                {
                    "id": 1,
                    "account_name": "acc",
                    "server_type": "imap",
                    "server": "srv",
                    "port": 993,
                    "username": "u",
                    "password": "p",
                }
            ]

        def update_account(self, account_id, data):
            raise RuntimeError("db fail")

    mgr = _make_manager(DummyAccountManager())

    with caplog.at_level(logging.ERROR):
        mgr._refresh_email_account_background(1)

    messages = [r.getMessage() for r in caplog.records]
    assert any("Failed to update account" in m for m in messages)
    assert 1 not in mgr.email_processing_status


def test_cleanup_failure_logs_warning(monkeypatch, caplog):
    class DummyAccountManager:
        def list_accounts(self, include_password=False):
            return [
                {
                    "id": 1,
                    "account_name": "acc",
                    "server_type": "imap",
                    "server": "srv",
                    "port": 993,
                    "username": "u",
                    "password": "p",
                }
            ]

        def update_account(self, account_id, data):
            return None

    class FailingDict(dict):
        def __delitem__(self, key):
            raise KeyError(key)

    mgr = _make_manager(DummyAccountManager(), FailingDict({0: 0}))

    with caplog.at_level(logging.WARNING):
        mgr._refresh_email_account_background(1)

    messages = [r.getMessage() for r in caplog.records]
    assert any("No processing status found for account" in m for m in messages)
