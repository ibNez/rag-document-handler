"""Tests for email account management routes."""

import os
import sys
from typing import Any, Dict
import types
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Stub external dependencies used by app module to avoid heavy installs
sys.modules.setdefault("pypdf", types.SimpleNamespace(PdfReader=lambda *a, **k: None))
sys.modules.setdefault("docx", types.SimpleNamespace(Document=lambda *a, **k: None))
sys.modules.setdefault("langchain_community.document_loaders", types.SimpleNamespace(PyPDFLoader=None))
sys.modules.setdefault("langchain_unstructured", types.SimpleNamespace(UnstructuredLoader=None))

class _DummySplitter:
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        pass

    def split_text(self, text: str) -> list[str]:
        return [text]

sys.modules.setdefault(
    "langchain_text_splitters", types.SimpleNamespace(RecursiveCharacterTextSplitter=_DummySplitter)
)
sys.modules.setdefault(
    "pymilvus",
    types.SimpleNamespace(
        connections=types.SimpleNamespace(connect=lambda *a, **k: None),
        utility=types.SimpleNamespace(get_server_version=lambda: ""),
        Collection=lambda *a, **k: None,
    ),
)
sys.modules.setdefault(
    "langchain_ollama",
    types.SimpleNamespace(OllamaEmbeddings=lambda *a, **k: None, ChatOllama=lambda *a, **k: None),
)
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

import app as app_module
from app import RAGKnowledgebaseManager

class DummyMilvusManager:
    """Lightweight stand-in for the real Milvus manager."""

    def __init__(self, config: Any) -> None:
        pass

    def get_collection_stats(self) -> Dict[str, Any]:
        return {}

    def check_connection(self) -> Dict[str, Any]:
        return {"connected": True}

    def delete_document(self, filename: str | None = None, document_id: str | None = None) -> Dict[str, Any]:
        return {"success": True}

class DummyEmailOrchestrator:
    """Dummy email orchestrator used in tests."""

    def __init__(self, config: Any) -> None:
        self.config = config

    def start(self) -> None:  # pragma: no cover - simple stub
        return None

@pytest.fixture
def client(tmp_path, monkeypatch):
    """Return a Flask test client with dependencies stubbed."""
    url_manager_cls = app_module.URLManager
    monkeypatch.setattr(app_module, "MilvusManager", DummyMilvusManager)
    monkeypatch.setattr(app_module, "EmailOrchestrator", DummyEmailOrchestrator)
    monkeypatch.setattr(RAGKnowledgebaseManager, "_start_scheduler", lambda self: None)

    def _url_manager_factory() -> app_module.URLManager:
        return url_manager_cls(db_path=str(tmp_path / "test.db"))

    monkeypatch.setattr(app_module, "URLManager", _url_manager_factory)
    mgr = RAGKnowledgebaseManager()
    return mgr.app.test_client()

def test_email_account_crud(client):
    response = client.post(
        "/email_accounts",
        data={
            "account_name": "Work",
            "server": "imap.example.com",
            "username": "user",
            "password": "pass",
            "port": "993",
            "mailbox": "INBOX",
            "batch_limit": "10",
            "use_ssl": "1",
        },
    )
    assert response.status_code == 302

    list_resp = client.get("/email_accounts")
    accounts = list_resp.get_json()
    assert isinstance(accounts, list)
    assert len(accounts) == 1
    account_id = accounts[0]["id"]

    response = client.post(f"/email_accounts/{account_id}", data={"username": "new"})
    assert response.status_code == 302

    list_resp = client.get("/email_accounts")
    accounts = list_resp.get_json()
    assert accounts[0]["username"] == "new"

    response = client.post(f"/email_accounts/{account_id}/delete")
    assert response.status_code == 302

    list_resp = client.get("/email_accounts")
    assert list_resp.get_json() == []
