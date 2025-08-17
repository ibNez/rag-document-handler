"""Tests for email account management routes and manager CRUD."""

import os
import sqlite3
import sys
from typing import Any, Dict
import types
import pytest
from cryptography.fernet import Fernet

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Stub external dependencies used by ingestion and app modules to avoid heavy installs
sys.modules.setdefault("pypdf", types.SimpleNamespace(PdfReader=lambda *a, **k: None))
sys.modules.setdefault("docx", types.SimpleNamespace(Document=lambda *a, **k: None))
sys.modules.setdefault("langchain_community.document_loaders", types.SimpleNamespace(PyPDFLoader=None))
sys.modules.setdefault("langchain_unstructured", types.SimpleNamespace(UnstructuredLoader=None))
sys.modules.setdefault("dotenv", types.SimpleNamespace(load_dotenv=lambda *a, **k: None))
sys.modules.setdefault("google.oauth2.credentials", types.SimpleNamespace(Credentials=object))
sys.modules.setdefault("googleapiclient.discovery", types.SimpleNamespace(build=lambda *a, **k: None))
sys.modules.setdefault("googleapiclient.errors", types.SimpleNamespace(HttpError=Exception))
sys.modules.setdefault("google.auth.transport.requests", types.SimpleNamespace(Request=object))


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
sys.modules["langchain_ollama"] = types.SimpleNamespace(
    OllamaEmbeddings=lambda *a, **k: None, ChatOllama=lambda *a, **k: None
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

from ingestion.email.account_manager import EmailAccountManager


@pytest.fixture
def manager(monkeypatch: pytest.MonkeyPatch) -> EmailAccountManager:
    """Return an ``EmailAccountManager`` backed by an in-memory database."""
    monkeypatch.setenv("EMAIL_ENCRYPTION_KEY", Fernet.generate_key().decode())
    conn = sqlite3.connect(":memory:")
    return EmailAccountManager(conn)


def test_email_account_manager_crud(manager: EmailAccountManager) -> None:
    """CRUD operations on ``EmailAccountManager`` should persist changes."""
    record = {
        "account_name": "Work",
        "server_type": "imap",
        "server": "imap.example.com",
        "port": 993,
        "username": "user",
        "password": "pass",
        "mailbox": "INBOX",
        "batch_limit": 10,
        "use_ssl": 1,
        "refresh_interval_minutes": 5,
    }

    account_id = manager.create_account(record)
    accounts = manager.list_accounts()
    assert len(accounts) == 1
    assert accounts[0]["account_name"] == "Work"
    assert accounts[0]["refresh_interval_minutes"] == 5
    assert "last_update_status" in accounts[0]
    assert accounts[0]["last_update_status"] is None

    # Password should not be returned by default
    assert "password" not in accounts[0]

    # Decrypted password should be available when requested
    accounts_with_pw = manager.list_accounts(include_password=True)
    assert accounts_with_pw[0]["password"] == "pass"

    # Database should store an encrypted value
    cur = manager.conn.cursor()
    cur.execute("SELECT password FROM email_accounts WHERE id = ?", (account_id,))
    stored = cur.fetchone()[0]
    assert stored != "pass"

    manager.update_account(account_id, {"username": "new"})
    accounts = manager.list_accounts(include_password=True)
    assert accounts[0]["username"] == "new"

    manager.delete_account(account_id)
    assert manager.list_accounts() == []

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

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.config = args[0] if args else None

    def start(self) -> None:  # pragma: no cover - simple stub
        return None


@pytest.fixture
def client(tmp_path, monkeypatch):
    """Return a Flask test client with dependencies stubbed."""
    monkeypatch.setenv("EMAIL_ENCRYPTION_KEY", Fernet.generate_key().decode())
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
            "server_type": "imap",
            "server": "imap.example.com",
            "username": "user",
            "password": "pass",
            "port": "993",
            "mailbox": "INBOX",
            "batch_limit": "10",
            "use_ssl": "on",
            "refresh_interval_minutes": "5",
        },
    )
    assert response.status_code == 302

    list_resp = client.get("/email_accounts")
    accounts = list_resp.get_json()
    assert isinstance(accounts, list)
    assert len(accounts) == 1
    account_id = accounts[0]["id"]
    assert "password" not in accounts[0]
    assert accounts[0]["refresh_interval_minutes"] == 5
    assert accounts[0]["use_ssl"] == 1
    assert "last_update_status" in accounts[0]
    assert accounts[0]["last_update_status"] is None

    response = client.post(f"/email_accounts/{account_id}", data={"username": "new"})
    assert response.status_code == 302

    list_resp = client.get("/email_accounts")
    accounts = list_resp.get_json()
    assert accounts[0]["username"] == "new"

    response = client.post(f"/email_accounts/{account_id}/delete")
    assert response.status_code == 302

    list_resp = client.get("/email_accounts")
    assert list_resp.get_json() == []

def test_email_status_endpoint(client):
    response = client.post(
        "/email_accounts",
        data={
            "account_name": "Work",
            "server_type": "imap",
            "server": "imap.example.com",
            "username": "user",
            "password": "pass",
            "port": "993",
        },
    )
    assert response.status_code == 302

    list_resp = client.get("/email_accounts")
    account_id = list_resp.get_json()[0]["id"]

    status_resp = client.get(f"/email_status/{account_id}")
    data = status_resp.get_json()
    assert data["status"] == "not_found"
    assert data.get("last_update_status") is None
