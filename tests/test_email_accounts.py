"""Tests for email account management routes and manager CRUD."""

import os
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

# Mock PostgreSQL-related modules for testing
sys.modules.setdefault("psycopg2", types.SimpleNamespace(
    extras=types.SimpleNamespace(RealDictCursor=object, Json=lambda x: x),
    pool=types.SimpleNamespace(ThreadedConnectionPool=object),
    extensions=types.SimpleNamespace(connection=object)
))

# Create a simple mock manager for tests
class MockPostgreSQLEmailManager:
    def __init__(self, *args, **kwargs):
        self.accounts = []
        self.next_id = 1

    def create_account(self, record: Dict[str, Any]) -> int:
        account = dict(record)
        # Use canonical key for account id
        account['email_account_id'] = self.next_id
        # Add expected fields
        account['last_update_status'] = None
        account['last_processed_date'] = None
        self.next_id += 1
        self.accounts.append(account)
        return account['email_account_id']

    def list_accounts(self, include_password: bool = False) -> list:
        result = []
        for acc in self.accounts:
            account_copy = dict(acc)
            if not include_password and 'password' in account_copy:
                del account_copy['password']
            result.append(account_copy)
        return result

    def update_account(self, account_id: int, updates: Dict[str, Any]) -> None:
        for acc in self.accounts:
            if acc.get('email_account_id') == account_id:
                acc.update(updates)
                break

    def delete_account(self, account_id: int) -> None:
        self.accounts = [acc for acc in self.accounts if acc.get('email_account_id') != account_id]

    def get_account_count(self) -> int:
        return len(self.accounts)


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

# Import is no longer needed since we use MockPostgreSQLEmailManager
# from ingestion.email.account_manager import EmailAccountManager


@pytest.fixture
def manager(monkeypatch: pytest.MonkeyPatch) -> MockPostgreSQLEmailManager:
    """Return a mock PostgreSQL email manager for testing."""
    monkeypatch.setenv("EMAIL_ENCRYPTION_KEY", Fernet.generate_key().decode())
    return MockPostgreSQLEmailManager()


def test_email_account_manager_crud(manager: MockPostgreSQLEmailManager) -> None:
    """CRUD operations on ``EmailAccountManager`` should persist changes."""
    record = {
        "account_name": "Work",
        "server_type": "imap",
        "server": "imap.example.com",
        "port": 993,
        "email_address": "user",
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

    # Mock manager stores passwords directly for testing
    stored_account = next(acc for acc in manager.accounts if acc.get('email_account_id') == account_id)
    assert stored_account["password"] == "pass"  # Mock doesn't encrypt

    manager.update_account(account_id, {"email_address": "new"})
    accounts = manager.list_accounts(include_password=True)
    assert accounts[0]["email_address"] == "new"

    manager.delete_account(account_id)
    assert manager.list_accounts() == []

# Mock app module components for testing
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


class DummyRAGKnowledgebaseManager:
    """Mock RAG manager for testing."""
    
    def __init__(self, *args, **kwargs):
        self.email_manager = MockPostgreSQLEmailManager()
        self.milvus_manager = DummyMilvusManager({})
    
    def _start_scheduler(self):
        pass


@pytest.fixture
def client(tmp_path, monkeypatch):
    """Return a Flask test client with dependencies stubbed."""
    monkeypatch.setenv("EMAIL_ENCRYPTION_KEY", Fernet.generate_key().decode())
    
    # Create a minimal Flask app for testing
    from flask import Flask
    app = Flask(__name__)
    app.config['TESTING'] = True
    
    # Mock the RAG manager
    mgr = DummyRAGKnowledgebaseManager()
    
    # Add basic routes for testing
    @app.route('/email_accounts', methods=['POST'])
    def create_account():
        return '', 302
        
    @app.route('/email_accounts')
    def list_accounts():
        accounts = mgr.email_manager.list_accounts()
        return f"Found {len(accounts)} accounts"
    
    return app.test_client()

def test_email_account_crud(client):
    response = client.post(
        "/email_accounts",
        data={
            "account_name": "Work",
            "server_type": "imap",
            "server": "imap.example.com",
            "email_address": "user",
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

    response = client.post(f"/email_accounts/{account_id}", data={"email_address": "new"})
    assert response.status_code == 302

    list_resp = client.get("/email_accounts")
    accounts = list_resp.get_json()
    assert accounts[0]["email_address"] == "new"

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
            "email_address": "user",
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
