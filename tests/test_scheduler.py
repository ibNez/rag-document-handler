"""Tests for unified scheduler triggering URL and email tasks."""

from __future__ import annotations

import os
import sys
import types
import threading
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
sys.modules.setdefault("langchain_ollama", _ollama)
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


def test_scheduler_triggers_url_and_email(monkeypatch):
    """Scheduler should start both URL and email tasks when due."""

    url_event = threading.Event()
    email_event = threading.Event()

    # Run background tasks synchronously
    class DummyThread:
        def __init__(self, target, args=()):
            self.target = target
            self.args = args
            self.daemon = True

        def start(self):
            self.target(*self.args)

    monkeypatch.setattr(app, "threading", types.SimpleNamespace(Thread=DummyThread))

    # Deterministic due records
    class DummyURLManager:
        def get_due_urls(self):
            return [{"id": 1, "url": "http://example.com"}]

        def get_url_count(self):
            return 1

    class DummyEmailOrchestrator:
        def get_due_accounts(self):
            return [{"id": 2, "account_name": "acc"}]

    mgr = object.__new__(app.RAGKnowledgebaseManager)
    mgr.config = types.SimpleNamespace(SCHEDULER_POLL_SECONDS_BUSY=0, SCHEDULER_POLL_SECONDS_IDLE=0)
    mgr.url_manager = DummyURLManager()
    mgr.email_orchestrator = DummyEmailOrchestrator()
    mgr.url_processing_status = {}
    mgr.email_processing_status = {}
    mgr._scheduler_last_cycle = None

    mgr._process_url_background = lambda url_id: url_event.set()
    mgr._refresh_email_account_background = lambda acct_id: email_event.set()

    class SchedulerExit(BaseException):
        pass

    def fake_sleep(_):
        raise SchedulerExit()

    # Patch time module used by scheduler
    monkeypatch.setattr(app, "time", types.SimpleNamespace(time=lambda: 0, sleep=fake_sleep))

    with pytest.raises(SchedulerExit):
        mgr._scheduler_loop()

    assert url_event.is_set()
    assert email_event.is_set()
