"""Tests for GmailConnector fetching and normalization."""

from __future__ import annotations

import base64
import os
import sys
from datetime import datetime
from typing import Any
from unittest.mock import MagicMock

import pytest

import types

# Stub google modules to avoid external dependencies in tests
sys.modules.setdefault("google.oauth2.credentials", types.SimpleNamespace(Credentials=object))
sys.modules.setdefault("googleapiclient.discovery", types.SimpleNamespace(build=lambda *a, **k: MagicMock()))
sys.modules.setdefault("googleapiclient.errors", types.SimpleNamespace(HttpError=Exception))
sys.modules.setdefault("google.auth.transport.requests", types.SimpleNamespace(Request=object))
sys.modules.setdefault("cryptography.fernet", types.SimpleNamespace(Fernet=object))

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import importlib.util

connector_path = os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")), "ingestion", "email", "connector.py")
spec = importlib.util.spec_from_file_location("gmail_connector", connector_path)
connector_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(connector_module)
GmailConnector = connector_module.GmailConnector

from tests.gmail_fixtures import gmail_list_response, gmail_get_response


@pytest.mark.usefixtures("gmail_list_response", "gmail_get_response")
def test_gmail_fetch_emails_returns_canonical_records(monkeypatch: pytest.MonkeyPatch, gmail_list_response: dict[str, Any], gmail_get_response: dict[str, Any]) -> None:
    """``GmailConnector.fetch_emails`` should convert Gmail API responses to canonical records."""
    service = MagicMock()
    users = service.users.return_value
    messages = users.messages.return_value
    messages.list.return_value.execute.return_value = gmail_list_response
    messages.get.return_value.execute.return_value = gmail_get_response

    monkeypatch.setattr(connector_module, "build", lambda *a, **k: service)

    connector = GmailConnector(credentials=MagicMock(), user_id="me", batch_limit=10)
    connector.primary_mailbox = None

    since = datetime(2024, 1, 1)
    records = connector.fetch_emails(since_date=since)

    # ensure query parameter included for since_date filtering
    messages.list.assert_called_with(userId="me", maxResults=500, q="after:2024/01/01")

    assert len(records) == 1
    record = records[0]
    assert record["from_addr"] == "alice@example.com"
    assert record["to_addrs"] == ["bob@example.com"]
    assert record["body_text"].strip() == "Hi Bob"
    assert record["has_attachments"] == 1
    assert record["attachment_manifest"][0]["filename"] == "note.txt"
    assert record["server_type"] == "gmail"
