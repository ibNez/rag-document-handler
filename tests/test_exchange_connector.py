"""Tests for ExchangeConnector fetching and normalization."""

from __future__ import annotations

import os
import sys
import types
from typing import Any

import pytest

# Stub external dependencies to avoid heavy installs
sys.modules.setdefault("google.oauth2.credentials", types.SimpleNamespace(Credentials=object))
sys.modules.setdefault("googleapiclient.discovery", types.SimpleNamespace(build=lambda *a, **k: None))
sys.modules.setdefault("googleapiclient.errors", types.SimpleNamespace(HttpError=Exception))
sys.modules.setdefault(
    "exchangelib",
    types.SimpleNamespace(Account=object, Configuration=object, Credentials=object, DELEGATE=object),
)

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import importlib.util

connector_path = os.path.join(
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..")),
    "ingestion",
    "email",
    "connector.py",
)
spec = importlib.util.spec_from_file_location("exchange_connector", connector_path)
connector_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(connector_module)
ExchangeConnector = connector_module.ExchangeConnector


def test_exchange_fetch_emails_returns_canonical_records(monkeypatch: pytest.MonkeyPatch) -> None:
    raw_email = (
        "Subject: Hello\r\n"
        "From: Alice <alice@example.com>\r\n"
        "To: Bob <bob@example.com>\r\n"
        "Date: Mon, 01 Jan 2024 00:00:00 +0000\r\n"
        "Message-ID: <e1@example.com>\r\n"
        "\r\n"
        "Hi Bob\r\n"
    ).encode("utf-8")

    item = types.SimpleNamespace(mime_content=raw_email)

    class FakeQuery:
        def __init__(self, items: list[Any]) -> None:
            self.items = items

        def all(self) -> "FakeQuery":
            return self

        def order_by(self, *args: Any, **kwargs: Any) -> "FakeQuery":
            return self

        def __getitem__(self, key: slice) -> list[Any]:
            return self.items[key]

        def __iter__(self):
            return iter(self.items)

    fake_account = types.SimpleNamespace(inbox=FakeQuery([item]))

    monkeypatch.setattr(connector_module, "Account", lambda *a, **k: fake_account)
    monkeypatch.setattr(connector_module, "EWSCredentials", lambda *a, **k: None)
    monkeypatch.setattr(connector_module, "Configuration", lambda *a, **k: None)

    connector = ExchangeConnector(
        server="ex.example.com",
        email_address="user@example.com",
        password="pass",
        batch_limit=10,
    )
    records = connector.fetch_emails()
    assert len(records) == 1
    record = records[0]
    assert record["from_addr"] == "alice@example.com"
    assert record["to_addrs"] == ["bob@example.com"]
    assert record["server_type"] == "exchange"
