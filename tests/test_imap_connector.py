"""Tests for the :class:`IMAPConnector`."""

from __future__ import annotations

from typing import Any
from types import SimpleNamespace

import sys
import os

import pytest

# Ensure project root on path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import importlib.util

connector_path = os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")), "ingestion", "email", "connector.py")
spec = importlib.util.spec_from_file_location("imap_connector", connector_path)
connector_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(connector_module)
IMAPConnector = connector_module.IMAPConnector


class DummyIMAP4:
    """Simple IMAP4 stub that records method calls."""

    def __init__(self, host: str, port: int) -> None:
        self.host = host
        self.port = port
        self.calls: list[str] = []

    def starttls(self, *_, **__):  # pragma: no cover - interface stub
        self.calls.append("starttls")
        return "OK", b""

    def login(self, *_: Any, **__: Any) -> None:  # pragma: no cover - interface stub
        self.calls.append("login")

    def select(self, *_: Any, **__: Any):  # pragma: no cover - interface stub
        return "OK", [b"0"]

    def search(self, *_: Any, **__: Any):  # pragma: no cover - interface stub
        return "OK", [b""]

    def logout(self) -> None:  # pragma: no cover - interface stub
        self.calls.append("logout")


class DummyIMAP4Fail(DummyIMAP4):
    """IMAP stub whose ``starttls`` call fails."""

    def starttls(self, *_, **__):  # pragma: no cover - interface stub
        raise connector_module.imaplib.IMAP4.error("STARTTLS not supported")


@pytest.fixture
def patch_imap(monkeypatch: pytest.MonkeyPatch) -> DummyIMAP4:
    """Patch ``imaplib.IMAP4`` to use a dummy implementation."""
    dummy = DummyIMAP4("host", 143)
    monkeypatch.setattr(
        connector_module,
        "imaplib",
        SimpleNamespace(IMAP4=lambda *a, **k: dummy, IMAP4_SSL=object),
    )
    return dummy


def test_imap_connector_uses_starttls(patch_imap: DummyIMAP4) -> None:
    """Connector should upgrade plain connections using ``STARTTLS``."""
    connector = IMAPConnector(
        host="host",
        username="user",
        password="pw",
        use_ssl=False,
    )
    connector.fetch_emails()
    assert "starttls" in patch_imap.calls
    assert patch_imap.calls.index("starttls") < patch_imap.calls.index("login")


def test_imap_connector_starttls_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    """A failing ``STARTTLS`` handshake should raise ``RuntimeError``."""
    dummy = DummyIMAP4Fail("host", 143)
    monkeypatch.setattr(
        connector_module,
        "imaplib",
        SimpleNamespace(IMAP4=lambda *a, **k: dummy, IMAP4_SSL=object),
    )
    connector = IMAPConnector(
        host="host",
        username="user",
        password="pw",
        use_ssl=False,
    )
    with pytest.raises(RuntimeError):
        connector.fetch_emails()
