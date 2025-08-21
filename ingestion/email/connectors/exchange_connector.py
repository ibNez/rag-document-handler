"""Exchange email connector implementation.

This module provides the :class:`ExchangeConnector` implementation for 
fetching emails from an Exchange server using EWS (Exchange Web Services).
"""

from __future__ import annotations

import logging
from datetime import datetime
from email import message_from_bytes
from typing import Any, Dict, List, Optional

from .base import EmailConnector
from .imap_connector import IMAPConnector

try:  # pragma: no cover - optional dependency
    from exchangelib import Account, Configuration, Credentials as EWSCredentials, DELEGATE
except Exception:  # pragma: no cover - optional dependency
    Account = Configuration = EWSCredentials = DELEGATE = None  # type: ignore

logger = logging.getLogger(__name__)


class ExchangeConnector(EmailConnector):
    """Retrieve emails from an Exchange server using EWS."""

    # Reuse email parsing utilities from IMAP connector
    _decode_header_value = IMAPConnector._decode_header_value
    _decode_part = IMAPConnector._decode_part
    _derive_thread_id = IMAPConnector._derive_thread_id
    _parse_email = IMAPConnector._parse_email

    def __init__(
        self,
        server: str,
        email_address: str,
        password: str,
        *,
        batch_limit: Optional[int] = 50,
    ) -> None:
        if Account is None:
            raise ImportError("exchangelib is required for ExchangeConnector")
        creds = EWSCredentials(username=email_address, password=password)
        config = Configuration(server=server, credentials=creds)
        self.account = Account(
            primary_smtp_address=email_address,
            config=config,
            autodiscover=False,
            access_type=DELEGATE,
        )
        self.batch_limit = batch_limit
        self.primary_mailbox = None
        self.server = server
        self.email_address = email_address

    # ------------------------------------------------------------------
    def fetch_emails(self, since_date: Optional[datetime] = None) -> List[Dict[str, Any]]:
        """Fetch emails via Exchange Web Services and return canonical records."""
        logger.info(
            "Fetching emails from Exchange server %s for %s",
            self.server,
            self.email_address,
        )
        qs = self.account.inbox.all().order_by("-datetime_received")
        if since_date is not None:
            qs = qs.filter(datetime_received__gte=since_date)
        if self.batch_limit is not None:
            qs = qs[: self.batch_limit]

        results: List[Dict[str, Any]] = []
        for item in qs:
            try:
                msg = message_from_bytes(item.mime_content)
                rec = self._parse_email(msg)
                rec["server_type"] = "exchange"
                results.append(rec)
            except Exception:
                continue
        logger.info(
            "Retrieved %d emails from Exchange server %s for %s",
            len(results),
            self.server,
            self.email_address,
        )
        return results
