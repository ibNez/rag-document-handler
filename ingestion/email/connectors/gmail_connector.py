"""Gmail API email connector implementation.

This module provides the :class:`GmailConnector` implementation for 
fetching emails using the Gmail API.
"""

from __future__ import annotations

import base64
import logging
from datetime import datetime
from email import message_from_bytes
from typing import Any, Dict, List, Optional

from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

from .base import EmailConnector
from .imap_connector import IMAPConnector

logger = logging.getLogger(__name__)


class GmailConnector(EmailConnector):
    """Retrieve emails using the Gmail API."""

    # Reuse email parsing utilities from IMAP connector
    _decode_header_value = IMAPConnector._decode_header_value
    _decode_part = IMAPConnector._decode_part
    _derive_thread_id = IMAPConnector._derive_thread_id
    _parse_email = IMAPConnector._parse_email

    def __init__(
        self,
        credentials: Optional[Credentials] = None,
        token_path: Optional[str] = None,
        *,
        user_id: str = "me",
        batch_limit: Optional[int] = 50,
    ) -> None:
        if credentials is None and token_path is None:
            raise ValueError("Either credentials or token_path must be provided")
        if credentials is None and token_path is not None:
            credentials = Credentials.from_authorized_user_file(token_path)
        self.creds = credentials
        self.service = build("gmail", "v1", credentials=credentials)
        self.user_id = user_id
        self.batch_limit = batch_limit

    # ------------------------------------------------------------------
    def fetch_emails(self, since_date: Optional[datetime] = None) -> List[Dict[str, Any]]:
        """Fetch emails via Gmail API and return canonical records."""
        logger.info("Fetching emails from Gmail for user %s", self.user_id)
        results: List[Dict[str, Any]] = []
        page_token: Optional[str] = None
        fetched = 0
        query = None
        if since_date is not None:
            query = f"after:{since_date.strftime('%Y/%m/%d')}"
        try:
            while True:
                list_kwargs: Dict[str, Any] = {"userId": self.user_id, "maxResults": 500}
                if query:
                    list_kwargs["q"] = query
                if page_token:
                    list_kwargs["pageToken"] = page_token
                response = self.service.users().messages().list(**list_kwargs).execute()
                messages = response.get("messages", [])
                for meta in messages:
                    if self.batch_limit is not None and fetched >= self.batch_limit:
                        return results
                    try:
                        msg_data = (
                            self.service.users()
                            .messages()
                            .get(userId=self.user_id, id=meta["id"], format="raw")
                            .execute()
                        )
                    except HttpError:
                        continue
                    try:
                        raw = base64.urlsafe_b64decode(msg_data.get("raw", "").encode("utf-8"))
                        msg = message_from_bytes(raw)
                        rec = self._parse_email(msg)
                        rec["server_type"] = "gmail"
                        results.append(rec)
                    except Exception:
                        continue
                    fetched += 1
                page_token = response.get("nextPageToken")
                if not page_token or (
                    self.batch_limit is not None and fetched >= self.batch_limit
                ):
                    break
        except HttpError as exc:
            logger.warning("Gmail API error: %s", exc)
            return []
        logger.info(
            "Retrieved %d emails from Gmail for user %s", len(results), self.user_id
        )
        return results
