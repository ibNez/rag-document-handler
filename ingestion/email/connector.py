from __future__ import annotations

"""Email connector implementations.

This module provides a base :class:`EmailConnector` abstract class and an
:class:`IMAPConnector` implementation capable of fetching emails from an IMAP
server and returning records in the canonical schema used by the project.

The implementation mirrors the logic demonstrated in the
``examples/Email-ETL-Process.ipynb`` notebook. Only lightweight parsing and
normalisation is performed – heavy processing such as summarisation or keyword
extraction is intentionally deferred to downstream stages.
"""

from abc import ABC, abstractmethod
from datetime import datetime, timezone
import ssl
from email import message_from_bytes
from email.header import decode_header, make_header
from email.message import Message
from email.utils import getaddresses, parsedate_to_datetime
import base64
import imaplib
import logging
import quopri
from typing import Any, Dict, Iterable, List, Optional

from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
try:  # pragma: no cover - optional dependency
    from exchangelib import Account, Configuration, Credentials as EWSCredentials, DELEGATE
except Exception:  # pragma: no cover - optional dependency
    Account = Configuration = EWSCredentials = DELEGATE = None  # type: ignore

logger = logging.getLogger(__name__)


class EmailConnector(ABC):
    """Abstract base class for fetching email records."""

    @abstractmethod
    def fetch_emails(self, since_date: Optional[datetime] = None) -> List[Dict[str, Any]]:
        """Fetch emails and return a list of canonical records.

        Parameters
        ----------
        since_date:
            If provided, only messages on or after this date are retrieved.
        """


class IMAPConnector(EmailConnector):
    """Retrieve emails from an IMAP server.

    When ``use_ssl`` is ``False`` the connector will automatically attempt to
    upgrade the connection using ``STARTTLS`` to avoid sending credentials in
    plaintext. If the server does not support ``STARTTLS`` a ``RuntimeError`` is
    raised.
    """

    def __init__(
        self,
        host: str,
        username: str,
        password: str,
        mailbox: str = "INBOX",
        *,
        port: int = 993,
        batch_limit: Optional[int] = 50,
        use_ssl: bool = True,
        primary_mailbox: Optional[str] = None,
    ) -> None:
        self.host = host
        self.port = port
        self.username = username
        self.password = password
        self.mailbox = mailbox
        self.batch_limit = batch_limit
        self.use_ssl = use_ssl
        self.primary_mailbox = (primary_mailbox or "").lower() or None

    # ------------------------------------------------------------------
    def fetch_emails(self, since_date: Optional[datetime] = None) -> List[Dict[str, Any]]:
        """Fetch emails via IMAP and return canonical records."""
        logger.info(
            "Connecting to IMAP server %s:%s mailbox=%s user=%s",
            self.host,
            self.port,
            self.mailbox,
            self.username,
        )
        conn = (
            imaplib.IMAP4_SSL(self.host, self.port)
            if self.use_ssl
            else imaplib.IMAP4(self.host, self.port)
        )
        try:
            if not self.use_ssl:
                try:
                    status, _ = conn.starttls(ssl_context=ssl.create_default_context())
                    if status != "OK":
                        raise imaplib.IMAP4.error("STARTTLS failed")
                except Exception as exc:  # pragma: no cover - defensive
                    raise RuntimeError(
                        "IMAP server requires a secure connection; STARTTLS negotiation failed"
                    ) from exc

            status, _ = conn.login(self.username, self.password)
            if status != "OK":
                logger.warning(
                    "IMAP login failed for user %s: status=%s", self.username, status
                )
                return []

            status, _ = conn.select(self.mailbox)
            if status != "OK":
                logger.warning(
                    "IMAP select failed for mailbox %s: status=%s", self.mailbox, status
                )
                return []

            criteria: List[str] = []
            if since_date is not None:
                criteria.append(f'SINCE "{since_date.strftime("%d-%b-%Y")}"')
            search_query = "ALL" if not criteria else " ".join(criteria)
            status, messages = conn.search(None, search_query)
            if status != "OK":
                logger.warning("IMAP search failed: status=%s", status)
                return []
            email_ids = messages[0].split()
            if self.batch_limit is not None:
                email_ids = email_ids[-self.batch_limit :]

            results: List[Dict[str, Any]] = []
            for eid in email_ids:
                status, msg_data = conn.fetch(eid, "(RFC822)")
                if status != "OK" or not msg_data or not msg_data[0]:
                    continue
                try:
                    msg = message_from_bytes(msg_data[0][1])
                    rec = self._parse_email(msg)
                    rec["server_type"] = "imap"
                    results.append(rec)
                except Exception as exc:  # pragma: no cover - defensive
                    decoded_eid = eid.decode() if isinstance(eid, bytes) else str(eid)
                    logger.warning(
                        "Failed to parse message %s: %s", decoded_eid, exc
                    )
                    results.append(
                        {
                            "message_id": None,
                            "thread_id": None,
                            "subject": None,
                            "from_addr": None,
                            "to_addrs": [],
                            "cc_addrs": [],
                            "date_utc": None,
                            "received_utc": None,
                            "in_reply_to": None,
                            "references_ids": [],
                            "is_reply": 0,
                            "is_forward": 0,
                            "raw_size_bytes": None,
                            "body_text": None,
                            "body_html": None,
                            "language": None,
                            "has_attachments": 0,
                            "attachment_manifest": [],
                            "processed": 0,
                            "ingested_at": None,
                            "updated_at": None,
                            "content_hash": None,
                            "summary": None,
                            "keywords": None,
                            "auto_topic": None,
                            "manual_topic": None,
                            "topic_confidence": None,
                            "topic_version": None,
                            "error_state": f"parse_error: {exc.__class__.__name__}",
                            "direction": None,
                            "participants": [],
                            "participants_hash": None,
                            "to_primary": None,
                            "server_type": "imap",
                        }
                    )

            logger.info(
                "Retrieved %d emails from IMAP server %s for user %s",
                len(results),
                self.host,
                self.username,
            )
            return results
        finally:
            try:
                conn.logout()
            except Exception as exc:  # pragma: no cover - defensive
                logger.warning("Error during IMAP logout: %s", exc)

    # ------------------------------------------------------------------
    def _decode_header_value(self, raw_val: Optional[str]) -> Optional[str]:
        if not raw_val:
            return None
        try:
            return str(make_header(decode_header(raw_val))).strip()
        except Exception:
            parts = decode_header(raw_val)
            decoded: List[str] = []
            for text, enc in parts:
                if isinstance(text, bytes):
                    try:
                        decoded.append(text.decode(enc or "utf-8", errors="ignore"))
                    except Exception:
                        decoded.append(text.decode("utf-8", errors="ignore"))
                else:
                    decoded.append(text)
            return "".join(decoded).strip()

    # ------------------------------------------------------------------
    def _parse_email(self, msg: Message) -> Dict[str, Any]:
        subject = self._decode_header_value(msg.get("Subject")) or None
        message_id = (msg.get("Message-ID") or "").strip() or None
        in_reply_to = (msg.get("In-Reply-To") or "").strip() or None
        references_raw = msg.get("References") or ""
        references_ids = [r.strip("<> ") for r in references_raw.split() if "@" in r] if references_raw else []
        date_raw = msg.get("Date")
        date_utc: Optional[str] = None
        if date_raw:
            try:
                dt = parsedate_to_datetime(date_raw)
                if dt and dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                date_utc = dt.astimezone(timezone.utc).isoformat()
            except Exception:
                date_utc = None
        received_utc = date_utc  # placeholder

        raw_from = msg.get("From")
        from_addr = None
        if raw_from:
            addrs = getaddresses([raw_from])
            if addrs:
                from_addr = (addrs[0][1] or addrs[0][0]).lower()
        to_addrs = [addr.lower() for _, addr in getaddresses([msg.get("To") or ""]) if addr]
        cc_addrs = [addr.lower() for _, addr in getaddresses([msg.get("Cc") or ""]) if addr]

        subj_lc = (subject or "").lower()
        is_reply = 1 if (in_reply_to or subj_lc.startswith("re:")) else 0
        is_forward = 1 if (subj_lc.startswith("fwd:") or subj_lc.startswith("fw:")) else 0

        body_text: Optional[str] = None
        body_html: Optional[str] = None
        attachment_manifest: List[Dict[str, Any]] = []
        has_attachments = 0
        raw_size_bytes: Optional[int] = None

        if msg.is_multipart():
            for part in msg.walk():
                ctype = part.get_content_type()
                disp = (part.get("Content-Disposition") or "").lower()
                if ctype == "text/plain" and "attachment" not in disp and body_text is None:
                    try:
                        body_text = self._decode_part(part)
                    except Exception:
                        body_text = None
                elif ctype == "text/html" and "attachment" not in disp and body_html is None:
                    try:
                        body_html = self._decode_part(part)
                    except Exception:
                        body_html = None
                elif "attachment" in disp or part.get_filename():
                    has_attachments = 1
                    fname = self._decode_header_value(part.get_filename()) if part.get_filename() else None
                    payload = part.get_payload(decode=True) or b""
                    attachment_manifest.append(
                        {
                            "filename": fname,
                            "size": len(payload),
                            "mime": ctype,
                        }
                    )
            size_acc = 0
            for part in msg.walk():
                try:
                    pl = part.get_payload(decode=True)
                    if pl:
                        size_acc += len(pl)
                except Exception:
                    pass
            raw_size_bytes = size_acc or None
        else:
            try:
                body_text = self._decode_part(msg)
            except Exception:
                body_text = None
            payload = msg.get_payload(decode=True) or b""
            raw_size_bytes = len(payload) if payload else None

        participants = sorted({p for p in ([from_addr] if from_addr else []) + to_addrs + cc_addrs})

        record = {
            "message_id": message_id,
            "thread_id": self._derive_thread_id(message_id, in_reply_to, references_ids),
            "subject": subject,
            "from_addr": from_addr,
            "to_addrs": to_addrs,
            "cc_addrs": cc_addrs,
            "date_utc": date_utc,
            "received_utc": received_utc,
            "in_reply_to": in_reply_to or None,
            "references_ids": references_ids,
            "is_reply": is_reply,
            "is_forward": is_forward,
            "raw_size_bytes": raw_size_bytes,
            "body_text": body_text,
            "body_html": body_html,
            "language": None,
            "has_attachments": has_attachments,
            "attachment_manifest": attachment_manifest,
            "processed": 0,
            "ingested_at": None,
            "updated_at": None,
            "content_hash": None,
            "summary": None,
            "keywords": None,
            "auto_topic": None,
            "manual_topic": None,
            "topic_confidence": None,
            "topic_version": None,
            "error_state": None,
            "direction": None,
            "participants": participants,
            "participants_hash": None,
            "to_primary": self.primary_mailbox if (self.primary_mailbox and self.primary_mailbox in to_addrs) else None,
        }
        return record

    # ------------------------------------------------------------------
    def _decode_part(self, part: Message) -> Optional[str]:
        charset = part.get_content_charset() or "utf-8"
        payload = part.get_payload(decode=True)
        if not payload:
            return None
        try:
            return payload.decode(charset, errors="ignore")
        except Exception:
            try:
                return quopri.decodestring(payload).decode("utf-8", errors="ignore")
            except Exception:
                try:
                    return base64.b64decode(payload).decode("utf-8", errors="ignore")
                except Exception:
                    return payload.decode("utf-8", errors="ignore")

    # ------------------------------------------------------------------
    def _derive_thread_id(
        self,
        message_id: Optional[str],
        in_reply_to: Optional[str],
        references_ids: Iterable[str],
    ) -> Optional[str]:
        if references_ids:
            return next(iter(references_ids))
        if in_reply_to:
            return in_reply_to
        return message_id


class ExchangeConnector(EmailConnector):
    """Retrieve emails from an Exchange server using EWS."""

    _decode_header_value = IMAPConnector._decode_header_value
    _decode_part = IMAPConnector._decode_part
    _derive_thread_id = IMAPConnector._derive_thread_id
    _parse_email = IMAPConnector._parse_email

    def __init__(
        self,
        server: str,
        username: str,
        password: str,
        *,
        batch_limit: Optional[int] = 50,
    ) -> None:
        if Account is None:
            raise ImportError("exchangelib is required for ExchangeConnector")
        creds = EWSCredentials(username=username, password=password)
        config = Configuration(server=server, credentials=creds)
        self.account = Account(
            primary_smtp_address=username,
            config=config,
            autodiscover=False,
            access_type=DELEGATE,
        )
        self.batch_limit = batch_limit
        self.primary_mailbox = None
        self.server = server
        self.username = username

    # ------------------------------------------------------------------
    def fetch_emails(self, since_date: Optional[datetime] = None) -> List[Dict[str, Any]]:
        """Fetch emails via Exchange Web Services and return canonical records."""
        logger.info(
            "Fetching emails from Exchange server %s for user %s",
            self.server,
            self.username,
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
            "Retrieved %d emails from Exchange server %s for user %s",
            len(results),
            self.server,
            self.username,
        )
        return results


class GmailConnector(EmailConnector):
    """Retrieve emails using the Gmail API."""

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
