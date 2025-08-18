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
        email_address: str,
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
        self.email_address = email_address
        self.password = password
        self.mailbox = mailbox
        self.batch_limit = batch_limit
        self.use_ssl = use_ssl
        self.primary_mailbox = (primary_mailbox or "").lower() or None

    # ------------------------------------------------------------------
    def fetch_emails(self, since_date: Optional[datetime] = None) -> List[Dict[str, Any]]:
        """Fetch emails via IMAP and return canonical records."""
        logger.info(
            "Connecting to IMAP server %s:%s mailbox=%s email=%s",
            self.host,
            self.port,
            self.mailbox,
            self.email_address,
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

            status, _ = conn.login(self.email_address, self.password)
            if status != "OK":
                logger.warning(
                    "IMAP login failed for %s: status=%s", self.email_address, status
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
                "Retrieved %d emails from IMAP server %s for %s",
                len(results),
                self.host,
                self.email_address,
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

    # ------------------------------------------------------------------
    def fetch_smart_batch(
        self,
        email_manager,
        since_date: Optional[datetime] = None,
        start_offset: int = 0
    ) -> tuple[List[Dict[str, Any]], bool]:
        """Fetch emails with intelligent duplicate detection and replacement.
        
        Implements the exact workflow specified:
        1. Fetch batch of emails equal to server's batch limit
        2. Check header_hash against database for each email
        3. Remove duplicates, process new emails
        4. Fetch additional emails equal to duplicates skipped to maintain batch size
        5. Repeat until batch size reached or end of mailbox
        
        Returns a tuple of (emails, has_more) where:
        - emails: List of unique email records not already in database
        - has_more: Boolean indicating if more emails are available
        
        Parameters
        ----------
        email_manager:
            EmailManager instance for database operations
        since_date:
            Optional datetime to filter emails from
        start_offset:
            Number of emails to skip from the beginning
        """
        logger.info(
            "Fetching smart batch from IMAP server %s:%s mailbox=%s email=%s offset=%d",
            self.host,
            self.port,
            self.mailbox,
            self.email_address,
            start_offset,
        )
        
        conn = (
            imaplib.IMAP4_SSL(self.host, self.port)
            if self.use_ssl
            else imaplib.IMAP4(self.host, self.port)
        )
        
        try:
            # Setup connection (same as fetch_emails)
            if not self.use_ssl:
                try:
                    status, _ = conn.starttls(ssl_context=ssl.create_default_context())
                    if status != "OK":
                        raise imaplib.IMAP4.error("STARTTLS failed")
                except Exception as exc:
                    raise RuntimeError(
                        "IMAP server requires a secure connection; STARTTLS negotiation failed"
                    ) from exc

            status, _ = conn.login(self.email_address, self.password)
            if status != "OK":
                logger.warning(
                    "IMAP login failed for %s: status=%s", self.email_address, status
                )
                return [], False

            status, _ = conn.select(self.mailbox)
            if status != "OK":
                logger.warning(
                    "IMAP select failed for mailbox %s: status=%s", self.mailbox, status
                )
                return [], False

            # Search for emails
            criteria: List[str] = []
            if since_date is not None:
                criteria.append(f'SINCE "{since_date.strftime("%d-%b-%Y")}"')
            search_query = "ALL" if not criteria else " ".join(criteria)
            status, messages = conn.search(None, search_query)
            if status != "OK":
                logger.warning("IMAP search failed: status=%s", status)
                return [], False
                
            email_ids = messages[0].split()
            total_emails = len(email_ids)
            target_batch_size = self.batch_limit or 50
            
            logger.info(
                "Found %d total emails, target batch size: %d, start offset: %d",
                total_emails,
                target_batch_size,
                start_offset
            )
            
            # Check if we've reached the end
            if start_offset >= total_emails:
                logger.info("Reached end of mailbox at offset %d", start_offset)
                return [], False
            
            unique_emails: List[Dict[str, Any]] = []
            current_offset = start_offset
            
            # Keep fetching until we have enough unique emails or reach end of mailbox
            while len(unique_emails) < target_batch_size and current_offset < total_emails:
                # Calculate how many more emails we need
                emails_needed = target_batch_size - len(unique_emails)
                
                # Fetch next batch of email IDs
                end_offset = min(current_offset + emails_needed, total_emails)
                batch_ids = email_ids[current_offset:end_offset]
                
                logger.debug(
                    "Fetching emails %d to %d (%d emails) to fill remaining %d slots",
                    current_offset,
                    end_offset - 1,
                    len(batch_ids),
                    emails_needed
                )
                
                # Process this batch
                batch_unique_count = 0
                batch_duplicate_count = 0
                
                for eid in batch_ids:
                    status, msg_data = conn.fetch(eid, "(RFC822)")
                    if status != "OK" or not msg_data or not msg_data[0]:
                        continue
                        
                    try:
                        msg_data_content = msg_data[0][1]
                        if isinstance(msg_data_content, bytes):
                            msg = message_from_bytes(msg_data_content)
                            rec = self._parse_email(msg)
                            rec["server_type"] = "imap"
                            
                            # Generate header hash for duplicate detection
                            header_hash = self._generate_header_hash(rec)
                            rec["header_hash"] = header_hash
                            
                            # Check if email already exists in database
                            if not self._email_exists_in_database(email_manager, header_hash):
                                unique_emails.append(rec)
                                batch_unique_count += 1
                                logger.debug("Added unique email %s", rec.get("message_id", "unknown"))
                            else:
                                batch_duplicate_count += 1
                                logger.debug("Skipped duplicate email %s", rec.get("message_id", "unknown"))
                        else:
                            logger.warning("Invalid message data type for email %s", eid)
                            
                    except Exception as exc:
                        decoded_eid = eid.decode() if isinstance(eid, bytes) else str(eid)
                        logger.warning("Failed to parse message %s: %s", decoded_eid, exc)
                
                # Move offset forward by the number of emails we processed
                current_offset = end_offset
                
                logger.debug(
                    "Batch complete: %d unique, %d duplicates, total unique so far: %d",
                    batch_unique_count,
                    batch_duplicate_count,
                    len(unique_emails)
                )
                
                # If we found no unique emails in this batch and we're not at the end,
                # we might be in a section with all duplicates - continue to next batch
                if batch_unique_count == 0 and current_offset < total_emails:
                    logger.debug("No unique emails in batch, continuing to next batch")
                    continue
                    
                # If we got unique emails but still need more, continue
                if len(unique_emails) < target_batch_size and current_offset < total_emails:
                    logger.debug("Need %d more unique emails, continuing", target_batch_size - len(unique_emails))
                    continue
                    
                # We either have enough emails or reached the end
                break
            
            # Determine if there are more emails available
            has_more = current_offset < total_emails
            
            logger.info(
                "Smart batch complete: %d unique emails collected, processed up to offset %d/%d, has_more: %s",
                len(unique_emails),
                current_offset,
                total_emails,
                has_more
            )
            
            return unique_emails, has_more
            
        finally:
            try:
                conn.logout()
            except Exception as exc:
                logger.warning("Error during IMAP logout: %s", exc)

    # ------------------------------------------------------------------
    def _email_exists_in_database(self, email_manager, header_hash: str) -> bool:
        """Check if email with given header_hash already exists in database.
        
        Parameters
        ----------
        email_manager:
            EmailManager instance for database operations
        header_hash:
            SHA256 hash of email headers for duplicate detection
            
        Returns
        -------
        bool:
            True if email exists in database, False otherwise
        """
        try:
            existing_email = email_manager.get_email_by_header_hash(header_hash)
            return existing_email is not None
        except Exception as exc:
            logger.warning("Error checking email existence for hash %s: %s", header_hash[:8], exc)
            return False

    # ------------------------------------------------------------------
    def _generate_header_hash(self, record: Dict[str, Any]) -> str:
        """Generate consistent SHA256 hash from email headers for duplicate detection.
        
        Parameters
        ----------
        record:
            Email record dictionary
            
        Returns
        -------
        str:
            SHA256 hash of normalized email headers
        """
        # Import compute_header_hash from email_manager to maintain consistency
        from .email_manager import compute_header_hash
        return compute_header_hash(record)


class ExchangeConnector(EmailConnector):
    """Retrieve emails from an Exchange server using EWS."""

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
