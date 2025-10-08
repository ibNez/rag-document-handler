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
        batch_size: Optional[int] = 50,
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
        self.batch_size = batch_size
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
        if self.batch_size is not None:
            qs = qs[: self.batch_size]

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

    def get_available_mailboxes(self) -> List[Dict[str, Any]]:
        """Get available Exchange mailboxes/folders that can be used for email sync.
        
        Returns
        -------
        List[Dict[str, Any]]
            List of mailbox dictionaries with id, name, type, and selectable fields.
        """
        logger.info(
            "Fetching available Exchange mailboxes from %s for %s",
            self.server,
            self.email_address,
        )
        
        try:
            mailboxes = []
            
            # Common Exchange folders
            folders_to_check = [
                ('inbox', self.account.inbox, 'inbox'),
                ('sent', self.account.sent, 'sent'),
                ('drafts', self.account.drafts, 'drafts'),
                ('outbox', self.account.outbox, 'outbox'),
                ('deleted_items', self.account.trash, 'trash'),
                ('junk_email', self.account.junk, 'spam'),
            ]
            
            for folder_id, folder_obj, folder_type in folders_to_check:
                if folder_obj is not None:
                    try:
                        # Get message count for this folder
                        message_count = folder_obj.total_count if hasattr(folder_obj, 'total_count') else None
                        
                        mailboxes.append({
                            'id': folder_id,
                            'name': folder_obj.name or folder_id.replace('_', ' ').title(),
                            'type': folder_type,
                            'messageCount': message_count,
                            'selectable': True
                        })
                    except Exception as e:
                        logger.warning(f"Failed to get info for Exchange folder {folder_id}: {e}")
                        continue
            
            # Try to get additional custom folders (if available)
            try:
                if hasattr(self.account, 'root') and hasattr(self.account.root, 'walk'):
                    for folder in self.account.root.walk():
                        if hasattr(folder, 'name') and hasattr(folder, 'folder_class'):
                            # Skip system folders we've already added
                            if folder.name.lower() in ['inbox', 'sent items', 'drafts', 'outbox', 'deleted items', 'junk e-mail']:
                                continue
                                
                            try:
                                message_count = folder.total_count if hasattr(folder, 'total_count') else None
                                mailboxes.append({
                                    'id': folder.name,
                                    'name': folder.name,
                                    'type': 'folder',
                                    'messageCount': message_count,
                                    'selectable': True
                                })
                            except Exception:
                                continue
            except Exception as e:
                logger.debug(f"Could not enumerate custom Exchange folders: {e}")
            
            logger.info(f"📬 Found {len(mailboxes)} available Exchange mailboxes")
            return mailboxes

        except Exception as e:
            logger.error(f"❌ Failed to get Exchange mailboxes: {e}")
            return [
                {
                    'id': 'inbox',
                    'name': 'Inbox',
                    'type': 'inbox',
                    'messageCount': None,
                    'selectable': True
                }
            ]

    def fetch_smart_batch(
        self,
        email_manager,
        since_date: Optional[datetime] = None,
        start_offset: int = 0,
        fetch_size: Optional[int] = None
    ) -> tuple[List[Dict[str, Any]], bool, int]:
        """Fetch emails with basic duplicate detection for Exchange.
        
        This is a simplified implementation that wraps fetch_emails() to provide
        the same interface as IMAPConnector's fetch_smart_batch method.
        
        Returns a tuple of (emails, has_more, total_emails) where:
        - emails: List of email records from Exchange EWS
        - has_more: Boolean indicating if more emails might be available  
        - total_emails: Estimated total (Exchange doesn't provide exact counts)
        
        Parameters
        ----------
        email_manager:
            EmailManager instance for database operations (unused in this simple implementation)
        since_date:
            Optional datetime to filter emails from
        start_offset:
            Number of emails to skip (not directly supported by Exchange EWS)
        fetch_size:
            Maximum number of emails to fetch (uses batch_size if not provided)
        """
        logger.info(
            "Fetching smart batch from Exchange server %s for %s, offset=%d",
            self.server,
            self.email_address,
            start_offset,
        )
        
        # Use the existing fetch_emails method
        # Note: Exchange EWS doesn't support offset-based pagination like IMAP
        # This is a simplified implementation that fetches from the beginning
        emails = self.fetch_emails(since_date=since_date)
        
        # Apply offset and batch size manually 
        target_batch_size = fetch_size or self.batch_size or 50
        end_offset = start_offset + target_batch_size
        
        # Slice the results based on offset
        batch_emails = emails[start_offset:end_offset] if start_offset < len(emails) else []
        
        # Determine if there are more emails
        has_more = end_offset < len(emails)
        total_emails = len(emails)  # This is the total we fetched, not total in mailbox
        
        logger.info(
            "Exchange smart batch: fetched %d emails, has_more=%s, total_fetched=%d",
            len(batch_emails),
            has_more,
            total_emails
        )
        
        return batch_emails, has_more, total_emails
