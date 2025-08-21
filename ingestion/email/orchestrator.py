"""
This module defines the EmailOrchestrator class, which is responsible for managing the periodic
synchronization of emails from configured accounts. It interacts with PostgreSQL for account
management and uses connectors for email retrieval.

Classes:
    EmailOrchestrator: Handles email synchronization tasks, including refreshing account
    configurations and determining accounts due for synchronization.
"""

import logging
from datetime import UTC, datetime, timedelta
from typing import Any, Dict, List, Optional

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials

from .connectors import IMAPConnector, GmailConnector, ExchangeConnector
from .manager import PostgreSQLEmailManager
from .processor import EmailProcessor
from .ingest import _normalize

logger = logging.getLogger(__name__)


class EmailOrchestrator:
    """Periodically sync emails from configured accounts."""

    def __init__(
        self,
        config,
        account_manager: Optional[PostgreSQLEmailManager] = None,
        processor: Optional[EmailProcessor] = None,
    ) -> None:
        self.config = config
        self.account_manager = account_manager
        self.processor = processor
        self.accounts: List[Dict[str, Any]] = []
        self.refresh_accounts()

    def refresh_accounts(self) -> None:
        """Reload the email account configurations."""
        if not self.account_manager:
            return
        try:
            logger.debug("Refreshing email accounts.")
            self.accounts = self.account_manager.list_accounts(include_password=True)
            logger.debug("Accounts retrieved: %s", self.accounts)
        except Exception as exc:  # pragma: no cover - defensive
            logger.error(f"Failed to load email accounts: {exc}")
            self.accounts = []

    def get_due_accounts(self) -> List[Dict[str, Any]]:
        """Return accounts that need a refresh based on their schedule."""
        if not self.account_manager:
            return []
        # Refresh accounts to ensure we operate on latest configuration
        self.refresh_accounts()
        now = datetime.now(UTC)
        due: List[Dict[str, Any]] = []
        for account in self.accounts:
            interval = account.get("refresh_interval_minutes")
            if interval is None:
                logger.warning(f"Email account {account.get('account_name')} missing refresh_interval_minutes, using 60 minutes default")
                interval = 60  # Default to 1 hour if missing
            try:
                last = account.get("last_synced")
                if last:
                    last_dt = datetime.fromisoformat(str(last))
                    # Ensure timezone-aware datetime for comparison
                    if last_dt.tzinfo is None:
                        # Assume UTC if no timezone info
                        last_dt = last_dt.replace(tzinfo=UTC)
                else:
                    last_dt = None
            except Exception:  # pragma: no cover - defensive
                last_dt = None

            # Always sync accounts that have never been synced before
            if not last_dt:
                due.append(account)
                continue

            # Skip disabled accounts after initial sync
            if interval <= 0:
                continue

            if last_dt + timedelta(minutes=int(interval)) <= now:
                due.append(account)
        return due

    def sync_account(
        self, account: Dict[str, Any], processor: Optional[EmailProcessor] = None
    ) -> None:
        """Fetch and process emails for a single account and update its last_synced timestamp."""
        name = account.get("account_name") or account.get("email_address")
        server_type = (account.get("server_type") or "").lower()
        batch = account.get("batch_limit")
        batch_limit = None if batch in (None, "all") else int(batch)

        if server_type == "imap":
            connector = IMAPConnector(
                host=account["server"],
                port=int(account["port"]),
                email_address=account["email_address"],
                password=account["password"],
                mailbox=account.get("mailbox") or "INBOX",
                batch_limit=batch_limit,
                use_ssl=bool(account.get("use_ssl", True)),
            )
        elif server_type == "gmail":
            token_path = account.get("token_file") or account.get("token_path")
            if not token_path:
                logger.error(f"No token file configured for Gmail account {name}")
                return
            try:
                creds = Credentials.from_authorized_user_file(token_path)
                if creds.expired and creds.refresh_token:
                    try:
                        creds.refresh(Request())
                        with open(token_path, "w") as fh:
                            fh.write(creds.to_json())
                    except Exception as exc:  # pragma: no cover - defensive
                        logger.warning(
                            "Failed to refresh Gmail token for %s: %s", name, exc
                        )
            except Exception as exc:
                logger.error(
                    f"Failed to load Gmail credentials for {name}: {exc}"
                )
                return
            connector = GmailConnector(
                credentials=creds,
                user_id=account.get("email_address") or "me",
                batch_limit=batch_limit,
            )
        elif server_type == "exchange":
            connector = ExchangeConnector(
                server=account["server"],
                email_address=account["email_address"],
                password=account["password"],
                batch_limit=batch_limit,
            )
        else:
            logger.warning("Unknown server type '%s' for %s", server_type, name)
            return

        logger.info("Syncing account %s using %s", name, server_type)

        processor = processor or self.processor
        if processor is None:
            class _NoopMilvus:
                def add_embeddings(self, embeddings, ids, metadatas):
                    pass

            try:
                # Use the existing PostgreSQL connection from the account manager
                conn = getattr(self.account_manager, 'conn', None) if hasattr(self, 'account_manager') else None
                if conn is not None:
                    processor = EmailProcessor(_NoopMilvus(), conn)
                else:
                    # Fallback - this should not happen in normal operation
                    logger.warning("No PostgreSQL connection available, EmailProcessor cannot be initialized")
                    processor = None
            except Exception as exc:  # pragma: no cover - defensive
                logger.error("Failed to initialize EmailProcessor: %s", exc)
                processor = None

        try:
            # Use smart batch processing for complete mailbox coverage
            if (processor and hasattr(processor, 'process_smart_batch') and 
                hasattr(connector, 'fetch_smart_batch')):
                logger.info("Using smart batch processing for account %s", name)
                stats = processor.process_smart_batch(
                    connector=connector,
                    since_date=None,  # Process all emails
                    max_batches=None  # No limit on batches
                )
                logger.info(
                    "Smart batch processing complete for %s: %d emails processed, %d batches, %d errors",
                    name, stats.get('total_emails_processed', 0), 
                    stats.get('total_batches', 0), stats.get('errors', 0)
                )
            else:
                # Fallback to traditional processing (legacy mode)
                logger.warning("Smart batch processing not available, using legacy mode for %s", name)
                records = connector.fetch_emails()
                logger.info("Fetched %d emails for account %s", len(records), name)
                if processor:
                    for rec in records:
                        try:
                            norm = _normalize(rec)
                            hh = norm.get("header_hash")
                            if hh and processor.manager.get_email_by_header_hash(hh):
                                continue
                            ch = norm.get("content_hash")
                            if ch and processor.manager.get_email_by_hash(ch):
                                continue
                            processor.process(norm)
                        except Exception as exc:  # pragma: no cover - defensive
                            mid = rec.get("message_id")
                            logger.error(
                                "Failed to process email %s for %s: %s", mid, name, exc
                            )
        except Exception as exc:  # pragma: no cover - defensive
            logger.error("Email sync error for %s: %s", name, exc)

        acct_id = account.get("id")
        if acct_id and self.account_manager and hasattr(
            self.account_manager, "update_account"
        ):
            try:
                self.account_manager.update_account(
                    acct_id,
                    {
                        "last_synced": datetime.now(UTC).isoformat(
                            sep=" ", timespec="seconds"
                        )
                    },
                )
            except Exception as exc:  # pragma: no cover - defensive
                logger.error(
                    "Failed to update account %s after sync: %s", name, exc
                )

    def run_cycle(self) -> bool:
        """Refresh accounts and sync any that are due.

        Returns True if at least one account was processed.
        """
        due_accounts = self.get_due_accounts()
        if not due_accounts:
            logger.debug("No email accounts due for sync")
            return False
        logger.info(
            "Email sync cycle start: accounts=%d", len(due_accounts)
        )
        for account in due_accounts:
            self.sync_account(account, processor=self.processor)
        logger.info("Email sync cycle complete")
        return True

    def run(self, account_id: Optional[int] = None) -> bool:
        """Public entry point for email synchronization.

        If ``account_id`` is provided, only that account will be synced.
        Otherwise, a full cycle is executed for any accounts that are due.

        Returns ``True`` if at least one account was processed.
        """
        if account_id is not None:
            if not self.account_manager:
                return False
            try:
                account = next(
                    acct
                    for acct in self.account_manager.list_accounts(
                        include_password=True
                    )
                    if acct.get("id") == account_id
                )
            except StopIteration:
                logger.error("No email account found for id %s", account_id)
                return False
            self.sync_account(account, processor=self.processor)
            return True
        return self.run_cycle()
