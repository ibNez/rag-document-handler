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
from typing import Any, Dict, List, Optional, Callable

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials

from .connectors import IMAPConnector, GmailConnector, ExchangeConnector
from .account_manager import EmailAccountManager
from .processor import EmailProcessor
from .ingest import _normalize

logger = logging.getLogger(__name__)


class EmailOrchestrator:
    """Periodically sync emails from configured accounts."""

    def __init__(
        self,
        config,
        account_manager: Optional[EmailAccountManager] = None,
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
            logger.debug("No account manager available")
            return []
        # Refresh accounts to ensure we operate on latest configuration
        logger.debug("Refreshing accounts before checking due status")
        self.refresh_accounts()
        logger.debug(f"Found {len(self.accounts)} total accounts after refresh")
        
        now = datetime.now(UTC)
        due: List[Dict[str, Any]] = []
        for account in self.accounts:
            account_name = account.get("account_name", "Unknown")
            logger.debug(f"Checking account: {account_name}")
            
            interval = account.get("refresh_interval_minutes")
            if interval is None:
                logger.warning(f"Email account {account_name} missing refresh_interval_minutes, using 60 minutes default")
                interval = 60  # Default to 1 hour if missing
            
            logger.debug(f"  Refresh interval: {interval} minutes")
            
            try:
                last = account.get("last_synced")
                logger.debug(f"  Last synced raw value: {last}")
                if last:
                    last_dt = datetime.fromisoformat(str(last))
                    # Ensure timezone-aware datetime for comparison
                    if last_dt.tzinfo is None:
                        # Assume UTC if no timezone info
                        last_dt = last_dt.replace(tzinfo=UTC)
                else:
                    last_dt = None
                logger.debug(f"  Last synced datetime: {last_dt}")
            except Exception as e:  # pragma: no cover - defensive
                logger.debug(f"  Error parsing last_synced: {e}")
                last_dt = None

            # Always sync accounts that have never been synced before
            if not last_dt:
                logger.debug(f"  Account {account_name} has never been synced, marking as due")
                due.append(account)
                continue

            # Skip disabled accounts after initial sync
            if interval <= 0:
                logger.debug(f"  Account {account_name} has disabled refresh interval ({interval}), skipping")
                continue

            next_sync_time = last_dt + timedelta(minutes=int(interval))
            logger.debug(f"  Next sync time: {next_sync_time}, current time: {now}")
            if next_sync_time <= now:
                logger.debug(f"  Account {account_name} is due for sync")
                due.append(account)
            else:
                logger.debug(f"  Account {account_name} not due yet")
                
        logger.debug(f"Total due accounts: {len(due)}")
        return due

    def sync_account(
        self, account: Dict[str, Any], processor: Optional[EmailProcessor] = None,
        status_callback: Optional[Callable[[str, int, str], None]] = None
    ) -> None:
        """Fetch and process emails for a single account and update its last_synced timestamp.
        
        Args:
            account: Account configuration dictionary
            processor: Email processor instance
            status_callback: Optional callback function(status, progress, message) for status updates
        """
        name = account.get("account_name") or account.get("email_address")
        acct_id = account.get("email_account_id")
        server_type = (account.get("server_type") or "").lower()
        batch = account.get("batch_limit")
        batch_limit = None if batch in (None, "all") else int(batch)
        
        def update_status(status: str, progress: int, message: str):
            """Helper to update status if callback provided."""
            if status_callback:
                status_callback(status, progress, message)
        
        # Get current offset for this account (where to start processing)
        current_offset = account.get("last_synced_offset", 0)
        logger.info(f"Account {name}: batch={batch}, batch_limit={batch_limit}, starting_offset={current_offset}")

        update_status("connecting", 10, f"Connecting to {server_type.upper()} server...")

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
            update_status("connecting", 15, "Authenticating with Gmail...")
            token_path = account.get("token_file") or account.get("token_path")
            if not token_path:
                logger.error(f"No token file configured for Gmail account {name}")
                update_status("error", 100, "No token file configured for Gmail")
                return
            try:
                creds = Credentials.from_authorized_user_file(token_path)
                if creds.expired and creds.refresh_token:
                    try:
                        update_status("connecting", 18, "Refreshing Gmail credentials...")
                        creds.refresh(Request())
                        with open(token_path, "w") as fh:
                            fh.write(creds.to_json())
                    except Exception as exc:  # pragma: no cover - defensive
                        logger.warning(
                            "Failed to refresh Gmail token for %s: %s", name, exc
                        )
                        update_status("error", 100, f"Failed to refresh Gmail token: {exc}")
                        return
            except Exception as exc:
                logger.error(
                    f"Failed to load Gmail credentials for {name}: {exc}"
                )
                update_status("error", 100, f"Failed to load Gmail credentials: {exc}")
                return
            connector = GmailConnector(
                credentials=creds,
                user_id=account.get("email_address") or "me",
                batch_limit=batch_limit,
            )
        elif server_type == "exchange":
            update_status("connecting", 15, "Connecting to Exchange server...")
            connector = ExchangeConnector(
                server=account["server"],
                email_address=account["email_address"],
                password=account["password"],
                batch_limit=batch_limit,
            )
        else:
            logger.warning("Unknown server type '%s' for %s", server_type, name)
            update_status("error", 100, f"Unknown server type: {server_type}")
            return

        update_status("connecting", 20, f"Connected to {server_type.upper()} server")
        logger.info("Syncing account %s using %s", name, server_type)

        # Get the current offset for this account
        current_offset = account.get("last_synced_offset", 0)
        logger.info("Starting sync for %s at offset %d", name, current_offset)

        update_status("preparing", 25, "Preparing email processor...")

        processor = processor or self.processor
        if processor is None:
            # A processor is required for email sync - fail if we can't create one
            logger.error("No EmailProcessor available for account %s", name)
            if not self.account_manager:
                update_status("error", 100, "No account manager available")
                raise RuntimeError(f"Cannot sync account {name}: no account manager available")
            
            # Try to create processor using account manager's database pool
            if not hasattr(self.account_manager, 'pool'):
                update_status("error", 100, "Account manager has no database pool")
                raise RuntimeError(f"Cannot sync account {name}: account manager has no database pool")
                
            try:
                class _NoopMilvus:
                    def add_embeddings(self, embeddings, ids, metadatas):
                        pass
                
                processor = EmailProcessor(_NoopMilvus(), self.account_manager)
                logger.info("Created EmailProcessor for account %s", name)
            except Exception as exc:
                logger.error("Failed to create EmailProcessor for account %s: %s", name, exc)
                update_status("error", 100, f"Failed to create EmailProcessor: {exc}")
                raise RuntimeError(f"Cannot sync account {name}: failed to create EmailProcessor - {exc}")
            
        if processor is None:
            update_status("error", 100, "EmailProcessor is required but not available")
            raise RuntimeError(f"Cannot sync account {name}: EmailProcessor is required but not available")

        try:
            # Smart batch processing is the only supported method
            update_status("syncing", 30, "Starting email synchronization...")
            logger.info("Using smart batch processing for account %s", name)
            # Respect batch limit: if batch_limit is set, process only 1 batch per sync cycle
            max_batches = 1 if batch_limit else None
            logger.info("Processing %s batches for account %s (batch_limit=%s)", 
                       "1" if max_batches else "unlimited", name, batch_limit)
            
            update_status("syncing", 40, f"Fetching emails from {server_type.upper()}...")
            
            stats = processor.process_smart_batch(
                connector=connector,
                since_date=None,  # Process all emails
                max_batches=max_batches,  # Respect batch limit setting
                start_offset=current_offset  # Start from stored offset
            )
            
            processed_count = stats.get('total_emails_processed', 0)
            batches_count = stats.get('total_batches', 0)
            errors_count = stats.get('errors', 0)
            
            update_status("processing", 70, f"Processed {processed_count} emails in {batches_count} batches")
            
            logger.info(
                "Smart batch processing complete for %s: %d emails processed, %d batches, %d errors",
                name, processed_count, batches_count, errors_count
            )
            
            # Update the account's offset for next batch cycle
            update_status("finalizing", 85, "Updating account sync status...")
            final_offset = stats.get('final_offset', current_offset)
            if acct_id and self.account_manager and hasattr(self.account_manager, "update_account"):
                try:
                    self.account_manager.update_account(
                        acct_id,
                        {"last_synced_offset": final_offset}
                    )
                    logger.info("Updated account %s offset to %d", name, final_offset)
                except Exception as exc:
                    logger.error("Failed to update offset for account %s: %s", name, exc)
                    
        except Exception as exc:  # pragma: no cover - defensive
            logger.error("Email sync error for %s: %s", name, exc)
            update_status("error", 100, f"Sync failed: {str(exc)}")
            raise  # Re-raise the exception so it fails properly

        update_status("finalizing", 95, "Updating last sync timestamp...")
        
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
                # Get processed_count, defaulting to 0 if not available
                processed_count = locals().get('processed_count', 0)
                update_status("completed", 100, f"Sync completed - {processed_count} emails processed")
            except Exception as exc:  # pragma: no cover - defensive
                logger.error(
                    "Failed to update account %s after sync: %s", name, exc
                )
                # Get processed_count, defaulting to 0 if not available
                processed_count = locals().get('processed_count', 0)
                update_status("completed", 100, f"Sync completed with warning - {processed_count} emails processed")

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

    def run(self, account_id: Optional[str] = None, status_callback: Optional[Callable[[str, int, str], None]] = None) -> bool:
        """Public entry point for email synchronization.

        If ``account_id`` is provided, only that account will be synced.
        Otherwise, a full cycle is executed for any accounts that are due.

        Args:
            account_id: Optional account ID to sync specific account
            status_callback: Optional callback function(status, progress, message) for status updates

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
                    if acct.get("email_account_id") == account_id
                )
            except StopIteration:
                logger.error("No email account found for id %s", account_id)
                if status_callback:
                    status_callback("error", 100, f"Account {account_id} not found")
                return False
            self.sync_account(account, processor=self.processor, status_callback=status_callback)
            return True
        return self.run_cycle()
