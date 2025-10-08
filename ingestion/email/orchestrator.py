"""
This module defines the EmailOrchestrator class, which is responsible for managing the periodic
synchronization of emails from configured accounts. It interacts with PostgreSQL for account
management and uses connectors for email retrieval.

Classes:
    EmailOrchestrator: Handles email synchronization tasks, including re            update_status("syncing", 40, f"Fetching emails from {server_type.upper()}...")
            
            # For Gmail, use a custom processing approach to handle windowed syncunt
    configurations and determining accounts due for synchronization.
"""

import logging
from datetime import UTC, datetime, timedelta
from typing import Any, Dict, List, Optional, Callable

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials

from .connectors.imap_connector import IMAPConnector
from .connectors.gmail_connector import GmailConnector
from .connectors.exchange_connector import ExchangeConnector
from .account_manager import EmailAccountManager
from .processor import EmailProcessor
from ingestion.utils.crypto import encrypt, decrypt

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
        batch = account.get("batch_size")
        batch_size = None if batch in (None, "all") else int(batch)
        
        def update_status(status: str, progress: int, message: str):
            """Helper to update status if callback provided."""
            if status_callback:
                status_callback(status, progress, message)
        
        # Get sync position for account (Gmail accounts always use 0)
        current_offset = account.get("last_synced_offset", 0) if server_type != "gmail" else 0
        logger.info(f"🚀 EMAIL_SYNC_START: Account={name}, server_type={server_type}, batch_size={batch_size}, offset={current_offset}, account_id={acct_id}")

        update_status("connecting", 10, f"Connecting to {server_type.upper()} server...")

        if server_type == "imap":
            connector = IMAPConnector(
                host=account["server"],
                port=int(account["port"]),
                email_address=account["email_address"],
                password=account["password"],
                mailbox=account.get("mailbox") or "INBOX",
                batch_size=batch_size,
                use_ssl=bool(account.get("use_ssl", True)),
            )
            logger.info(f"🔗 IMAP_CONNECTOR: Created IMAP connector for {name}")
        elif server_type == "gmail":
            update_status("connecting", 15, "Authenticating with Gmail...")
            logger.info(f"🔗 GMAIL_AUTH_START: Starting Gmail authentication for {name}")
            
            # Check for OAuth tokens in database (new format)
            oauth_access_token = account.get("oauth_access_token")
            oauth_refresh_token = account.get("oauth_refresh_token")
            oauth_expires_at = account.get("oauth_expires_at") or account.get("oauth_token_expires_at")
            
            logger.info(f"🔑 GMAIL_TOKENS: access_token={bool(oauth_access_token)}, refresh_token={bool(oauth_refresh_token)}, expires_at={oauth_expires_at}")
            
            if oauth_access_token and oauth_refresh_token:
                # Create credentials from database OAuth tokens
                try:
                    from ingestion.utils.crypto import decrypt
                    
                    # Decrypt the tokens
                    access_token = decrypt(oauth_access_token)
                    refresh_token = decrypt(oauth_refresh_token)
                    
                    # Parse expiry date
                    token_expiry = None
                    if oauth_expires_at:
                        try:
                            if isinstance(oauth_expires_at, str):
                                token_expiry = datetime.fromisoformat(oauth_expires_at.replace('Z', '+00:00'))
                            else:
                                token_expiry = oauth_expires_at
                        except Exception as e:
                            logger.warning(f"Failed to parse OAuth expiry date: {e}")
                    
                    # Create credentials object
                    creds = Credentials(
                        token=access_token,
                        refresh_token=refresh_token,
                        token_uri="https://oauth2.googleapis.com/token",
                        client_id=self.config.GMAIL_CLIENT_ID,
                        client_secret=self.config.GMAIL_CLIENT_SECRET,
                        expiry=token_expiry
                    )
                    
                    # Refresh token if expired
                    if creds.expired and creds.refresh_token:
                        try:
                            update_status("connecting", 18, "Refreshing Gmail credentials...")
                            creds.refresh(Request())
                            # Update the account with new tokens
                            if self.account_manager and acct_id:
                                try:
                                    updates = {
                                        'oauth_access_token': encrypt(creds.token),
                                        'oauth_token_expires_at': creds.expiry.isoformat() if creds.expiry else None
                                    }
                                    self.account_manager.update_account(str(acct_id), updates)
                                    logger.info(f"Updated refreshed OAuth tokens for account {name}")
                                except Exception as update_exc:
                                    logger.warning(f"Failed to update refreshed tokens: {update_exc}")
                        except Exception as exc:
                            logger.warning(f"Failed to refresh Gmail token for {name}: {exc}")
                            update_status("error", 100, f"Failed to refresh Gmail token: {exc}")
                            return
                            
                except Exception as exc:
                    logger.error(f"Failed to create Gmail credentials from database tokens for {name}: {exc}")
                    logger.error(f"Exception type: {type(exc)}")
                    logger.error(f"Exception args: {exc.args}")
                    import traceback
                    logger.error(f"Full traceback: {traceback.format_exc()}")
                    update_status("error", 100, f"Failed to load Gmail credentials: {exc}")
                    return
                    
            else:
                # No OAuth tokens configured - this is an error
                logger.error(f"No OAuth tokens configured for Gmail account {name}")
                logger.error(f"Gmail accounts require oauth_access_token and oauth_refresh_token to be configured")
                update_status("error", 100, "No Gmail OAuth tokens configured")
                raise RuntimeError(f"Gmail account {name} missing required OAuth tokens")
                    
            # Create Gmail connector
            connector = GmailConnector(account)
        elif server_type == "exchange":
            update_status("connecting", 15, "Connecting to Exchange server...")
            connector = ExchangeConnector(
                server=account["server"],
                email_address=account["email_address"],
                password=account["password"],
                batch_size=batch_size,
            )
        else:
            logger.warning("Unknown server type '%s' for %s", server_type, name)
            update_status("error", 100, f"Unknown server type: {server_type}")
            return

        update_status("connecting", 20, f"Connected to {server_type.upper()} server")
        logger.info("Syncing account %s using %s", name, server_type)

        # Set sync position based on protocol type
        current_offset = account.get("last_synced_offset", 0) if server_type != "gmail" else 0
        if server_type == "gmail":
            logger.info("Gmail account %s uses history-based sync (no offset)", name)
        else:
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
            logger.info("Using smart batch processing for account %s with batch_size=%s", name, batch_size)
            
            update_status("syncing", 40, f"Fetching emails from {server_type.upper()}...")
            
            # For Gmail, use a custom processing approach to handle windowed sync
            if server_type == "gmail":
                stats = self._process_gmail_smart_batch(
                    connector=connector,
                    processor=processor,
                    account_id=str(acct_id) if acct_id else None,
                    batch_size=batch_size
                )
            else:
                # Use standard processing for IMAP/Exchange with offset support
                stats = processor.process_smart_batch(
                    connector=connector,
                    since_date=None,  # Process all emails
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
            
            # Update offset for non-Gmail accounts only
            if server_type != "gmail":
                update_status("finalizing", 85, "Updating account sync status...")
                final_offset = stats.get('final_offset', current_offset)
                if acct_id and self.account_manager and hasattr(self.account_manager, "update_account"):
                    try:
                        self.account_manager.update_account(
                            acct_id,
                            {"last_synced_offset": final_offset}
                        )
                        logger.info("📍 OFFSET_UPDATE: Updated account %s offset to %d", name, final_offset)
                    except Exception as exc:
                        logger.error("Failed to update offset for account %s: %s", name, exc)
            else:
                logger.info("📍 GMAIL_NO_OFFSET: Gmail account %s uses history-based sync - historyId updates handled separately", name)
                    
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

    def _process_gmail_smart_batch(
        self,
        connector,
        processor,
        account_id: Optional[str] = None,
        batch_size: Optional[int] = None
    ) -> Dict[str, int]:
        """
        Incremental Gmail sync using a durable checkpoint and message-id boundary.
        """
        logger.info("📧 GMAIL_INCREMENTAL_START: Starting Gmail incremental sync")

        stats = {
            'total_emails_processed': 0,
            'total_batches': 0,
            'errors': 0,
            'final_offset': 0
        }

        try:
            if not account_id:
                logger.error("❌ GMAIL_NO_ACCOUNT_ID: Cannot perform Gmail sync without account_id")
                return stats | {'errors': 1}

            # Access EmailDataManager for checkpoint and existence checks
            email_data_manager = getattr(processor, 'email_data_manager', None) or \
                                 getattr(self.account_manager, 'email_data_manager', None)
            if not email_data_manager:
                logger.error("❌ GMAIL_NO_EMAIL_DATA_MANAGER: Missing EmailDataManager for sync")
                return stats | {'errors': 1}

            # Get checkpoint (newest processed date) to bound queries
            sync_window = email_data_manager.get_account_sync_window(account_id)
            since_dt = sync_window.get('newest_processed_date') if sync_window else None
            logger.info(f"🧭 GMAIL_CHECKPOINT: newest_processed_date={since_dt}")

            # List message IDs newest-first optionally filtered by since date
            msg_ids = connector.list_message_ids(since_dt)
            if not msg_ids:
                logger.info("📭 GMAIL_NO_MESSAGES: No Gmail messages found for sync")
                return stats

            total = len(msg_ids)
            logger.info(f"📬 GMAIL_IDS: {total} message IDs to evaluate")

            max_seen_date = since_dt
            processed = 0
            batch_counter = 0

            for idx, msg_id in enumerate(msg_ids, start=1):
                # Fetch and parse first to get message date for proper boundary check
                email_record = connector.get_email_by_id(msg_id)
                if not email_record:
                    continue

                try:
                    # Boundary: if checkpoint set and this message is older/equal AND already exists, stop
                    if since_dt:
                        msg_dt = email_record.get('date_utc')
                        if msg_dt and msg_dt <= since_dt and email_data_manager.email_exists_by_message_id(msg_id):
                            logger.info(f"⛔️ GMAIL_BOUNDARY_REACHED at #{idx}: encountered already-processed message at or before checkpoint")
                            break

                    processor.process(email_record, account_id=account_id)
                    processed += 1
                    # Track newest date
                    msg_dt = email_record.get('date_utc')
                    if msg_dt and (max_seen_date is None or msg_dt > max_seen_date):
                        max_seen_date = msg_dt
                except Exception as email_exc:
                    stats['errors'] += 1
                    logger.warning(f"⚠️ GMAIL_EMAIL_ERROR: Failed to process message {msg_id}: {email_exc}")

                # Light progress logs
                if idx % 50 == 0:
                    logger.info(f"📈 GMAIL_PROGRESS: {idx}/{total} checked, {processed} processed")

            # Batches are conceptual pages; increment when we cross 500 ids boundary
            batch_counter = (total // 500) + (1 if total % 500 else 0)
            stats['total_batches'] = batch_counter
            stats['total_emails_processed'] = processed

            # Update checkpoint
            try:
                if max_seen_date:
                    email_data_manager.update_account_sync_window(account_id, newest_processed_date=max_seen_date)
                    logger.info(f"✅ GMAIL_CHECKPOINT_UPDATED: newest_processed_date={max_seen_date}")
            except Exception as ex:
                logger.warning(f"Failed to update Gmail checkpoint: {ex}")

            logger.info(
                f"✅ GMAIL_INCREMENTAL_DONE: processed={processed}, checked={total}, errors={stats['errors']}"
            )
            return stats

        except Exception as exc:
            logger.error(f"❌ GMAIL_INCREMENTAL_FAILED: {exc}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return stats | {'errors': stats['errors'] + 1}
