import threading
import logging
from typing import Any, Dict, List, Optional

from .connector import IMAPConnector
from .account_manager import EmailAccountManager

logger = logging.getLogger(__name__)


class EmailOrchestrator:
    """Periodically sync emails from configured IMAP accounts."""

    def __init__(self, config, account_manager: Optional[EmailAccountManager] = None):
        self.config = config
        self.account_manager = account_manager
        self.accounts: List[Dict[str, Any]] = []
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self.refresh_accounts()

    def start(self) -> None:
        """Start the background email sync loop if enabled."""
        if not self.config.EMAIL_ENABLED:
            logger.info("Email ingestion disabled; orchestrator not started")
            return
        if self._thread and self._thread.is_alive():
            return
        # Ensure we have the latest account list before starting
        self.refresh_accounts()
        self._thread = threading.Thread(target=self._run_loop, name="email-sync", daemon=True)
        self._thread.start()
        logger.info("Email orchestrator started")

    def refresh_accounts(self) -> None:
        """Reload the email account configurations."""
        if not self.account_manager:
            return
        try:
            self.accounts = self.account_manager.list_accounts(include_password=True)
        except Exception as exc:  # pragma: no cover - defensive
            logger.error(f"Failed to load email accounts: {exc}")
            self.accounts = []

    def _run_loop(self) -> None:
        """Background loop that fetches emails for each configured account."""
        interval = max(1, int(self.config.EMAIL_SYNC_INTERVAL_SECONDS))
        while not self._stop_event.is_set():
            # Refresh accounts each cycle to pick up changes
            self.refresh_accounts()
            for account in self.accounts:
                if account.get("server_type", "").lower() != "imap":
                    continue
                batch = account.get("batch_limit")
                batch_limit = None if batch in (None, "all") else int(batch)
                connector = IMAPConnector(
                    host=account["server"],
                    port=int(account["port"]),
                    username=account["username"],
                    password=account["password"],
                    mailbox=account.get("mailbox") or "INBOX",
                    batch_limit=batch_limit,
                    use_ssl=bool(account.get("use_ssl", True)),
                )
                name = account.get("account_name") or account.get("username")
                try:
                    records = connector.fetch_emails()
                    logger.info(
                        "Fetched %d emails for account %s", len(records), name
                    )
                except Exception as exc:  # pragma: no cover - defensive
                    logger.error(f"Email sync error for {name}: {exc}")
            self._stop_event.wait(interval)

    def stop(self) -> None:
        """Stop the background sync loop."""
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join()
