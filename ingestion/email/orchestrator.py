import threading
import time
import logging
from typing import Optional

from .connector import IMAPConnector

logger = logging.getLogger(__name__)


class EmailOrchestrator:
    """Periodically sync emails from an IMAP source using configuration."""

    def __init__(self, config):
        self.config = config
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

    def start(self) -> None:
        """Start the background email sync loop if enabled."""
        if not self.config.EMAIL_ENABLED:
            logger.info("Email ingestion disabled; orchestrator not started")
            return
        if self._thread and self._thread.is_alive():
            return
        self._thread = threading.Thread(target=self._run_loop, name="email-sync", daemon=True)
        self._thread.start()
        logger.info("Email orchestrator started")

    def _run_loop(self) -> None:
        """Background loop that fetches emails at the configured interval."""
        connector = IMAPConnector(
            host=self.config.IMAP_HOST,
            username=self.config.IMAP_USERNAME,
            password=self.config.IMAP_PASSWORD,
            mailbox=self.config.IMAP_MAILBOX,
            batch_limit=self.config.IMAP_BATCH_LIMIT,
            use_ssl=self.config.IMAP_USE_SSL,
        )
        interval = max(1, int(self.config.EMAIL_SYNC_INTERVAL_SECONDS))
        while not self._stop_event.is_set():
            try:
                connector.fetch_emails()
            except Exception as exc:  # pragma: no cover - defensive
                logger.error(f"Email sync error: {exc}")
            self._stop_event.wait(interval)

    def stop(self) -> None:
        """Stop the background sync loop."""
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join()
