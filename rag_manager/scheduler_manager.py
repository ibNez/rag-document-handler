import threading
import time
from typing import Any, Dict
import logging
from rag_manager.core.models import URLProcessingStatus, EmailProcessingStatus

logger = logging.getLogger(__name__)

class SchedulerManager:
    def __init__(self, url_manager, config, email_orchestrator=None):
        self.url_manager = url_manager
        self.config = config
        self.email_orchestrator = email_orchestrator
        self._scheduler_thread = None
        self._scheduler_last_cycle = None
        # These need to be passed from the main app
        self.url_processing_status = {}
        self.email_processing_status = {}
        self._process_url_background = None
        self._refresh_email_account_background = None
    
    def set_background_processors(self, url_processor, email_processor):
        """Set the background processing methods from the main app."""
        self._process_url_background = url_processor
        self._refresh_email_account_background = email_processor
    
    def set_processing_status(self, url_status, email_status):
        """Set the processing status dictionaries from the main app."""
        self.url_processing_status = url_status
        self.email_processing_status = email_status
    
    def set_email_orchestrator(self, email_orchestrator):
        """Set the email orchestrator after initialization."""
        self.email_orchestrator = email_orchestrator

    def start_scheduler(self):
        """Start the unified scheduler thread if not already running."""
        if self._scheduler_thread and self._scheduler_thread.is_alive():
            logger.debug("Scheduler thread already running")
            return
        th = threading.Thread(target=self._scheduler_loop, name="scheduler")
        th.daemon = True
        th.start()
        self._scheduler_thread = th
        logger.info(f"Scheduler thread started (ident={th.ident})")

    def scheduler_status(self) -> Dict[str, Any]:
        """Return current scheduler diagnostic info."""
        alive = bool(self._scheduler_thread and self._scheduler_thread.is_alive())
        try:
            due_preview = self.url_manager.get_due_urls()[:5]
        except Exception:
            due_preview = []
        last_cycle_age = None
        if self._scheduler_last_cycle:
            try:
                last_cycle_age = round(time.time() - self._scheduler_last_cycle, 2)
            except Exception:
                pass
        return {
            'running': alive,
            'thread_ident': getattr(self._scheduler_thread, 'ident', None),
            'due_count': len(due_preview),
            'due_sample': [d.get('url') for d in due_preview],
            'last_cycle_age_seconds': last_cycle_age,
            'poll_busy_seconds': self.config.SCHEDULER_POLL_SECONDS_BUSY,
            'poll_idle_seconds': self.config.SCHEDULER_POLL_SECONDS_IDLE
        }

    def _scheduler_loop(self):
        """Background loop that schedules URL and email refresh tasks."""
        logger.info("Scheduler started")
        cycle = 0
        while True:
            try:
                cycle += 1
                self._scheduler_last_cycle = time.time()
                due_urls = self.url_manager.get_due_urls()
                due_accounts = []
                active_emails_total = 0
                
                email_orchestrator = getattr(self, 'email_orchestrator', None)
                logger.debug(f"Email orchestrator found: {email_orchestrator is not None}")
                
                if email_orchestrator:
                    try:
                        logger.debug("Getting due email accounts...")
                        due_accounts = email_orchestrator.get_due_accounts()
                        logger.debug(f"Found {len(due_accounts)} due email accounts")
                        for account in due_accounts:
                            logger.debug(f"  Due account: {account.get('account_name')} (ID: {account.get('id')})")
                    except Exception as exc:
                        logger.error(f"Failed to get due email accounts: {exc}")
                        due_accounts = []
                    try:
                        logger.debug("Getting email account count...")
                        active_emails_total = email_orchestrator.account_manager.get_account_count()
                        logger.debug(f"Email account count: {active_emails_total}")
                    except Exception as exc:
                        logger.error(f"Failed to get email account count: {exc}")
                        active_emails_total = 0
                else:
                    logger.warning("Email orchestrator not available in scheduler")
                    due_accounts = []
                    
                logger.info(
                    "Scheduler cycle %s heartbeat: urls_due=%s emails_due=%s active_urls_total=%s active_emails_total=%s",
                    cycle,
                    len(due_urls),
                    len(due_accounts),
                    self.url_manager.get_url_count(),
                    active_emails_total,
                )
                started = 0
                # Process due URLs
                for rec in due_urls:
                    url_id = rec.get('id')
                    if url_id is None or url_id in self.url_processing_status:
                        continue
                    self.url_processing_status[url_id] = URLProcessingStatus(url=rec.get('url'))
                    if self._process_url_background:
                        t = threading.Thread(target=self._process_url_background, args=(url_id,))
                        t.daemon = True
                        t.start()
                        started += 1
                
                # Process due email accounts
                for account in due_accounts:
                    acct_id = account.get('id')
                    if acct_id is None or acct_id in self.email_processing_status:
                        continue
                    self.email_processing_status[acct_id] = EmailProcessingStatus(email_id=acct_id)
                    if self._refresh_email_account_background:
                        t = threading.Thread(target=self._refresh_email_account_background, args=(acct_id,))
                        t.daemon = True
                        t.start()
                        started += 1
                sleep_for = self.config.SCHEDULER_POLL_SECONDS_BUSY if started else self.config.SCHEDULER_POLL_SECONDS_IDLE
                logger.debug(
                    f"Scheduler cycle {cycle}: started {started} task(s); sleeping {sleep_for}s"
                )
                time.sleep(sleep_for)
            except Exception as e:
                logger.error(f"Scheduler loop error: {e}")
                time.sleep(30)
