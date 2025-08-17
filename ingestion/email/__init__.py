"""Email ingestion connectors and orchestration utilities."""

from .connector import EmailConnector, IMAPConnector, GmailConnector
from .orchestrator import EmailOrchestrator
from .email_manager import EmailManager
from .account_manager import EmailAccountManager
from .ingest import run_email_ingestion

__all__ = [
    "EmailConnector",
    "IMAPConnector",
    "GmailConnector",
    "EmailOrchestrator",
    "EmailManager",
    "EmailAccountManager",
    "run_email_ingestion",
]
