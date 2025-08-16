"""Email ingestion connectors and orchestration utilities."""

from .connector import EmailConnector, IMAPConnector
from .orchestrator import EmailOrchestrator
from .email_manager import EmailManager
from .ingest import run_email_ingestion

__all__ = [
    "EmailConnector",
    "IMAPConnector",
    "EmailOrchestrator",
    "EmailManager",
    "run_email_ingestion",
]
