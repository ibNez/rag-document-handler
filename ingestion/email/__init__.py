"""Email ingestion connectors and orchestration utilities."""

from .connector import EmailConnector, IMAPConnector
from .orchestrator import EmailOrchestrator
from .email_manager import EmailManager

__all__ = [
    "EmailConnector",
    "IMAPConnector",
    "EmailOrchestrator",
    "EmailManager",
]
