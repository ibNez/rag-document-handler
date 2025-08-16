"""Email ingestion connectors and orchestration utilities."""

from .connector import EmailConnector, IMAPConnector
from .orchestrator import EmailOrchestrator

__all__ = ["EmailConnector", "IMAPConnector", "EmailOrchestrator"]
