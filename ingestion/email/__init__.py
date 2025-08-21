"""Email ingestion connectors and orchestration utilities.

Modern email processing architecture:
- EmailOrchestrator: High-level account synchronization coordination
- EmailProcessor: Email content processing and embedding generation  
- EmailConnector variants: Protocol-specific email retrieval
- run_email_ingestion: CLI entry point for direct email processing
"""

from .connectors import EmailConnector, IMAPConnector, GmailConnector, ExchangeConnector
from .orchestrator import EmailOrchestrator
from .processor import EmailProcessor
from .ingest import run_email_ingestion

__all__ = [
    "EmailConnector",
    "IMAPConnector", 
    "GmailConnector",
    "ExchangeConnector",
    "EmailOrchestrator",
    "EmailProcessor",
    "run_email_ingestion",
]
