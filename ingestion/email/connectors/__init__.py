"""Email connector implementations package.

This package provides modular email connectors for different email providers:

- EmailConnector: Abstract base class defining the connector interface
- IMAPConnector: IMAP server email retrieval with smart batching support
- ExchangeConnector: Exchange Web Services (EWS) email retrieval  
- GmailConnector: Gmail API email retrieval

All connectors return emails in a standardized canonical format using
email protocol field names (from_addr, to_addrs, date_utc, etc.).

Example Usage:
    from ingestion.email.connectors import IMAPConnector
    
    connector = IMAPConnector(
        host="imap.example.com",
        email_address="user@example.com", 
        password="password",
        mailbox="INBOX"
    )
    emails = connector.fetch_emails()
"""

from .base import EmailConnector
from .imap_connector import IMAPConnector  
from .exchange_connector import ExchangeConnector
from .gmail_connector import GmailConnector

__all__ = [
    "EmailConnector",
    "IMAPConnector", 
    "ExchangeConnector",
    "GmailConnector",
]
