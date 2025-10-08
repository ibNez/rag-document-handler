"""Base abstract email connector class.

This module provides the abstract :class:`EmailConnector` base class that defines
the interface all email connectors must implement.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Dict, List, Optional


class EmailConnector(ABC):
    """Abstract base class for fetching email records."""

    @abstractmethod
    def fetch_emails(self, since_date: Optional[datetime] = None) -> List[Dict[str, Any]]:
        """Fetch emails and return a list of canonical records.

        Parameters
        ----------
        since_date:
            If provided, only messages on or after this date are retrieved.
        """

    @abstractmethod
    def get_available_mailboxes(self) -> List[Dict[str, Any]]:
        """Get available mailboxes/folders that can be used for email sync.
        
        Returns
        -------
        List[Dict[str, Any]]
            List of mailbox dictionaries with keys:
            - id: Unique identifier for the mailbox
            - name: Display name for the mailbox  
            - type: Type of mailbox (e.g., 'inbox', 'sent', 'folder', 'label')
            - messageCount: Number of messages in mailbox (optional)
            - selectable: Whether this mailbox can be selected for sync
        """
