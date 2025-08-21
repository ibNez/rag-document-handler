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
