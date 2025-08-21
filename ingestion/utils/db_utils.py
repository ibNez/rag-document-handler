"""
Database utility functions for shared PostgreSQL operations.
"""

import logging
from contextlib import contextmanager
from typing import Any

logger = logging.getLogger(__name__)


class PostgreSQLManager:
    """Utility class for managing PostgreSQL connections."""

    def __init__(self, connection_pool: Any) -> None:
        self.connection_pool = connection_pool

    @contextmanager
    def get_connection(self):
        """Provide a connection from the pool."""
        conn = self.connection_pool.getconn()
        try:
            yield conn
        except Exception as e:
            logger.error("Database connection error: %s", e)
            raise
        finally:
            self.connection_pool.putconn(conn)
