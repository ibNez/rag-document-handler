"""URL manager for PostgreSQL-based URL metadata and processing."""

# Re-export the main class for backward compatibility
from .manager import PostgreSQLURLManager

__all__ = ['PostgreSQLURLManager']
