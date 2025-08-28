"""
Async HTTP client for web crawling with proper error handling and logging.

Following DEVELOPMENT_RULES.md for all development requirements.
"""

import asyncio
import logging
from typing import Optional, Dict, Any
from contextlib import asynccontextmanager
import httpx

from .crawler_config import get_config

logger = logging.getLogger(__name__)


class AsyncHttpClient:
    """
    Async HTTP client wrapper with proper error handling and configuration.
    
    Provides a clean interface for making HTTP requests with consistent
    User-Agent headers, timeouts, and error handling.
    """
    
    def __init__(self, config: Optional[Any] = None):
        """
        Initialize the async HTTP client.
        
        Args:
            config: Optional crawler configuration. If None, uses global config.
        """
        self.config = config or get_config()
        self._client: Optional[httpx.AsyncClient] = None
        self._session_headers = self.config.get_user_agent_header()
        
    async def __aenter__(self) -> 'AsyncHttpClient':
        """Async context manager entry."""
        await self._ensure_client()
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self._client:
            await self._client.aclose()
            self._client = None
            
    async def _ensure_client(self) -> None:
        """Ensure the HTTP client is initialized."""
        if self._client is None:
            self._client = httpx.AsyncClient(
                headers=self._session_headers,
                timeout=httpx.Timeout(self.config.default_http_timeout),
                follow_redirects=True,
                limits=httpx.Limits(max_keepalive_connections=10, max_connections=20)
            )
            logger.debug("Initialized async HTTP client")
    
    async def get(self, url: str, timeout: Optional[float] = None, **kwargs) -> httpx.Response:
        """
        Make an async GET request.
        
        Args:
            url: The URL to request
            timeout: Optional timeout override
            **kwargs: Additional arguments passed to httpx.get()
            
        Returns:
            httpx.Response: The response object
            
        Raises:
            httpx.HTTPError: For HTTP-related errors
            asyncio.TimeoutError: For timeout errors
        """
        await self._ensure_client()
        assert self._client is not None  # For type checker
        
        effective_timeout = timeout or self.config.default_http_timeout
        
        try:
            logger.debug(f"GET request to {url} (timeout: {effective_timeout}s)")
            
            response = await self._client.get(
                url,
                timeout=effective_timeout,
                **kwargs
            )
            
            logger.debug(f"GET {url} -> {response.status_code} ({len(response.content)} bytes)")
            return response
            
        except httpx.TimeoutException as e:
            logger.warning(f"Timeout error for GET {url}: {e}")
            raise asyncio.TimeoutError(f"Request to {url} timed out after {effective_timeout}s") from e
            
        except httpx.HTTPError as e:
            logger.warning(f"HTTP error for GET {url}: {e}")
            raise
            
        except Exception as e:
            logger.error(f"Unexpected error for GET {url}: {e}")
            raise
    
    async def head(self, url: str, timeout: Optional[float] = None, **kwargs) -> httpx.Response:
        """
        Make an async HEAD request.
        
        Args:
            url: The URL to request
            timeout: Optional timeout override
            **kwargs: Additional arguments passed to httpx.head()
            
        Returns:
            httpx.Response: The response object
            
        Raises:
            httpx.HTTPError: For HTTP-related errors
            asyncio.TimeoutError: For timeout errors
        """
        await self._ensure_client()
        assert self._client is not None  # For type checker
        
        effective_timeout = timeout or self.config.robots_fetch_timeout
        
        try:
            logger.debug(f"HEAD request to {url} (timeout: {effective_timeout}s)")
            
            response = await self._client.head(
                url,
                timeout=effective_timeout,
                **kwargs
            )
            
            logger.debug(f"HEAD {url} -> {response.status_code}")
            return response
            
        except httpx.TimeoutException as e:
            logger.warning(f"Timeout error for HEAD {url}: {e}")
            raise asyncio.TimeoutError(f"Request to {url} timed out after {effective_timeout}s") from e
            
        except httpx.HTTPError as e:
            logger.warning(f"HTTP error for HEAD {url}: {e}")
            raise
            
        except Exception as e:
            logger.error(f"Unexpected error for HEAD {url}: {e}")
            raise
    
    async def get_robots_txt(self, origin_url: str) -> tuple[str, int]:
        """
        Fetch robots.txt for a given origin.
        
        Args:
            origin_url: The origin URL (scheme://host:port)
            
        Returns:
            tuple: (robots.txt content, HTTP status code)
        """
        robots_url = origin_url.rstrip("/") + "/robots.txt"
        
        try:
            response = await self.get(
                robots_url,
                timeout=self.config.robots_fetch_timeout
            )
            
            content = response.text if response.status_code == 200 else ""
            logger.info(f"Fetched robots.txt from {robots_url} -> {response.status_code} ({len(content)} chars)")
            
            return content, response.status_code
            
        except (httpx.HTTPError, asyncio.TimeoutError) as e:
            logger.warning(f"Failed to fetch robots.txt from {robots_url}: {e}")
            return "", 0
            
        except Exception as e:
            logger.error(f"Unexpected error fetching robots.txt from {robots_url}: {e}")
            return "", 0
    
    def is_rate_limited(self, response: httpx.Response) -> bool:
        """
        Check if a response indicates rate limiting.
        
        Args:
            response: The HTTP response to check
            
        Returns:
            bool: True if the response indicates rate limiting
        """
        return response.status_code in (429, 503)
    
    def get_retry_after(self, response: httpx.Response) -> Optional[float]:
        """
        Extract retry-after delay from response headers.
        
        Args:
            response: The HTTP response to check
            
        Returns:
            Optional[float]: Retry delay in seconds, or None if not specified
        """
        retry_after = response.headers.get("retry-after")
        
        if not retry_after:
            return None
            
        try:
            # Try to parse as seconds
            return float(retry_after)
        except ValueError:
            # Could be HTTP date format, but we'll just return None for simplicity
            logger.warning(f"Could not parse Retry-After header: {retry_after}")
            return None
    
    async def close(self) -> None:
        """Close the HTTP client explicitly."""
        if self._client:
            await self._client.aclose()
            self._client = None
            logger.debug("Closed async HTTP client")


@asynccontextmanager
async def http_client(config: Optional[Any] = None):
    """
    Async context manager for HTTP client.
    
    Args:
        config: Optional crawler configuration
        
    Yields:
        AsyncHttpClient: The HTTP client instance
    """
    client = AsyncHttpClient(config)
    try:
        yield client
    finally:
        await client.close()
